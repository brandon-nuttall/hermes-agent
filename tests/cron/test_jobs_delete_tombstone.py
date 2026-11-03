"""Regression for #148437 — out-of-band removes must survive stale-writer flushes.

The #80624 shrink-merge guard protects the create direction (disk rows missing from a save
payload are recovered). The delete direction was unprotected: ``hermes cron remove`` deleted a
completed one-shot, and hours later a writer holding a pre-removal payload flushed and re-added
the row byte-identical, with no log line anywhere (measured on a 15-store fleet; the resurrected
wake row re-fired and cost two monitoring cards plus an RCA cycle).

The fix: saves that land with ``removed_ids`` persist delete intent as a timestamped tombstone
in ``cron/jobs.removed.json``, including the content-identity hash of the removed record;
merging saves drop payload rows that replay a removed record (hash match, or no parseable
``created_at``) — and log it — while any modified row (replacement record, re-create) is kept.
``replace=True`` keeps skipping both directions so disaster recovery stays a functioning
escape hatch.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return home


def _disk_ids(home: Path) -> set[str]:
    data = json.loads((home / "cron" / "jobs.json").read_text())
    return {str(j["id"]) for j in data.get("jobs", [])}


def _sidecar(home: Path) -> Path:
    return home / "cron" / "jobs.removed.json"


def _make_job(name: str, schedule: str = "2030-01-01T00:00:00Z") -> str:
    from cron.jobs import create_job

    return create_job(prompt="probe", schedule=schedule, name=name, deliver="local")["id"]


def test_stale_writer_cannot_resurrect_out_of_band_remove(hermes_env, caplog):
    """THE incident: writer B's payload predates a remove; B's flush must not undo it."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("wake-row")
    stale = load_jobs()  # writer B's snapshot, taken before the removal
    assert any(str(j["id"]) == jid for j in stale)

    assert remove_job(jid)
    assert jid not in _disk_ids(hermes_env)

    with caplog.at_level(logging.WARNING, logger="cron.jobs_store_guard"):
        save_jobs(stale)  # writer B flushes its stale copy

    assert jid not in _disk_ids(hermes_env), "out-of-band remove was silently resurrected"
    # Silence was half the defect: the drop must name the class and the ids.
    assert any("148437" in r.getMessage() and "Dropped" in r.getMessage() for r in caplog.records)


def test_repeated_stale_flushes_stay_healed(hermes_env):
    """A catch-up burst is several stale flushes; one tombstone read must outlast all of them."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("burst-row")
    stale = load_jobs()
    remove_job(jid)
    save_jobs(stale)
    save_jobs(stale)
    save_jobs(stale)
    assert jid not in _disk_ids(hermes_env)
    assert jid in json.loads(_sidecar(hermes_env).read_text())["tombstones"]


def test_recreate_with_same_id_and_later_created_at_lands(hermes_env):
    """The tombstone guards replays, not re-creates: a row of the same id with a later
    created_at is new intent and must land."""
    from cron.jobs import _hermes_now, load_jobs, remove_job, save_jobs

    jid = _make_job("recreated")
    stale = load_jobs()
    remove_job(jid)
    original = next(dict(j) for j in stale if str(j["id"]) == jid)
    original["created_at"] = _hermes_now().isoformat()
    save_jobs(load_jobs() + [original])
    assert jid in _disk_ids(hermes_env)


def test_replacement_record_with_same_created_at_is_not_dropped(hermes_env):
    """The tombstone must never launder a legitimate replacement of the removed id: another
    owner's record may inherit the old created_at (only content marks it new), so the guard
    drops provable replays — identical content or an undatable row — and keeps modified rows."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("replaced")
    original = next(j for j in load_jobs() if str(j["id"]) == jid)
    remove_job(jid)
    replacement = dict(original, name="second-owner")  # created_at deliberately inherited
    save_jobs(load_jobs() + [replacement])
    assert jid in _disk_ids(hermes_env)
    remaining = json.loads((hermes_env / "cron" / "jobs.json").read_text())["jobs"]
    assert next(j for j in remaining if str(j["id"]) == jid)["name"] == "second-owner"


def test_expired_tombstone_stops_suppressing(hermes_env):
    """The guard is transient: past the TTL the sidecar no longer drops rows (and a normal
    save prunes the entry)."""
    from cron import jobs_store_guard
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("old-wake")
    stale = load_jobs()
    remove_job(jid)
    expired = time.time() - jobs_store_guard.DELETE_TOMBSTONE_TTL_SECONDS - 3600
    _sidecar(hermes_env).write_text(
        json.dumps({"tombstones": {jid: {"at": expired, "identities": []}}}))
    save_jobs(stale)
    assert jid in _disk_ids(hermes_env)


def test_siblings_in_the_same_stale_payload_survive(hermes_env):
    """The guard drops exactly the tombstoned id — never anything else."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    doomed = _make_job("doomed")
    keeper = _make_job("keeper")
    stale = load_jobs()
    remove_job(doomed)
    save_jobs(stale)
    ids = _disk_ids(hermes_env)
    assert doomed not in ids and keeper in ids


def test_replace_save_remains_the_disaster_recovery_hatch(hermes_env):
    """A wholesale rewrite from a pre-removal backup must restore removed rows: replace=True
    skips tombstones both ways."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("restore-me")
    backup = load_jobs()
    remove_job(jid)
    save_jobs(backup, replace=True)
    assert jid in _disk_ids(hermes_env)


def test_corrupt_sidecar_fails_open_never_freezes_saves(hermes_env, caplog):
    """A broken tombstone file degrades to pre-fix behavior with a warning — it must never
    raise from the save path, and the delete lane must keep working on top of it."""
    from cron.jobs import load_jobs, remove_job, save_jobs

    jid = _make_job("junk-side")
    remove_job(jid)
    _sidecar(hermes_env).write_text("{not json!!")
    with caplog.at_level(logging.WARNING, logger="cron.jobs_store_guard"):
        save_jobs(load_jobs())  # must not raise
    assert any("tombstone" in r.getMessage().lower() for r in caplog.records)

    jid2 = _make_job("after-junk")
    payload = load_jobs()  # writer payload holding jid2
    remove_job(jid2)       # rewrites the sidecar fresh
    save_jobs(payload)     # stale flush of the just-removed row
    assert jid2 not in _disk_ids(hermes_env)


def test_sweep_style_removed_ids_save_also_tombstones(hermes_env):
    """The completed-one-shot retention sweep removes through the same save path; those
    removes are protected too (wake residue is a sweep removal)."""
    from cron.jobs import load_jobs, save_jobs

    jid = _make_job("swept")
    stale = load_jobs()
    after = [j for j in load_jobs() if str(j["id"]) != jid]
    save_jobs(after, removed_ids={jid})
    save_jobs(stale)
    assert jid not in _disk_ids(hermes_env)


def test_cross_process_remove_survives_stale_flush(hermes_env):
    """The remove happens in ANOTHER process under the jobs flock (the CLI shape); the parent
    writer's stale flush must still not resurrect the row."""
    from cron.jobs import load_jobs, save_jobs

    jid = _make_job("cross")
    stale = load_jobs()
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c",
         f"from cron.jobs import remove_job; print(remove_job({jid!r}))"],
        capture_output=True, text=True, env={
            "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
            "HERMES_HOME": str(hermes_env)},
        cwd=repo_root, timeout=120, check=False,
    )
    assert "True" in out.stdout, f"remove failed: {out.stdout} {out.stderr[-400:]}"
    save_jobs(stale)
    assert jid not in _disk_ids(hermes_env)
