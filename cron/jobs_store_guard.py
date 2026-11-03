"""Save-path guards for the cron job store: create- and delete-direction protection.

Two symmetric halves of one contract — a save whose payload was assembled from a snapshot the
disk has since moved past must lose nothing that still exists, and must not restore what was
intentionally removed:

* **#80624 (create direction):** on-disk jobs missing from a stale save payload are recovered
  back into it and the event logged, so a writer running under a degraded (flock-timeout) lock
  cannot clobber concurrent creates.
* **#148437 (delete direction):** a job removed out-of-band by another process (CLI ``cron
  remove``, a sweep in another tick) used to be re-added silently — byte-identical, unlogged —
  by the next save from a writer whose payload predated the removal. ``removed_ids`` was known
  only to the *removing* process's own save; every other writer re-flushed the row. Now every
  save that lands with ``removed_ids`` persists a timestamped tombstone to ``jobs.removed.json``
  beside ``.jobs.lock``, and every merging save drops tombstoned payload rows instead of
  resurrecting them — with a log line, because the silence was half the defect.

The sidecar is not a ``jobs.json`` meta key on purpose: ``replace=True`` repairs and external
whole-file rewrites must not be able to erase the delete intent that protects the store from
them. A re-create is never blocked either: the tombstone also records the content-identity hash
of every removed record, and only a row that replays one of those hashes — or cannot show a
parseable ``created_at`` — is dropped. A modified row (a replacement record, any new intent) is
kept; the incident shape was the flush coming back byte-identical.

This module is jobs.py's internal save-path seam; the facade imports these names at module
level and production reads them from the facade. Store paths, ``_secure_file`` and the lock
state are read off the facade at call time (late import — never a module-level cycle) so
facade-level test seams keep working here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Optional

from utils import atomic_write_text

logger = logging.getLogger(__name__)

# Outlives any writer payload worth defending against (the incident window was ~24h) while
# staying far shorter than the life of the ids themselves, so the sidecar stays bounded.
DELETE_TOMBSTONE_TTL_SECONDS = 7 * 24 * 3600


def _jobs():
    """Late facade import (sibling rule): every store/lock helper below is read from
    ``cron.jobs`` at call time."""
    from cron import jobs
    return jobs


def _delete_tombstones_file() -> Path:
    """Sidecar path for delete intent, beside ``.jobs.lock`` in the active store's cron dir."""
    return _jobs()._current_cron_store().cron_dir / "jobs.removed.json"


def _job_identity_hash(job: dict[str, Any]) -> str:
    """Content identity of a job record: stable-hash of its canonical JSON. The incident
    resurrection was byte-identical, so exact replay of a removed record is provable from
    content alone — the strongest possible keep/drop discriminator."""
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode("utf-8")).hexdigest()


def _has_parseable_created_at(job: dict[str, Any]) -> bool:
    """Whether the row can date itself. A tombstoned row that cannot is indistinguishable from
    a replay of the removed record (missing/garbage created_at never accompanies real new
    intent), so it loses against the tombstone."""
    created = job.get("created_at")
    if not isinstance(created, str):
        return False
    try:
        datetime.fromisoformat(created)
        return True
    except ValueError:
        return False


def _fresh_delete_tombstones() -> dict[str, dict[str, Any]]:
    """Fresh ``{job_id: {"at": removed_at_epoch, "identities": [content_hash, ...]}}`` from the
    sidecar. Missing file => ``{}``; unreadable or malformed content is treated as no tombstones
    (fail open): the guard only ever strengthens delete intent, and a broken sidecar must not
    freeze every save. Entries older than the TTL or dated in the future (clock jump) are
    ignored; expired entries are pruned on the next write."""
    try:
        data = json.loads(_delete_tombstones_file().read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable cron delete-tombstone sidecar", exc_info=True)
        return {}
    entries = data.get("tombstones") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    now_ts = time.time()
    fresh: dict[str, dict[str, Any]] = {}
    for job_id, entry in entries.items():
        if not isinstance(job_id, str) or not isinstance(entry, dict):
            continue
        removed_at = entry.get("at")
        if (
            isinstance(removed_at, (int, float)) and not isinstance(removed_at, bool)
            and 0.0 <= now_ts - float(removed_at) <= DELETE_TOMBSTONE_TTL_SECONDS
        ):
            identities = entry.get("identities")
            fresh[job_id] = {
                "at": float(removed_at),
                "identities": [i for i in identities if isinstance(i, str)]
                if isinstance(identities, list) else [],
            }
    return fresh


def _record_delete_tombstones(
    removed_ids: Collection[str], removed_rows: dict[str, dict[str, Any]]
) -> None:
    """Persist delete intent for a save that has just landed. Called under the jobs lock AFTER
    jobs.json is durable, so no tombstone can ever stand for a delete that never reached disk
    (that shape would silently drop a live job at the next save). Best-effort: losing this write
    loses only the resurrection guard, never the delete that already succeeded. Re-removing an
    id accumulates its removed-record identities (bounded), so every replay shape is provable.
    *removed_rows* maps ids to the records as they were on disk BEFORE the save (the removing
    payload by definition no longer holds them)."""
    fresh = {str(job_id) for job_id in removed_ids if job_id}
    if not fresh:
        return
    now_ts = time.time()
    keep = {j: e for j, e in _fresh_delete_tombstones().items() if j not in fresh}
    by_id = removed_rows
    for job_id in fresh:
        entry = keep.get(job_id)
        if entry is None:
            entry = keep[job_id] = {"at": now_ts, "identities": []}
        else:
            entry["at"] = now_ts
        job = by_id.get(job_id)
        if job is not None:
            digest = _job_identity_hash(job)
            if digest not in entry["identities"]:
                entry["identities"] = (entry["identities"] + [digest])[-8:]
    try:
        tombstones_file = _delete_tombstones_file()
        atomic_write_text(tombstones_file, json.dumps({"tombstones": keep}))
        _jobs()._secure_file(tombstones_file)
    except OSError as exc:
        logger.warning("Failed to persist cron delete tombstones for %s: %s", sorted(fresh), exc)


def _drop_tombstoned_payload_jobs(
    jobs: list[dict[str, Any]], tombstones: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split *jobs* against fresh *tombstones*: a row leaves the payload only when it replays
    the removed record (content hash matches a stored identity) or cannot date itself — either
    shape is indistinguishable from the byte-identical resurrection. Any other row (a
    replacement record, a re-create) is kept: only the replay is provably the bug. Returns
    ``(kept, dropped_ids)``; never mutates *jobs*."""
    if not tombstones:
        return jobs, []
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for job in jobs:
        job_id = str(job["id"]) if isinstance(job, dict) and job.get("id") else ""
        entry = tombstones.get(job_id)
        if entry is not None and (
                _job_identity_hash(job) in entry["identities"]
                or not _has_parseable_created_at(job)):
            dropped.append(job_id)
            continue
        kept.append(job)
    return kept, dropped


def _unmerged_disk_jobs(
    jobs: list[dict[str, Any]], removed_ids: Optional[Collection[str]],
    tombstones: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """On-disk jobs missing from *jobs* and not intentionally removed. Stamp match => nothing
    landed, return without parsing; unreadable store => raise RuntimeError (fail closed: writing
    over an unknown baseline would silently drop every job in it). *tombstones* (fresh ids
    deleted out-of-band) count as intentional: a tombstoned row still on disk — the crash shape
    where the delete never persisted — is not "recovered", and the verify-after-stage loop never
    spins on it. ``None`` loads the sidecar fresh."""
    jobs_mod = _jobs()
    stamp = getattr(jobs_mod._jobs_lock_state, "load_stamp", None)
    if stamp is not None and jobs_mod._jobs_file_stamp(
            jobs_mod._current_cron_store().jobs_file) == stamp:
        return []
    if tombstones is None:
        tombstones = _fresh_delete_tombstones()
    disk_jobs = jobs_mod._peek_jobs_unlocked()
    if disk_jobs is None:
        raise RuntimeError(
            f"Cron database corrupted; refusing to overwrite "
            f"{jobs_mod._current_cron_store().jobs_file}")
    seen = {str(j["id"]) for j in jobs if isinstance(j, dict) and j.get("id")}
    seen |= {str(i) for i in (removed_ids or ()) if i}
    seen |= tombstones.keys()
    recovered: list[dict[str, Any]] = []
    for disk_job in disk_jobs:
        if not isinstance(disk_job, dict) or not disk_job.get("id"):
            continue
        disk_id = str(disk_job["id"])
        if disk_id not in seen:
            recovered.append(disk_job)
            seen.add(disk_id)
    return recovered


def _merge_unexpected_disk_jobs(
    jobs: list[dict[str, Any]], *, removed_ids: Optional[Collection[str]] = None,
    tombstones: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """*jobs* plus on-disk jobs absent from the payload (under the degraded flock-timeout path a
    stale writer would otherwise clobber concurrent creates), minus payload rows a concurrent
    process already deleted (#80624 protects the create direction; #148437 protects the delete
    direction). Deletes pass ``removed_ids``; never mutates *jobs*."""
    if tombstones is None:
        tombstones = _fresh_delete_tombstones()
    recovered = _unmerged_disk_jobs(jobs, removed_ids, tombstones)
    if recovered:
        logger.warning(
            "Preserved %d cron job(s) present on disk but missing from the "
            "in-memory save payload (concurrent create under degraded lock "
            "or stale writer) (#80624): %s",
            len(recovered), [j.get("id") for j in recovered])
        jobs = jobs + recovered
    jobs, dropped = _drop_tombstoned_payload_jobs(jobs, tombstones)
    if dropped:
        logger.warning(
            "Dropped %d cron job(s) removed out-of-band from the in-memory save payload "
            "(delete tombstone; only replays of the removed record - matching content "
            "hash or no parseable created_at - are dropped) (#148437): %s",
            len(dropped), dropped)
    return jobs
