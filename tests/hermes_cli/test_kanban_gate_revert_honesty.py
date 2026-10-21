"""Kernel honesty: an in-transaction demote must not report as success.

A board gatekeeper implemented as a SQLite AFTER UPDATE trigger (the operator-gate
quarantine guard) rewrites ``status`` back to ``blocked`` INSIDE the caller's own
write_txn, after the verb has already computed ``rowcount == 1``. The kernel verbs
returned success over a row that never moved — every surface (bare CLI, kanban
toolset, dashboard) told the operator "Unblocked t_x" while the stored status stayed
``blocked``. A RAISE in the trigger is not an option: ``recompute_ready`` promotes
siblings in one shared write_txn and an abort would roll them back and crash the
dispatcher tick. So the kernel itself must verify after writing: the verb's verdict
comes from the ROW, never from ``rowcount``, and a refused transition commits nothing.

The trigger installed by these fixtures is a minimal stand-in for the estate's
``gate_guard_quarantine`` — same shape (AFTER UPDATE OF status, demote in-body),
which is exactly the contract the verbs must survive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


# Mirrors the estate's gate_guard_quarantine: demotes promotions of gate-marker
# cards back to blocked inside the caller's transaction. Recursive triggers are
# off by default, so the inner UPDATE does not re-fire the WHEN clause.
GATE_TRIGGER = """
CREATE TRIGGER test_gate_quarantine
AFTER UPDATE OF status ON tasks
WHEN NEW.status = 'ready' AND lower(NEW.title) LIKE '%needs brandon%'
BEGIN
    UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' WHERE id = NEW.id;
    INSERT INTO task_events (task_id, run_id, kind, payload, created_at)
    VALUES (NEW.id, NULL, 'gate_quarantined', NULL, strftime('%s', 'now'));
END;
"""


def _install_gate(conn):
    conn.execute(GATE_TRIGGER)
    conn.commit()


def _blocked_card(conn, title, *, failures=2):
    """Blocked card with a spent-ish retry budget, set up deterministically."""
    tid = kb.create_task(conn, title=title, assignee="tester")
    conn.execute(
        "UPDATE tasks SET status = 'blocked', consecutive_failures = ? WHERE id = ?",
        (failures, tid),
    )
    conn.commit()
    return tid


def _events(conn, tid, kind):
    return conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = ?", (tid, kind)
    ).fetchall()


# --- promote_task ---------------------------------------------------------


def test_promote_on_gated_board_fails_honestly(conn):
    _install_gate(conn)
    tid = _blocked_card(conn, "Fix (needs brandon@neuron keystroke)")

    ok, err = kb.promote_task(conn, tid, actor="tester")

    assert not ok, "promote must not report success over a reverted row"
    assert err and "reverted" in err
    assert kb.get_task(conn, tid).status == "blocked"
    # No success event is left behind, and the revert is audited.
    assert not _events(conn, tid, "promoted_manual")
    assert _events(conn, tid, "transition_reverted")


def test_promote_ungated_still_succeeds_with_gate_installed(conn):
    _install_gate(conn)
    tid = _blocked_card(conn, "innocent title")

    ok, err = kb.promote_task(conn, tid, actor="tester")

    assert ok and err is None
    assert kb.get_task(conn, tid).status == "ready"
    assert _events(conn, tid, "promoted_manual")
    assert not _events(conn, tid, "transition_reverted")


# --- unblock_task ---------------------------------------------------------


def test_unblock_on_gated_board_fails_honestly_and_atomically(conn):
    _install_gate(conn)
    tid = _blocked_card(conn, "Deploy (needs brandon)")

    ok, err = kb.unblock_task(conn, tid, with_reason=True)

    assert not ok, "unblock must not report success over a reverted row"
    assert err and "reverted" in err
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    # Atomic refusal: the retry-budget reset and the success event must not survive.
    assert task.consecutive_failures == 2
    assert not _events(conn, tid, "unblocked")
    assert _events(conn, tid, "transition_reverted")


def test_unblock_returns_bool_by_default(conn):
    # Back-compat: every existing caller consumes a plain bool.
    _install_gate(conn)
    gated = _blocked_card(conn, "x (needs brandon)")
    clean = _blocked_card(conn, "y")

    assert kb.unblock_task(conn, gated) is False
    assert kb.unblock_task(conn, clean) is True


def test_unblock_ungated_still_succeeds_with_gate_installed(conn):
    _install_gate(conn)
    tid = _blocked_card(conn, "innocent title")

    assert kb.unblock_task(conn, tid) is True
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.get_task(conn, tid).consecutive_failures == 0
    assert _events(conn, tid, "unblocked")


# --- recompute_ready (shared dispatcher txn, never-abort contract) --------


def test_recompute_ready_does_not_count_or_claim_a_reverted_resume(conn):
    _install_gate(conn)
    # failures=0: below the breaker limit, so recompute_ready genuinely attempts
    # the auto-resume (the path the incident saw revert silently every tick).
    gated = _blocked_card(conn, "wake (needs brandon)", failures=0)
    clean = _blocked_card(conn, "wake (innocent)", failures=0)

    promoted = kb.recompute_ready(conn)

    # Only the row that actually landed counts.
    assert promoted == 1
    assert kb.get_task(conn, gated).status == "blocked"
    assert kb.get_task(conn, clean).status == "ready"
    assert not _events(conn, gated, "promoted")
    assert _events(conn, gated, "transition_reverted")
    assert _events(conn, clean, "promoted")
