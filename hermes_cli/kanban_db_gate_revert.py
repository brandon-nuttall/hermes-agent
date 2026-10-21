"""Verify-after-write for kanban status transitions: a board gate that demotes
in-transaction must never be reported as success.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via the facade so monkeypatching ``kanban_db.<name>`` keeps working.

A board gatekeeper implemented as a SQLite AFTER UPDATE trigger (e.g. an
operator-gate quarantine guard) rewrites the row back to its prior status INSIDE
the verb's own write_txn, after ``rowcount`` has already reported success. The
trigger must never RAISE — ``recompute_ready`` promotes siblings in one shared
write_txn and an abort would roll them back and crash the dispatcher tick — so
the verbs themselves verify after writing. Raising :class:`_TransitionReverted`
inside the txn rolls the refused transition back entirely (retry budget
included); the caller then records a ``transition_reverted`` audit event and
returns failure. Trust the row, never ``rowcount``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import sqlite3


class _TransitionReverted(Exception):
    """Internal signal: a status UPDATE was reverted inside the same transaction."""

    def __init__(self, task_id: str, verb: str, expected: Optional[str], actual: Optional[str]):
        self.task_id, self.verb = task_id, verb
        self.expected, self.actual = expected, actual
        super().__init__(
            f"{task_id}: {verb} to {expected!r} was reverted in-transaction "
            f"to {actual!r} by a database trigger"
        )


def _assert_status_landed(
    conn: sqlite3.Connection, task_id: str, expected: str, *, verb: str,
) -> None:
    """Verify after writing: the row must carry ``expected`` now.

    Read back INSIDE the caller's write_txn — the AFTER UPDATE trigger has already
    run (it fires synchronously with the UPDATE) and the IMMEDIATE lock excludes
    any other writer, so the verdict is race-free.
    """
    from hermes_cli.kanban_db import _task_status

    actual = _task_status(conn, task_id)
    if actual != expected:
        raise _TransitionReverted(task_id, verb, expected, actual)


def record_reverted_transition(
    conn: sqlite3.Connection, reverted: _TransitionReverted,
) -> str:
    """Audit a refused transition after its write_txn rolled back, and return the
    honest failure text the verb should hand its caller."""
    from hermes_cli.kanban_db import _append_event, write_txn

    with write_txn(conn):
        _append_event(
            conn, reverted.task_id, "transition_reverted",
            {"verb": reverted.verb, "expected_status": reverted.expected,
             "actual_status": reverted.actual},
        )
    return (
        f"{reverted.task_id}: {reverted.verb} to {reverted.expected!r} was reverted "
        f"in-transaction to {reverted.actual!r} by a database trigger (a board gate "
        f"refused the transition); the row is unchanged — see the transition_reverted event"
    )
