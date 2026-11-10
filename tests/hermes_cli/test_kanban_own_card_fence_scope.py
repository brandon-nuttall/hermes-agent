"""Own-card scope of the re-stamped worker fence (#152639).

A dispatcher-spawned Kanban worker's own tool env is re-stamped per command with the
delegate-child fence marker (``delegated_child_subprocess_env`` gate on
``HERMES_KANBAN_TASK`` → ``scrub_kanban_env``). Before #152639 that re-stamp fenced the
worker's own lineage from EVERY board mutation, including lifecycle verbs on the card
the dispatcher itself bound it to — while the same worker's in-process ``kanban_*``
tools still exercised exactly that scope.

Two-mode suite (run the same file against base and patched trees):
  PATCHED — all legs green.
  BASELINE (unpatched tree) — RED must be exactly the defect legs:
      test_worker_restamp_marker_carries_own_scope
      test_compound_marker_path_fence_parity   (base cannot parse the compound form)
      test_own_card_comment_cli_passes_under_own_scope
      test_own_card_claim_cli_reaches_db_verdict
  every control leg (foreign cards, bare/legacy markers, direct importers,
  orchestrator verbs, in-process child lineage) is green in BOTH modes — the
  controls prove the suite exercises the fence, not the assertion text.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]

# Import-pin: the suite must never silently run against an installed engine tree
# (namespace-package fallthrough; the image venv's .pth reaches /opt/hermes).
import hermes_cli.kanban_db as kb

assert Path(kb.__file__).resolve().is_relative_to(ROOT), (
    f"import pin: hermes_cli resolved to {kb.__file__}, expected under {ROOT}"
)

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    delegated_child_subprocess_env,
    kanban_path_is_fenced,
)
from hermes_cli import kanban as kanban_cli
from hermes_cli.kanban_db_connect import connect

FENCE_MSG = "delegate_task child contexts"


class OwnCardFenceScopeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db = self.root / "kanban.db"
        env = {
            "HERMES_HOME": str(self.root), "HOME": str(self.root),
            "USERPROFILE": str(self.root),
        }
        for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
                    "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", DELEGATED_CHILD_ENV_MARKER):
            env[key] = ""
        self._patcher = mock.patch.dict(os.environ, env, clear=False)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmp.cleanup)

        conn = connect(self.db)
        self.own = kb.create_task(conn, title="own card", assignee="worker")
        self.foreign = kb.create_task(conn, title="foreign card", assignee="other")
        for tid in (self.own, self.foreign):
            kb.claim_task(conn, tid)
        conn.close()

    # --- env seam -------------------------------------------------------------

    def _restamped_worker_env(self) -> dict[str, str]:
        """The worker's REAL tool-env shape: dispatcher identity keys, no marker, run
        through the actual per-command re-stamp seam."""
        worker = {
            **os.environ,
            "HERMES_KANBAN_TASK": self.own, "HERMES_KANBAN_RUN_ID": "1",
            "HERMES_KANBAN_CLAIM_LOCK": "host:1", "HERMES_KANBAN_DB": str(self.db),
        }
        worker.pop(DELEGATED_CHILD_ENV_MARKER, None)
        return delegated_child_subprocess_env(worker)

    def test_worker_restamp_marker_carries_own_scope(self):
        """DEFECT leg: the re-stamp must carry the dispatcher's one-task grant."""
        env = self._restamped_worker_env()
        marker = env[DELEGATED_CHILD_ENV_MARKER]
        self.assertTrue(marker, "re-stamp must keep a fence marker at all")
        self.assertEqual(marker, f"{kb.kanban_home()}::own={self.own}",
                         "worker-lineage re-stamp must carry the own-task scope (#152639)")
        for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK"):
            self.assertNotIn(key, env, "identity scrub itself must not regress")
        self.assertEqual(env["HERMES_KANBAN_DB"], str(self.db),
                         "board location must survive the scrub")

    def test_inprocess_child_restamp_gets_no_scope(self):
        """CONTROL: an in-process delegate child sharing the worker env is NOT the
        worker — its re-stamp must stay a bare fence with no own-card grant."""
        with mock.patch.dict(os.environ, {"HERMES_KANBAN_TASK": self.own, "HERMES_KANBAN_DB": str(self.db)}, clear=False):
            with delegated_child_context("child-session"):
                os.environ.pop(DELEGATED_CHILD_ENV_MARKER, None)
                env = delegated_child_subprocess_env(dict(os.environ))
        self.assertNotIn("::own=", env[DELEGATED_CHILD_ENV_MARKER])

    def test_compound_marker_path_fence_parity(self):
        """CONTROL: the compound marker fences exactly what the bare root fenced."""
        with mock.patch.dict(os.environ, {
            DELEGATED_CHILD_ENV_MARKER: f"{self.root}::own={self.own}",
            "HERMES_KANBAN_DB": str(self.db),
        }):
            self.assertTrue(kanban_path_is_fenced(self.db))
            self.assertTrue(kanban_path_is_fenced(self.root / "kanban" / "boards" / "x" / "kanban.db"))
            self.assertFalse(kanban_path_is_fenced(Path(tempfile.mkdtemp()) / "other.db"))

    # --- CLI end-to-end -------------------------------------------------------

    def _cli(self, env: dict[str, str], *args: str, expect_error: bool = False) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
            cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=90, check=False,
        )
        # belt and braces: the CLI subprocess must resolve the tree under test too
        self.assertNotIn("/opt/hermes/", proc.stderr, proc.stderr)
        return proc

    def _subprocess_env(self, marker: str | None = None, base: dict | None = None) -> dict[str, str]:
        src = base if base is not None else os.environ
        env = {k: v for k, v in src.items() if not k.startswith("HERMES_KANBAN") and k != DELEGATED_CHILD_ENV_MARKER}
        env.update({"HERMES_HOME": str(self.root), "HOME": str(self.root),
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "PYTHONIOENCODING": "utf-8"})
        env.pop("HERMES_CONFIG", None)
        if marker is not None:
            env[DELEGATED_CHILD_ENV_MARKER] = marker
        elif src.get(DELEGATED_CHILD_ENV_MARKER):
            env[DELEGATED_CHILD_ENV_MARKER] = src[DELEGATED_CHILD_ENV_MARKER]
        return env

    def test_own_card_comment_cli_passes_under_own_scope(self):
        """DEFECT leg: a re-stamped worker lineage may comment on its OWN bound card."""
        env = self._subprocess_env(base=self._restamped_worker_env())
        proc = self._cli(env, "comment", self.own, "handoff note from the worker")
        self.assertNotIn(FENCE_MSG, proc.stderr, f"own-card verb fenced (#152639): {proc.stderr}")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = sqlite3.connect(self.db).execute(
            "SELECT body FROM task_comments WHERE task_id = ?", (self.own,)).fetchall()
        self.assertTrue(any("handoff note" in r[0] for r in rows), rows)

    def test_own_card_claim_cli_reaches_db_verdict(self):
        """DEFECT leg: own-card claim must reach the genuine DB decision, not the fence.
        (The fixture card is already claimed+running, so the honest verdict is a
        refusal mentioning the live claim — not the delegate-child message.)"""
        env = self._subprocess_env(base=self._restamped_worker_env())
        proc = self._cli(env, "claim", self.own)
        self.assertNotIn(FENCE_MSG, proc.stderr, f"own-card verb fenced (#152639): {proc.stderr}")

    def test_foreign_card_comment_still_refused(self):
        """CONTROL: the own-scope never leaks to another card."""
        env = self._subprocess_env(base=self._restamped_worker_env())
        proc = self._cli(env, "comment", self.foreign, "must refuse")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(FENCE_MSG, proc.stderr)
        rows = sqlite3.connect(self.db).execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (self.foreign,)).fetchone()[0]
        self.assertEqual(rows, 0)

    def test_bare_marker_lineage_never_exempt(self):
        """CONTROL: the delegate-child lineage (bare-root marker) keeps the blanket denial."""
        env = self._subprocess_env(marker=str(kb.kanban_home()))
        proc = self._cli(env, "comment", self.own, "must refuse")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(FENCE_MSG, proc.stderr)

    def test_legacy_one_marker_never_exempt(self):
        """CONTROL: the pre-path ``"1"`` marker keeps fencing everything."""
        env = self._subprocess_env(marker="1")
        proc = self._cli(env, "comment", self.own, "must refuse")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(FENCE_MSG, proc.stderr)

    def test_create_never_exempt_even_under_own_scope(self):
        """CONTROL: orchestrator verbs stay fenced for the worker lineage (create names
        no own task and would mint board state)."""
        env = self._subprocess_env(base=self._restamped_worker_env())
        proc = self._cli(env, "create", "nope", "--assignee", "worker")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(FENCE_MSG, proc.stderr)

    def test_direct_importer_never_exempt(self):
        """CONTROL: the durable fence fails closed for a mutator that skips the CLI
        entry (the fast-fail exemption is UX; the ContextVar scope is CLI-only)."""
        env = self._subprocess_env(base=self._restamped_worker_env())
        script = "\n".join([
            f"import sys; sys.path.insert(0, {str(ROOT)!r})",
            "import hermes_cli.kanban_db as kb",
            "from hermes_cli.kanban_db_connect import connect",
            "conn = connect(kb.kanban_db_path())",
            "try:",
            f"    kb.add_comment(conn, {self.own!r}, author='direct-import', body='direct import mutation')",
            "    print('MUTATED')",
            "except PermissionError as exc:",
            "    print('REFUSED:', exc)",
        ])
        proc = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=90, check=False)
        self.assertIn("REFUSED:", proc.stdout, f"{proc.stdout}\n{proc.stderr}")
        self.assertNotIn("MUTATED", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
