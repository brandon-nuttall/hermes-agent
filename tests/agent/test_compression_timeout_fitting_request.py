"""Regression for #154404: a preflight compression timeout must not end a turn whose
request still fits the configured context window.

The pre-API compression pass summarises with the auxiliary compression model
(``auxiliary.compression``; when no auxiliary provider is configured it resolves to
the session's main model) — a stalled or saturated summariser is evidence about the
summariser, not about the request.  Only a provider-proven overflow (413) may end
the turn, or a rough count at/over the window, or an unknown window (the #98722
fail-closed stop, same contract as ``_fail_closed_after_preflight_timeout``, #98424/
#113646).  Otherwise the attempt is refunded and the turn continues UNCOMPACTED: the
timeout cooldown ladder already blocks re-compression on the next call, and a real
413 still lands on the overflow path, which fails closed.

The timeout's final response is also surface-aware: a dispatched kanban worker
cannot run ``/new`` — its retry is a FRESH session the dispatcher seeds — so for
worker runs the copy states what happened and what the next attempt must do
(re-read the card, verify on-disk state) instead of handing out a command only a
human can follow.  The text also lands in the run's ``last_output`` for the NEXT
attempt to read, so it must not describe an action the worker cannot take.
"""

from __future__ import annotations

import agent.turn_preflight as tp


def _mk_msgs():
    return [{"role": "user", "content": "work"}, {"role": "assistant", "content": "done"}]


class _Harness:
    """Stubs every side seam ``run_preflight_compression`` touches except the
    timeout branch itself — the compression pass reports a timeout with the
    snapshot unchanged, exactly the shape a stalled summariser produces."""

    def __init__(self, monkeypatch, window, *, request=20_000, overflow=False):
        self.request, self.overflow, self.window = request, overflow, window
        self.unchanged = _mk_msgs()
        for name, val in (
            ("_review_fork_first_request_pending", lambda a: False),
            ("ensure_compression_feasibility_checked", lambda a, t: None),
            ("automatic_compaction_status_message", lambda *a, **k: None),
            ("_refund_api_call", lambda a, n: n - 1),
            ("_clear_overflow_warn", lambda a: None),
            ("context_compression_timed_out", lambda a: True),
        ):
            monkeypatch.setattr(tp, name, val)

    def _agent(self):
        class A:
            compression_enabled = True
            model = "test-model"
            _maybe_grow = None

            def _emit_status(self, *a, **k):
                pass

            def _compress_context(self, messages, system_message, **kw):
                return messages, system_message  # timed out: unchanged snapshot

        return A()

    def _compressor(self):
        class C:
            threshold_tokens = int(self.window * 0.5) if self.window else 16_000
            context_length = self.window

            def should_compress(self, tokens):
                return True

            def get_active_compression_failure_cooldown(self):
                return None

        return C()

    def run(self):
        v = tp.PreflightGateVerdict(
            action="fallthrough", pending_moa_prepared_request=None,
            messages=list(self.unchanged), active_system_prompt="sys",
            conversation_history=[], api_call_count=3, compression_attempts=0,
            final_response=None, failed=False, _turn_exit_reason=None,
            _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
            _provider_overflow_recovery_pending=False, _last_preflight_pressure=None)
        return tp.run_preflight_compression(
            self._agent(), v, compressor=self._compressor(),
            request_pressure_tokens=self.request, provider_overflow_preflight=self.overflow,
            defer_preflight=lambda t: False, moa_prepared_request=None,
            system_message="sys", user_message="work", max_compression_attempts=3,
            effective_task_id="t_x")


def _run(monkeypatch, window, *, request=20_000, overflow=False):
    return _Harness(monkeypatch, window, request=request, overflow=overflow).run()


def test_timeout_below_window_continues_uncompacted(monkeypatch):
    """Threshold-triggered (not provider-proven) timeout with the request below the
    window: the turn continues instead of dying — the defect class of #154404."""
    v = _run(monkeypatch, window=32_000, request=20_000)
    assert v.action == "fallthrough"
    assert not v.failed
    assert v.final_response is None
    assert v.api_call_count == 2  # provider call refunded
    assert not v._compression_timeout_exhausted
    assert v._last_preflight_pressure is None  # pressure blocker disarmed


def test_timeout_at_window_still_fails_closed(monkeypatch):
    """Rough count AT the window: no headroom, so the #98722 stop still applies."""
    v = _run(monkeypatch, window=32_000, request=32_000)
    assert v.action == "break"
    assert v.failed
    assert v._turn_exit_reason == "context_compression_timeout"


def test_timeout_over_window_still_fails_closed(monkeypatch):
    """Rough count over the window: the request likely cannot fit; fail closed."""
    v = _run(monkeypatch, window=32_000, request=40_000)
    assert v.action == "break"
    assert v.failed
    assert v._turn_exit_reason == "context_compression_timeout"


def test_timeout_window_unknown_fails_closed(monkeypatch):
    """No context length: the conservative default — never send blind (#98424)."""
    v = _run(monkeypatch, window=0, request=20_000)
    assert v.action == "break"
    assert v.failed
    assert v._turn_exit_reason == "context_compression_timeout"


def test_timeout_provider_proven_overflow_still_fails_closed(monkeypatch):
    """A provider 413 IS proof the request cannot fit: fail closed, and surface the
    worker copy on a dispatched run — ``/new`` is not actionable there."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake_task")
    v = _run(monkeypatch, window=32_000, request=40_000, overflow=True)
    assert v.action == "break"
    assert v.failed
    assert v._turn_exit_reason == "context_compression_timeout"
    assert v._compression_timeout_exhausted
    assert "dispatched kanban worker" in v.final_response
    assert "Start a fresh session with /new" not in v.final_response


def test_timeout_final_response_switches_for_dispatched_worker(monkeypatch):
    """The timeout close is surface-aware: the human copy keeps the ``/new`` advice;
    the worker copy states the fresh-session retry instead."""
    from agent import conversation_loop as cl

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cl._compression_timeout_final_response() == cl._COMPRESSION_TIMEOUT_FINAL_RESPONSE

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fake")
    text = cl._compression_timeout_final_response()
    assert "kanban worker" in text
    assert "Start a fresh session with /new" not in text
    assert "re-read the card" in text
