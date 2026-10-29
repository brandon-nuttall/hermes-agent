"""A CLOSED session must never hold the canonical 'Bot Chat' title against cron delivery.

The estate's P1 page mirrors died ``not_found`` on every beat (fleet-wide since one gateway
restore): a TUI opened a session, took the canonical title, a served dashboard login stamped it
(``basic:<user>``), and it was closed — ``cli_close`` by hand, ``session_reset`` by the bounce.
Title resolution matched the dead row and the door kept it in place as "local history a native
open adopts" while ``adopt_legacy_session`` refused the delivery, so the profile's
``deliver: bot-chat`` jobs failed forever. #143715 made an unadoptable stamp displace the row;
this is the closed dimension of that class — rows shaped exactly like the production stores,
planted while the authority runs (pre-boot planting gets adopted by startup migration, which is
not the production sequence), delivered through the real cron producer ``_deliver_to_bot_chat``.
"""
from contextlib import closing, contextmanager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon


@contextmanager
def _live_profile(tmp_path, monkeypatch):
    """A real daemon over a temp profile home, with its model stub shut down on exit."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{model.server_port}/v1'
    try:
        (home / 'config.yaml').write_text(json.dumps({
            'gateway': {'multiplex_profiles': False},
            'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': url},
            'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}),
            encoding='utf-8')
        env = child_env()
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                   OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
        monkeypatch.setenv('HERMES_HOME', str(home))
        monkeypatch.setattr(Path, 'home', lambda: user)
        with daemon(root, home, env, barrier=False):
            yield home
    finally:
        model.shutdown()
        model.server_close()


def _plant_closed_title_holder(home, sid, *, source, stamp, route_column=False, end_reason='cli_close'):
    """Plant the dead title holder while the authority runs (the production sequence: the chat
    was taken and closed mid-flight, not before the store's first boot)."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(sid, source)
    db.set_session_title(sid, 'Bot Chat')
    sets, params = [], []
    if stamp:
        sets.append('user_id=?'); params.append(stamp)
    if route_column:
        sets.append('chat_id=?'); params.append(sid)  # a route column left by the old boot, never bound
    if sets:
        db._execute_write(lambda conn: conn.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", (*params, sid)))
    db.end_session(sid, end_reason)
    db.close()


def _assert_page_landed_in_a_fresh_chat(home, dead_sid, payload):
    from cron.scheduler_delivery import _deliver_to_bot_chat
    job = {'id': 'page', 'name': 'Page', 'execution_id': 'run-1'}
    error = _deliver_to_bot_chat(job, payload, '')
    receipt = job['_bot_chat_delivery_receipts']['bot-chat:(own)']
    assert receipt['status'] in ('queued', 'claimed', 'settled'), (error, receipt)
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as store:
        fresh = store.execute(
            "SELECT id FROM sessions WHERE title='Bot Chat' AND ended_at IS NULL").fetchall()
        assert len(fresh) == 1 and fresh[0][0] != dead_sid, fresh
        held = store.execute('SELECT title, ended_at FROM sessions WHERE id=?', (dead_sid,)).fetchone()
        assert held[0].startswith('Bot Chat (') and held[1] is not None, held
        admitted = store.execute(
            'SELECT target_session_id, payload_json FROM session_admissions').fetchall()
    assert len(admitted) == 1 and payload in admitted[0][1], admitted
    assert admitted[0][0] == fresh[0][0], admitted


@pytest.mark.platforms("linux")
def test_cron_delivery_recreates_a_bot_chat_held_by_a_closed_login_stamped_session(tmp_path, monkeypatch):
    # The NOC/company store shape: TUI session, served-login label stamp, cli_close.
    with _live_profile(tmp_path, monkeypatch) as home:
        _plant_closed_title_holder(home, '20261015_073159_cb5be7', source='tui', stamp='basic:hermes')
        _assert_page_landed_in_a_fresh_chat(home, '20261015_073159_cb5be7', 'CLOSED_STAMPED_PAYLOAD')


@pytest.mark.platforms("linux")
def test_cron_delivery_recreates_a_bot_chat_closed_by_a_gateway_bounce(tmp_path, monkeypatch):
    # The devsecops store shape: cli session stamped with the host's own identity, but carrying
    # the chat_id route column its old boot left without ever binding it, ended 'session_reset'
    # by the gateway restart. A route column makes the row unadoptable for every caller, and a
    # bounce ends the row on every bounce — without this the mirror lane dies on restart.
    with _live_profile(tmp_path, monkeypatch) as home:
        _plant_closed_title_holder(home, '20261025_050154_a58575', source='cli', stamp='uid:1000',
                                   route_column=True, end_reason='session_reset')
        _assert_page_landed_in_a_fresh_chat(home, '20261025_050154_a58575', 'BOUNCE_CLOSED_PAYLOAD')
