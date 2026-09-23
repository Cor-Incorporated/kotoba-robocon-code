"""Characterize the reviewed G1 script without LCM, network or a real clock.

PASS here means the old behavior (including defects) was reproduced.
This is NOT a product acceptance suite. Do not run the snapshot as a CLI.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import struct
import sys
import types

import pytest

# RED面: 旧G1スクリプト（blob 2bd97a9b...）の欠陥characterization。PASS=欠陥の再現。
# 出典: docs/external-review/g1-2026-09-15/tests/test_g1_characterization.py
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'evidence' / 'evo1-g1' / 'closed_loop_walk.py'
EXPECTED_BLOB = '2bd97a9b6f6ae1e7ad516b3df410a032e2752c52'


class Clock:
    def __init__(self):
        self.t = 0.0
    def monotonic(self):
        return self.t
    def sleep(self, dt):
        assert dt >= 0
        self.t += dt


class Handle:
    def __init__(self, clock):
        self.clock = clock
        self.sent = []
        self.callbacks = []
    def publish(self, channel, payload):
        self.sent.append((self.clock.t, channel, payload))
    def subscribe(self, channel, callback):
        self.callbacks.append(callback)
        return object()
    def handle_timeout(self, milliseconds):
        # A queued/arriving 500-Hz observation causes an early return.
        self.clock.t += min(max(milliseconds, 0) / 1000.0, 0.002)
        return 1


@pytest.fixture
def env(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Network is forbidden in this test suite')
    monkeypatch.setattr(socket, 'socket', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    clock = Clock()
    handles = []
    stub = types.ModuleType('lcm')
    def lcm_factory(_url):
        h = Handle(clock)
        handles.append(h)
        return h
    stub.LCM = lcm_factory
    monkeypatch.setitem(sys.modules, 'lcm', stub)
    spec = importlib.util.spec_from_file_location('g1_review_snapshot', SNAPSHOT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'time', clock)
    monkeypatch.delenv('KOTOBA_PUBLISH', raising=False)
    monkeypatch.delenv('ENGINEAI_SIDECAR_PUBLISH', raising=False)
    return mod, clock, handles


def packet(*, stamp=0.0, fp=123, pos=(0.0, 0.0, 0.82),
           vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    # Minimal synthetic packet sufficient for the reviewed hand decoder.
    return struct.pack('>qdi3d3d4d', fp, stamp, 0, *pos, *vel, *quat)


def obsrow(pos, vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return (0.0, 0.0, pos, vel, quat)


def run_main(env, monkeypatch, tmp_path, rows):
    mod, clock, handles = env
    sequence = iter(rows)
    class ScriptedObserver:
        def __init__(self):
            self.latest = rows[0]
            self.handle = Handle(clock)
        def fresh(self, *args, **kwargs):
            self.latest = next(sequence)
            return self.latest
        def spin(self, seconds):
            clock.t += seconds
    monkeypatch.setattr(mod, 'Observer', ScriptedObserver)
    out = tmp_path / 'synthetic-main.json'
    monkeypatch.setattr(mod.sys, 'argv', ['closed_loop_walk.py', 'go', str(out)])
    rc = mod.main()
    return rc, json.loads(out.read_text()), [p for h in handles for p in h.sent]


def successful_rows(final_z=0.82):
    return [
        obsrow((0., 0., 0.82)),  # initial observation
        obsrow((0., 0., 0.82)),  # stand settled
        obsrow((0.25, 0., 0.82)),  # calibration burst
        obsrow((0.4, 0., 0.82), (0.2, 0., 0.)),  # stop trigger
        obsrow((0.5, 0., 0.82)),  # settled
        obsrow((0.5, 0., final_z)),  # final check after 2 seconds
    ]


def test_01_snapshot_git_blob_matches():
    b = SNAPSHOT.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
    assert actual == EXPECTED_BLOB


def test_02_twenty_hz_collapses_with_incoming_observations(env):
    mod, clock, _ = env
    sender = Handle(clock)
    receiver = types.SimpleNamespace(handle=Handle(clock))
    mod.publish_sequence(sender, [mod.IDLE] * 20, hz=20, obs=receiver)
    assert clock.t == pytest.approx(0.04)
    assert [sender.sent[i+1][0] - sender.sent[i][0] for i in range(19)] == pytest.approx([0.002] * 19)
    print('synthetic incoming=500Hz: 20 frames elapsed=0.040s; nominal elapsed=1.000s')


def test_03_twenty_hz_no_observer_control_condition(env):
    mod, clock, _ = env
    sender = Handle(clock)
    mod.publish_sequence(sender, [mod.IDLE] * 20, hz=20, obs=None)
    assert clock.t == pytest.approx(1.0)
    assert sender.sent[-1][0] == pytest.approx(0.95)


def test_04_stale_packet_is_relabelled_fresh_at_callback(env):
    mod, clock, _ = env
    observer = mod.Observer()
    clock.t = 10.
    old_packet = packet(stamp=0.)
    observer._on_state(mod.STATE_CHANNEL, old_packet)
    first = observer.fresh()
    clock.t = 20.
    observer._on_state(mod.STATE_CHANNEL, old_packet)
    second = observer.fresh()
    assert first[1:] == second[1:]
    assert second[0] == 20.  # same old sample now labelled freshly observed


def test_05_drain_does_not_consume_backlog_when_callback_stamp_is_recent(env):
    mod, clock, _ = env
    observer = mod.Observer()
    observer._on_state(mod.STATE_CHANNEL, packet())
    calls = []
    observer.handle.handle_timeout = lambda ms: calls.append(ms)
    observer.drain()
    assert calls == []


def test_06_decoder_accepts_unvalidated_fingerprint(env):
    mod, _, _ = env
    assert mod.decode_state(packet(fp=123))[1] == (0., 0., 0.82)


def test_07_decoder_accepts_nonfinite_position(env):
    mod, _, _ = env
    result = mod.decode_state(packet(pos=(math.nan, 0., 0.82)))
    assert math.isnan(result[1][0])


def test_08_main_emits_without_arming_environment(env, monkeypatch, tmp_path):
    rc, result, sent = run_main(env, monkeypatch, tmp_path, successful_rows())
    assert sent and rc == 0 and result['verdict'] == 'PASS'
    print(f'no arming environment: fake publish calls={len(sent)} (no real network)')


def test_09_main_sends_walk_from_already_fallen_stand(env, monkeypatch, tmp_path):
    mod, _, _ = env
    rows = [obsrow((0., 0., 0.086))] * 3
    rc, result, sent = run_main(env, monkeypatch, tmp_path, rows)
    assert any(p[2] == mod.WALK_STICK_FRAME for p in sent)
    assert result['verdict'] == 'FAIL_burst_no_displacement'
    assert result['events'][1]['event'] == 'pd_stand_settled'
    assert result['events'][1]['pos'][2] < mod.FALL_HEIGHT_M


def test_10_final_tail_fall_can_be_marked_pass(env, monkeypatch, tmp_path):
    rc, result, _ = run_main(env, monkeypatch, tmp_path, successful_rows(final_z=0.07))
    assert rc == 0 and result['verdict'] == 'PASS'
    assert result['events'][-1]['final_pos'][2] == 0.07
    print('synthetic final z=0.070m and zero xy error: old scorer returns PASS')


def test_11_unknown_combo_defaults_to_walk(env):
    mod, _, _ = env
    assert mod.combo('not_a_valid_motion') == mod.combo('walk')


def test_12_stop_transition_follows_nonzero_walk_without_deceleration(env, monkeypatch, tmp_path):
    mod, _, _ = env
    _, _, sent = run_main(env, monkeypatch, tmp_path, successful_rows())
    frames = [s[2] for s in sent]
    stand = mod.combo('pd_stand')
    walk = mod.WALK_STICK_FRAME
    assert any(a == walk and b == stand for a, b in zip(frames, frames[1:]))


def test_13_target_is_chosen_after_calibration_displacement(env, monkeypatch, tmp_path):
    _, result, _ = run_main(env, monkeypatch, tmp_path, successful_rows())
    names = [e['event'] for e in result['events']]
    assert names.index('heading_measured') < names.index('target_set')
    assert result['events'][names.index('target_set')]['target_xy'] == [0.5, 0.0]
