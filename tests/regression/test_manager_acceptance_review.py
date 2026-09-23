"""Characterize the pinned manager, not the EVO or a repaired product.

Executes actual AST function bodies from a hash-verified snapshot.
Only LCM transport, clock, gateway and standing-profile dependencies are stubs.
All packets are synthetic. No network, Docker or business process access.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

# RED面: 検収レビューによる3反例のcharacterization（PASS=旧欠陥の再現）
# 出典: docs/external-review/acceptance-2026-09-16/
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'docs' / 'external-review' / 'acceptance-2026-09-16' / 'snapshot' / 'kotoba_stand_manager.py'
RESULTS: list[dict] = []


def run_case(tmp_path: Path, scenario: str) -> dict:
    clock = SimpleNamespace(t=1000.0)
    stats = {'sent': 0, 'state_packets': 0, 'clock_packets': 0,
             'current_valid_s': 0.0, 'longest_valid_s': 0.0}
    start = clock.t
    last_valid_at = None
    ns: dict = {}
    callbacks: dict = {}
    fp = 0x4B544F4241434C31

    class EndMonitor(BaseException):
        pass

    class Handle:
        def subscribe(self, channel, callback):
            callbacks[channel] = callback
            return channel

        def handle_timeout(self, ms):
            nonlocal last_valid_at
            if (tmp_path / 'boot-ready.json').exists():
                raise EndMonitor()
            clock.t += 0.01
            elapsed = clock.t - start
            packet_gap = scenario == 'state_gap' and 8.0 < elapsed < 11.0
            height = 0.7 if scenario == 'posture_gap' and 8.0 < elapsed < 11.0 else 0.82
            is_valid = not packet_gap and height >= 0.75
            if not is_valid:
                last_valid_at = None
                stats['current_valid_s'] = 0.0
            else:
                if last_valid_at is None:
                    last_valid_at = clock.t
                stats['current_valid_s'] = clock.t - last_valid_at
                stats['longest_valid_s'] = max(stats['longest_valid_s'], stats['current_valid_s'])
            if not packet_gap:
                packet = struct.pack('>qdi', 1, 0.0, 24)
                packet += struct.pack('>72d', *([0.0] * 72))
                packet += struct.pack('>3d3d4d', 0., 0., height, 0., 0., 0., 1., 0., 0., 0.)
                callbacks['sim_state']('sim_state', packet)
                stats['state_packets'] += 1
            if scenario != 'missing_nonce':
                sim_t = 0.0 if scenario == 'frozen_sim_clock' else elapsed
                seq = 1 if scenario == 'frozen_sim_clock' else stats['state_packets']
                payload = struct.pack('>qqqdd', fp, 777, seq, sim_t, clock.t)
                callbacks['kotoba_sim_clock']('kotoba_sim_clock', payload)
                stats['clock_packets'] += 1
            return 1

    class Gateway:
        def __init__(self, *args, **kwargs):
            pass
        def prepare(self, name):
            return b'synthetic-frame'
        def issue(self, *args, **kwargs):
            stats['sent'] += 1
        def close(self):
            pass

    ns.update({
        'RUNTIME': tmp_path, 'READY_FILE': tmp_path/'boot-ready.json',
        'LOST_FILE': tmp_path/'boot-lost.json', 'PASSIVE_FLAG': tmp_path/'passive-detected',
        'CLOCK_FP': fp, 'URL': 'synthetic-no-network',
        'samples': [], 'clock_nonce': None, 'clock_sim_t': None,
        'Path': Path, 'struct': struct, 'json': json,
        'time': SimpleNamespace(monotonic=lambda: clock.t, time=lambda: clock.t),
        'lcm': SimpleNamespace(LCM=lambda _: Handle()),
        'SIM_PROFILE': SimpleNamespace(),
        'RunManifest': lambda **kwargs: SimpleNamespace(**kwargs),
        'SendGateway': Gateway,
        'STAND_HEIGHT_MIN_M': .75, 'STAND_HEIGHT_MAX_M': .90,
        'up_vector_tilt_deg': lambda q: math.degrees(math.acos(max(-1., min(1., 1.-2.*(q[1]**2+q[2]**2))))),
    })
    tree = ast.parse(SNAPSHOT.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])),
                 str(SNAPSHOT), 'exec'), ns)
    (tmp_path/'passive-detected').write_text('synthetic-current-boot')
    stdout = io.StringIO()
    exitcode = None
    with contextlib.redirect_stdout(stdout):
        try:
            exitcode = ns['main']()
        except EndMonitor:
            pass
    ready = tmp_path/'boot-ready.json'
    row = {'scenario': scenario, 'ready_published': ready.exists(),
           'exitcode': exitcode, 'fake_elapsed_s': round(clock.t-start, 3),
           'last_sim_time_s': ns['clock_sim_t'],
           'continuous_valid_before_ready_s': round(stats['current_valid_s'], 3),
           'gateway_stub_sends': stats['sent'], 'state_packets': stats['state_packets'],
           'clock_packets': stats['clock_packets'],
           'ready': json.loads(ready.read_text()) if ready.exists() else None,
           'stdout': stdout.getvalue()}
    RESULTS.append(row)
    (ROOT/'tests/evidence/acceptance-manager-probe-results.json').write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2)+'\n')
    return row


def test_positive_fresh_progressing_state_and_clock_publish_ready(tmp_path):
    r = run_case(tmp_path, 'positive_control')
    assert r['ready_published'] and r['last_sim_time_s'] > 0


def test_negative_missing_nonce_refuses_ready(tmp_path):
    r = run_case(tmp_path, 'missing_nonce')
    assert not r['ready_published'] and r['exitcode'] == 3


def test_characterization_frozen_sim_time_still_publishes_ready(tmp_path):
    r = run_case(tmp_path, 'frozen_sim_clock')
    assert r['ready_published'] and r['last_sim_time_s'] == 0


def test_characterization_pose_gap_does_not_reset_five_second_hold(tmp_path):
    r = run_case(tmp_path, 'posture_gap')
    assert r['ready_published'] and r['continuous_valid_before_ready_s'] < 2


def test_characterization_state_gap_does_not_reset_five_second_hold(tmp_path):
    r = run_case(tmp_path, 'state_gap')
    assert r['ready_published'] and r['continuous_valid_before_ready_s'] < 2
