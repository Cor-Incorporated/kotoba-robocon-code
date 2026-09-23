"""PR7 source characterization, NOT product acceptance or EVO physics tests.
All inputs are synthetic. lcm is replaced before module import: no sockets,
Docker, model inference, private hosts, SDK or robot commands are executed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import types

import pytest

# RED面: PR7時点の製品コード欠陥characterization（PASS=欠陥の再現）
# 出典: docs/external-review/pr7-2026-09-15/
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'docs' / 'external-review' / 'pr7-2026-09-15' / 'snapshot'
# 他テストで既に読み込まれた現行kotoba_harnessを破棄し、snapshot版を確実に読む
for _m in [k for k in sys.modules if k.startswith('kotoba_harness')]:
    del sys.modules[_m]
sys.path.insert(0, str(SNAPSHOT))
from kotoba_harness.auth import SIM_PROFILE, RunManifest, SendGateway
from kotoba_harness.errors import AuthorizationRefused, HarnessError
from kotoba_harness.scheduler import DeadlineScheduler
from kotoba_harness.trial import ReadyGate, Scorer


class Clock:
    def __init__(self):
        self.t = 100.0
    def now(self):
        return self.t
    def sleep(self, seconds):
        self.t += max(float(seconds), 0.0000001)


class Sink:
    def __init__(self, clock):
        self.clock = clock
        self.sent = []
        self.subs = {}
    def publish(self, channel, payload):
        self.sent.append((self.clock.now(), channel, payload))
    def subscribe(self, channel, callback):
        self.subs[channel] = callback
        return len(self.subs)
    def handle_timeout(self, milliseconds):
        self.clock.sleep(milliseconds / 1000)
        return 0


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock = Clock()
    handles = []
    lcm_stub = types.ModuleType('lcm')
    def factory(_url):
        h = Sink(clock)
        handles.append(h)
        return h
    lcm_stub.LCM = factory
    monkeypatch.setitem(sys.modules, 'lcm', lcm_stub)
    SendGateway._active = None
    name = 'review_runner_' + tmp_path.name.replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, SNAPSHOT / 'kotoba_runner.py')
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    mod.time = types.SimpleNamespace(monotonic=clock.now, time=clock.now, sleep=clock.sleep)
    mod.LIVE_PATH = str(tmp_path / 'live.json')
    mod.DeadlineScheduler = lambda: DeadlineScheduler(clock.now, clock.sleep)
    yield types.SimpleNamespace(mod=mod, clock=clock, handles=handles, tmp=tmp_path)
    SendGateway._active = None


def sample(height):
    return types.SimpleNamespace(position=(0., 0., height), velocity=(0.,0.,0.),
                                 quaternion_wxyz=(1.,0.,0.,0.))


def state_payload(*, fingerprint=0, x=0., height=.82, quaternion=(1.,0.,0.,0.)):
    # Exactly the prefix read by the product parser. This is a synthetic
    # callback payload, not a real LCM transmission.
    n = 24
    return (struct.pack('>qdi', fingerprint, 0., n) + b'\0' * (3*n*8)
            + struct.pack('>3d3d4d', x, 0., height, 0.,0.,0., *quaternion))


def clock_payload(mod, *, nonce=123, seq=1, sim_t=0.002):
    return struct.pack('>qqqdd', mod.CLOCK_FP, nonce, seq, sim_t, 100.0)


def run_main(env, monkeypatch, *, height=.30, expire=240, cast_distance=False):
    mod, clock = env.mod, env.clock
    if cast_distance:
        source = (SNAPSHOT / 'kotoba_runner.py').read_text()
        old = '    result_path, target_distance, manifest_path = sys.argv[1:4]\n'
        assert source.count(old) == 1
        patched = source.replace(old, old + '    target_distance = float(target_distance)\n')
        # Counterfactual-only one-line modification in memory. The snapshot
        # stays untouched and remains hash-verifiable.
        exec(compile(patched, '<counterfactual-float-only>', 'exec'), mod.__dict__)
        mod.time = types.SimpleNamespace(monotonic=clock.now, time=clock.now, sleep=clock.sleep)
        mod.DeadlineScheduler = lambda: DeadlineScheduler(clock.now, clock.sleep)
        mod.LIVE_PATH = str(env.tmp / 'live.json')

    class FakeObservation:
        def __init__(self):
            self.received = clock.now()
            self.x = 0.
            self.spin_calls = []
        def spin(self, seconds):
            self.spin_calls.append(seconds)
            clock.sleep(seconds)
            self.received = clock.now()
            if seconds == 4.0:
                self.x = .20  # synthetic calibration displacement
        def latest(self):
            age = clock.now() - self.received
            if age > .5:
                raise HarnessError(f'stale_observation:{age:.2f}s')
            return ((self.x,0.,height), (0.,0.,0.), (1.,0.,0.,0.),
                    (self.received, 123, 1, .002, self.received))

    obs = FakeObservation()
    monkeypatch.setattr(mod, 'Observation', lambda: obs)
    manifest_path = env.tmp / 'synthetic-manifest.json'
    manifest_path.write_text(json.dumps({'run_id':'synthetic-review', 'arming':'1',
                                        'expires_in_s':expire, 'boot_expect':'boot-pending'}))
    result_path = env.tmp / 'result.json'
    monkeypatch.setattr(mod.sys, 'argv', ['runner.py', str(result_path), '.45', str(manifest_path)])
    code = mod.main()
    result = json.loads(result_path.read_text())
    return code, result, obs


def extracted_send_seq(env, gateway):
    tree = ast.parse((SNAPSHOT / 'kotoba_runner.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name=='send_seq')
    ns = dict(env.mod.__dict__)
    ns.update(gateway=gateway, gw_handle=gateway.handle, events=[],
              sched=DeadlineScheduler(env.clock.now, env.clock.sleep))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<exact-send-seq-ast>', 'exec'), ns)
    return ns['send_seq']


def gateway_for(env, expires):
    sink = Sink(env.clock)
    manifest = RunManifest('synthetic', 'evaluation', SIM_PROFILE, 'synthetic-boot', expires)
    return SendGateway(manifest, sink, 'synthetic-only', now_monotonic=env.clock.now(),
                       sim_mode_confirmed=True, arming_env='1')


def test_01_ready_gate_rejects_official_pm01_reference_height():
    gate = ReadyGate()
    decisions = [gate.observe(sample(.82), t, None) for t in (0.,1.,2.,3.,10.)]
    assert not any(decisions)


def test_02_ready_gate_accepts_synthetic_030_height():
    gate = ReadyGate()
    assert not gate.observe(sample(.30),0.,None)
    assert gate.observe(sample(.30),2.1,None)


def test_03_scorer_rejects_upright_082_even_at_goal():
    result = Scorer().verdict(ready_latched=True, fall_category=None, tail_ok=True,
                            tail_held_s=5., final_sample=sample(.82), target_xy=(0.,0.))
    assert result['verdict']=='FAIL'
    assert result['reasons']==['posture:h=0.820 tilt=0.0deg']


def test_04_main_on_upright_082_never_reaches_walk(env, monkeypatch):
    _, result, _ = run_main(env, monkeypatch, height=.82)
    assert result['verdict']=='FAIL_INVALID_START_NEVER_READY'
    assert not any(e['event']=='ready' for e in result['events'])


def test_05_main_argv_distance_remains_string_and_crashes_after_ready(env, monkeypatch):
    code, result, _ = run_main(env, monkeypatch)
    assert code==3 and result['verdict']=='FAIL_INTERNAL'
    assert any('TypeError' in r and 'multiply sequence' in r for r in result['reasons'])
    assert result['target_distance_m']=='.45'
    assert any(e['event']=='ready' for e in result['events'])


def test_06_counterfactual_float_fix_exposes_unserviced_observer(env, monkeypatch):
    _, result, _ = run_main(env, monkeypatch, cast_distance=True)
    assert result['verdict'].startswith('FAIL_STALE_OBSERVATION:')
    assert any(e['event']=='calibrated' for e in result['events'])


def test_07_gateway_expiry_rejects_when_actual_send_method_used(env):
    gw = gateway_for(env, env.clock.now()-1)
    try:
        with pytest.raises(AuthorizationRefused, match='expired'):
            gw.send('idle', now_monotonic=env.clock.now())
        assert gw.handle.sent==[]
    finally:
        gw.close()


def test_08_product_send_seq_bypasses_expiry_and_audit(env):
    gw = gateway_for(env, env.clock.now()-1)
    try:
        extracted_send_seq(env, gw)(['idle'])
        assert len(gw.handle.sent)==1
        assert gw.sent_log==[]
    finally:
        gw.close()


def test_09_product_batch_boundaries_allow_zero_send_interval(env):
    gw = gateway_for(env, env.clock.now()+30)
    try:
        send = extracted_send_seq(env, gw)
        send(['idle']*3)
        send(['idle']*3)
        times = [x[0] for x in gw.handle.sent]
        assert len(times)==6
        assert times[3]-times[2] == 0
    finally:
        gw.close()


def test_10_state_fingerprint_is_not_validated(env):
    obs = env.mod.Observation()
    obs._on_clock('', clock_payload(env.mod))
    obs._on_state('', state_payload(fingerprint=0))
    assert obs.latest()[0][2]==.82
    obs._on_state('', state_payload(fingerprint=123456789))
    assert obs.latest()[0][2]==.82


def test_11_nonfinite_position_is_accepted(env):
    obs = env.mod.Observation()
    obs._on_clock('', clock_payload(env.mod))
    obs._on_state('', state_payload(x=float('nan')))
    assert math.isnan(obs.latest()[0][0])


def test_12_non_unit_quaternion_is_accepted(env):
    obs = env.mod.Observation()
    obs._on_clock('', clock_payload(env.mod))
    obs._on_state('', state_payload(quaternion=(2.,0.,0.,0.)))
    assert obs.latest()[2]==(2.,0.,0.,0.)


def test_13_stopped_clock_reused_with_new_state(env):
    obs = env.mod.Observation()
    obs._on_clock('', clock_payload(env.mod))
    obs._on_state('', state_payload())
    first = obs.latest()[3]
    env.clock.sleep(10)
    obs._on_state('', state_payload(x=.1))
    result = obs.latest()
    assert result[3]==first
    assert env.clock.now()-result[3][0] >= 10


def test_14_clock_sequence_regression_is_accepted(env):
    obs = env.mod.Observation()
    obs._on_clock('', clock_payload(env.mod, seq=10, sim_t=.020))
    obs._on_state('', state_payload())
    assert obs.latest()[3][2]==10
    obs._on_clock('', clock_payload(env.mod, seq=9, sim_t=.018))
    obs._on_state('', state_payload())
    assert obs.latest()[3][2]==9


def test_15_clock_bad_fingerprint_positive_control_rejects(env):
    obs = env.mod.Observation()
    obs._on_clock('', struct.pack('>qqqdd',0,123,1,.002,100.))
    assert obs.clock is None
    obs._on_state('',state_payload())
    with pytest.raises(HarnessError, match='clock_not_bound'):
        obs.latest()
