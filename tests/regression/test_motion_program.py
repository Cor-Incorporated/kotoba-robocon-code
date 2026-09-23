"""通常モードの動作プログラム（順序付きstep列）の回帰試験。

指示書B項: 「右に90°旋回して1m歩く」等の複合指示を、決定論parser→
plan(motion_program)→承認→manifestへ通す経路を実ProductSessionで検証。
LLMは呼ばれない（parserが確定した入力のみplan化する）。
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "services" / "harness" / "src" / "kotoba_harness"


def _ensure_real_harness_pkg():
    spec = importlib.util.spec_from_file_location(
        "kotoba_harness",
        HARNESS / "__init__.py",
        submodule_search_locations=[str(HARNESS)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["kotoba_harness"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _session():
    _ensure_real_harness_pkg()
    from kotoba_api.service import ProductSession

    return ProductSession()


# ---- interpret: parser優先経路 --------------------------------------------
def test_interpret_compound_turn_walk_builds_motion_program():
    """「右に90°旋回して1m歩いて」→ REVIEW・kind=motion_program・step列保持。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "右に90°旋回して1m歩いて")
    assert out["decision"] == "execute"
    assert out["program"] is True
    assert out["review"]["steps"] == ["右に90°向く", "前へ1m進む"]
    r = sess.rounds[sid]
    assert r.phase == "REVIEW"
    plan = r.plan_obj
    assert plan.kind == "motion_program"
    assert plan.goal_target_id is None
    assert [s.action for s in plan.steps] == ["turn", "translate"]
    assert plan.steps[0].direction == "right"
    assert plan.steps[0].target == pytest.approx(math.pi / 2, abs=1e-6)
    assert plan.steps[1].direction == "forward"
    assert plan.steps[1].target == pytest.approx(1.0)


def test_interpret_simple_motion_also_program():
    """単発の移動指示も motion_program として REVIEW へ。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "前に1m歩いて")
    assert out["decision"] == "execute"
    assert out["program"] is True
    assert len(sess.rounds[sid].plan_obj.steps) == 1


def test_interpret_stop_outside_run_is_rejected():
    """実行中でない通常モードの「止まって」は実行対象なしで拒否。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "止まって")
    assert out["decision"] == "reject"
    assert out["reason"] == "unsupported_action"
    assert sess.rounds[sid].phase == "READY"


def test_interpret_strike_outside_game_is_rejected():
    """通常モードの「割って」はゲーム専用として拒否。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "割って")
    assert out["decision"] == "reject"
    assert "スイカ割り" in out["message"]


def test_interpret_prohibition_is_rejected_without_llm():
    """禁止文はLLMを呼ばず専用応答で拒否。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "前に進まないで")
    assert out["decision"] == "reject"
    assert out["reason"] == "prohibit"
    assert "出しません" in out["message"]


def test_interpret_unclear_motion_not_routed_to_marker_llm(monkeypatch):
    """R5-02: 運動表現のunclearはmarker用途のLLMへ回さない。

    marker LLMをstubして「呼ばれない」ことを確認し、代わりに
    具体的な聞き返し（clarify）が返ることを検査する。"""
    import kotoba_api.service as svc_mod

    called = {}

    def _fake(text, targets, clarify_context=None):
        called["text"] = text
        return {
            "decision": "reject",
            "reason_code": "unsupported_action",
            "explanation": "わかりません",
        }

    monkeypatch.setattr(svc_mod, "interpret", _fake)
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "ふわっと動いて")
    assert "text" not in called  # marker LLMへ回さない
    assert out["decision"] == "clarify"
    assert "goal_near" not in out["message"]  # 内部IDを露出しない


def test_interpret_marker_ambiguous_still_goes_to_llm(monkeypatch):
    """marker参照が曖昧な入力は従来どおりmarker LLM経路へ。

    「マーカーまで」は方向が決まらないため聞き返し対象。"""
    import kotoba_api.service as svc_mod

    called = {}

    def _fake(text, targets, clarify_context=None):
        called["text"] = text
        return {
            "decision": "clarify",
            "question": "手前と奥のどちらですか？",
            "candidate_target_ids": ["goal_near", "goal_far"],
        }

    monkeypatch.setattr(svc_mod, "interpret", _fake)
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "マーカーまで進んで")
    assert called["text"] == "マーカーまで進んで"
    assert out["decision"] == "clarify"


def test_interpret_marker_direct_resolves_without_llm(monkeypatch):
    """「手前のマーカー」は一意に決まる — LLMを待たず決定的にREVIEWへ。"""
    import kotoba_api.service as svc_mod

    def _boom(text, targets, clarify_context=None):
        raise AssertionError("marker解決にLLMは不要")

    monkeypatch.setattr(svc_mod, "interpret", _boom)
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "手前のマーカーまで")
    assert out["decision"] == "execute"
    assert out["target_label"] == "手前のマーカー"


def test_interpret_run_request_rejected_not_silently_walked():
    """R5-01: 「走って」は受理されるが、能力台帳でrunが未検収なら拒否。

    歩行への黙置換をしない — 代替として速歩きを提示する。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "前へ走って")
    assert out["decision"] == "reject"
    assert out["reason"] == "unsupported_run"
    assert "速く歩いて" in out["message"]


def test_interpret_arbitrary_angle_normal_route():
    """R5-03: 通常経路で任意角度がそのままplan化される（110°≠90°）。"""
    import math

    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "右に110°旋回して")
    assert out["decision"] == "execute"
    plan = sess.rounds[sid].plan_obj
    step = plan.steps[0]
    assert step.action == "turn"
    assert abs(step.target - math.radians(110.0)) < 1e-6


def test_interpret_turn_around_phrases_normal_route():
    """R5-02: 「後ろを振り返って」が通常経路で180°旋回になる。"""
    import math

    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "後ろを振り返って")
    assert out["decision"] == "execute"
    plan = sess.rounds[sid].plan_obj
    assert abs(plan.steps[0].target - math.pi) < 1e-6


# ---- plan digest / 検査 ---------------------------------------------------
def test_motion_plan_digest_binds_steps():
    """step列が承認hashの対象 — 異なるstep列は異なるdigest。"""
    from kotoba_contracts.canonical import canonical_plan_sha256
    from kotoba_orchestrator.validator import build_motion_plan

    sess = _session()
    sid = sess.create_session()
    sess.interpret(sid, "右に90°旋回して1m歩いて")
    p1 = sess.rounds[sid].plan_obj

    from kotoba_api.service import _virtual_world

    r = sess.rounds[sid]
    world = _virtual_world(r.round_id)
    p2 = build_motion_plan(
        [{"action": "turn", "dir": "left", "deg": 90.0, "action_key": "", "label": ""}],
        world,
        session_id=sid,
        round_id=r.round_id,
        plan_id=p1.plan_id,
        profile=p1.profile,
        created_monotonic=p1.created_monotonic,
    )
    # 同一plan_idでもstepsが違えばdigestが変わる（承認はprogramに結合）
    assert canonical_plan_sha256(p1) != canonical_plan_sha256(p2)


def test_build_motion_plan_bounds():
    """validator: 上限超過・不正方向・空列はPlanRejected。"""
    from kotoba_orchestrator.errors import PlanRejected
    from kotoba_orchestrator.validator import build_motion_plan

    sess = _session()
    sid = sess.create_session()
    sess.interpret(sid, "前に1m歩いて")
    r = sess.rounds[sid]
    from kotoba_api.service import _virtual_world

    world = _virtual_world(r.round_id)
    kw = dict(
        session_id=sid,
        round_id=r.round_id,
        plan_id="p1",
        profile=r.plan_obj.profile,
        created_monotonic=0.0,
    )
    with pytest.raises(PlanRejected):
        build_motion_plan(
            [{"action": "translate", "dir": "forward", "m": 3.0}], world, **kw
        )
    with pytest.raises(PlanRejected):
        build_motion_plan(
            [{"action": "turn", "dir": "around", "deg": 270.0}], world, **kw
        )
    with pytest.raises(PlanRejected):
        build_motion_plan(
            [{"action": "translate", "dir": "around", "m": 1.0}], world, **kw
        )
    with pytest.raises(PlanRejected):
        build_motion_plan([], world, **kw)
    # 合計距離の上限（2m超の複合は拒否 — parser側でも弾くが二重検査）
    with pytest.raises(PlanRejected):
        build_motion_plan(
            [
                {"action": "translate", "dir": "forward", "m": 1.5},
                {"action": "translate", "dir": "forward", "m": 1.0},
            ],
            world,
            **kw,
        )


def test_turn_180_pi_boundary_full_path():
    """run 54e75a37 回帰: 「後ろを向いて」→ plan → runner MotionStep まで
    180°が境界で弾かれない（bad_turn_target:3.141593 → FAIL_INTERNAL）。"""
    sess = _session()
    sid = sess.create_session()
    out = sess.interpret(sid, "後ろを向いて")
    assert out["decision"] == "execute"
    step = sess.rounds[sid].plan_obj.steps[0]
    assert step.action == "turn" and step.direction == "around"
    assert step.target == pytest.approx(math.pi)
    # runner側のMotionStep構築が境界でValueErrorにならない
    _ensure_real_harness_pkg()
    from kotoba_harness.motion import MotionStep

    ms = MotionStep("turn", "around", step.target)
    assert ms.target <= math.pi


def test_motion_plan_approve_binds_program():
    """approve → start_runまで同一planに結合（承認消費の経路はmarkerと同一）。"""
    sess = _session()
    sid = sess.create_session()
    sess.boot_nonce = "boot-1"
    sess.interpret(sid, "右に90°旋回して1m歩いて")
    r = sess.rounds[sid]
    out = sess.approve(sid, r.pending_plan_id)
    assert out["approved"] is True


def test_interpret_reject_keeps_ready():
    """parser reject後に次の指示を受け付ける（phase stuckしない）。"""
    sess = _session()
    sid = sess.create_session()
    sess.interpret(sid, "止まって")
    out = sess.interpret(sid, "前に50cm進んで")
    assert out["decision"] == "execute"
    assert sess.rounds[sid].phase == "REVIEW"
