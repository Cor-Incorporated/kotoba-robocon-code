"""承認storeのbinding検証 (A02: 内容・世界・セッション・期限・単回消費の結合)。"""

from datetime import datetime, timedelta, timezone

import pytest

from kotoba_contracts.approval import ApprovalRecord
from kotoba_contracts.canonical import canonical_plan_sha256
from kotoba_contracts.plan import ControllerProfile, ExecutionPlan
from kotoba_orchestrator.approval_store import ApprovalStore
from kotoba_orchestrator.errors import ApprovalError

NOW = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

PROFILE = ControllerProfile(
    name="sim_profile", max_linear_mps=0.85, max_yaw_radps=0.8, max_duration_s=3.0
)


def _plan(**overrides) -> ExecutionPlan:
    fields = dict(
        session_id="sess1",
        round_id="round1",
        plan_id="plan1",
        world_version=3,
        goal_target_id="goal_a",
        avoid_zone_ids=[],
        kind="walk",
        profile=PROFILE,
        created_monotonic=10.0,
    )
    fields.update(overrides)
    return ExecutionPlan(**fields)


def _record(plan: ExecutionPlan, expires=NOW + timedelta(minutes=5)) -> ApprovalRecord:
    return ApprovalRecord(
        session_id=plan.session_id,
        round_id=plan.round_id,
        plan_id=plan.plan_id,
        canonical_plan_sha256=canonical_plan_sha256(plan),
        world_version=plan.world_version,
        controller_profile_sha256="a" * 64,
        sim_boot_id="boot1",
        expires_at=expires,
    )


def _consume(store, approval_id, plan, now=NOW):
    return store.verify_and_consume(
        approval_id,
        plan_sha256=canonical_plan_sha256(plan),
        world_version=plan.world_version,
        session_id=plan.session_id,
        round_id=plan.round_id,
        sim_boot_id="boot1",
        now=now,
    )


def test_issue_then_single_consume_succeeds():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    grant = _consume(store, approval_id, plan)
    assert grant.plan_id == "plan1"
    assert store.is_consumed(approval_id)


def test_double_consume_is_rejected_atomically():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    _consume(store, approval_id, plan)
    with pytest.raises(ApprovalError) as err:
        _consume(store, approval_id, plan)
    assert err.value.reason == "consumed"


def test_tampered_plan_after_approval_is_plan_sha_mismatch():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    tampered = _plan(created_monotonic=plan.created_monotonic + 1.0)
    with pytest.raises(ApprovalError) as err:
        _consume(store, approval_id, tampered)
    assert err.value.reason == "plan_sha_mismatch"  # A02


def test_stale_world_version_is_rejected():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            approval_id,
            plan_sha256=canonical_plan_sha256(plan),
            world_version=plan.world_version + 1,  # world が bump した後
            session_id=plan.session_id,
            round_id=plan.round_id,
            sim_boot_id="boot1",
            now=NOW,
        )
    assert err.value.reason == "world_version_mismatch"


def test_old_session_and_round_are_rejected():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            approval_id,
            plan_sha256=canonical_plan_sha256(plan),
            world_version=plan.world_version,
            session_id="other-session",
            round_id=plan.round_id,
            sim_boot_id="boot1",
            now=NOW,
        )
    assert err.value.reason == "session_mismatch"
    with pytest.raises(ApprovalError) as err2:
        store.verify_and_consume(
            approval_id,
            plan_sha256=canonical_plan_sha256(plan),
            world_version=plan.world_version,
            session_id=plan.session_id,
            round_id="other-round",
            sim_boot_id="boot1",
            now=NOW,
        )
    assert err2.value.reason == "round_mismatch"


def test_expired_approval_is_rejected():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan, expires=NOW - timedelta(seconds=1)))
    with pytest.raises(ApprovalError) as err:
        _consume(store, approval_id, plan)
    assert err.value.reason == "expired"


def test_unknown_id_is_rejected():
    store = ApprovalStore()
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            "nope",
            plan_sha256="0" * 64,
            world_version=1,
            session_id="s",
            round_id="r",
            sim_boot_id="b",
            now=NOW,
        )
    assert err.value.reason == "unknown_id"


def test_reboot_mismatch_is_rejected():
    store = ApprovalStore()
    plan = _plan()
    approval_id = store.issue(_record(plan))
    with pytest.raises(ApprovalError) as err:
        store.verify_and_consume(
            approval_id,
            plan_sha256=canonical_plan_sha256(plan),
            world_version=plan.world_version,
            session_id=plan.session_id,
            round_id=plan.round_id,
            sim_boot_id="boot2",  # reset/再起動後
            now=NOW,
        )
    assert err.value.reason == "boot_mismatch"
