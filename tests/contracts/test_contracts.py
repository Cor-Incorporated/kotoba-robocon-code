"""WP-02契約テスト: handoff schema mirror・特権field拒否・canonical hash。"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from kotoba_contracts.approval import ApprovalRecord
from kotoba_contracts.canonical import canonical_json_bytes, canonical_plan_sha256
from kotoba_contracts.intent import IntentEnvelope, parse_intent
from kotoba_contracts.plan import ControllerProfile, ExecutionPlan
from kotoba_contracts.snapshot import StateSnapshot
from kotoba_contracts.world import World

HANDOFF = Path(__file__).resolve().parents[2] / "docs" / "handoff" / "contracts"

PROFILE = ControllerProfile(
    name="sim_profile_test", max_linear_mps=0.85, max_yaw_radps=0.8, max_duration_s=3.0
)


def _plan(**overrides) -> ExecutionPlan:
    fields = dict(
        session_id="s00000000000000000000000000000000",
        round_id="r00000000000000000000000000000000",
        plan_id="p00000000000000000000000000000000",
        world_version=1,
        goal_target_id="goal_a",
        avoid_zone_ids=["zone_b"],
        kind="walk",
        profile=PROFILE,
        created_monotonic=100.0,
    )
    fields.update(overrides)
    return ExecutionPlan(**fields)


def _load(name):
    return json.loads((HANDOFF / name).read_text(encoding="utf-8"))


EXAMPLES = [
    ("examples/intent-execute.json", "intent.schema.json", parse_intent),
    ("examples/intent-clarify.json", "intent.schema.json", parse_intent),
    ("examples/intent-reject.json", "intent.schema.json", parse_intent),
    ("examples/world.json", "world.schema.json", World.model_validate),
    (
        "examples/state-snapshot.json",
        "state-snapshot.schema.json",
        StateSnapshot.model_validate,
    ),
    (
        "examples/approval-binding.json",
        "approval-binding.schema.json",
        ApprovalRecord.model_validate,
    ),
]


@pytest.mark.parametrize("example,schema,loader", EXAMPLES)
def test_handoff_examples_parse_with_pydantic(example, schema, loader):
    instance = _load(example)
    assert loader(instance) is not None


@pytest.mark.parametrize("example,schema,loader", EXAMPLES)
def test_handoff_examples_validated_against_json_schema(example, schema, loader):
    instance = _load(example)
    validator = Draft202012Validator(_load(schema), format_checker=FormatChecker())
    errors = list(validator.iter_errors(instance))
    assert errors == []
    assert loader(instance) is not None


# --- A01: LLM envelope に数値の運動fieldは存在しない ---
def test_intent_model_has_no_motion_numeric_fields():
    for name in (
        "target_ids",
        "avoid_ids",
        "explanation",
        "decision",
        "schema_version",
    ):
        assert name in IntentEnvelope.__args__[0].model_fields
    execute_model = [
        a for a in IntentEnvelope.__args__ if "target_ids" in a.model_fields
    ][0]
    banned = {
        "velocity",
        "speed",
        "duration",
        "position",
        "approval",
        "token",
        "torque",
    }
    assert banned.isdisjoint(set(execute_model.model_fields))


def test_intent_rejects_privileged_extra_fields():
    raw = _load("examples/intent-execute.json")
    raw["velocity"] = 0.8
    with pytest.raises(ValidationError):
        parse_intent(raw)


def test_intent_rejects_nonfinite():
    raw = _load("examples/world.json")
    raw["targets"][0]["position_m"][0] = float("nan")
    with pytest.raises(ValidationError):
        World.model_validate(raw)


# --- canonical hash ---
def test_canonical_hash_is_key_order_independent():
    plan = _plan()
    canonical = json.loads(canonical_json_bytes(plan.to_canonical_dict()).decode())
    reordered = {k: canonical[k] for k in reversed(list(canonical))}
    assert canonical_json_bytes(canonical) == canonical_json_bytes(reordered)
    assert canonical_plan_sha256(plan) == canonical_plan_sha256(_plan())


def test_canonical_hash_changes_on_tamper():
    base = _plan()
    tampered = _plan(created_monotonic=999.0)
    assert canonical_plan_sha256(base) != canonical_plan_sha256(tampered)


# --- A03: duration<=0 は pydanticでも拒否 ---
def test_profile_zero_duration_rejected():
    with pytest.raises(ValidationError):
        ControllerProfile(
            name="x", max_linear_mps=0.5, max_yaw_radps=0.5, max_duration_s=0.0
        )


# --- snapshot の形 ---
def test_snapshot_rejects_extra_and_bad_quat_len():
    raw = _load("examples/state-snapshot.json")
    raw["quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]  # xyzw誤用の混入は拒否
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(raw)
