"""世界の版管理 (world_version)。"""

import pytest

from kotoba_contracts.world import World, WorldTarget
from kotoba_orchestrator.errors import OrchestratorError
from kotoba_orchestrator.world import WorldRegistry


def _world(version=1) -> World:
    return World(
        world_version=version,
        scene_sha256="0" * 64,
        targets=[
            WorldTarget(
                id="goal_a", label="あか", position_m=[1.0, 0.0, 0.0], radius_m=0.25
            )
        ],
        forbidden_regions=[],
        capability_profile_sha256="1" * 64,
    )


def test_bump_increases_version_and_keeps_immutability():
    reg = WorldRegistry()
    w1 = reg.register(_world(1))
    w2 = reg.bump("tabletop placement")
    assert w1.world_version == 1
    assert w2.world_version == 2
    assert reg.current.world_version == 2
    assert w1.world_version == 1  # 旧オブジェクトは不変


def test_register_requires_increasing_version():
    reg = WorldRegistry()
    reg.register(_world(2))
    with pytest.raises(OrchestratorError) as err:
        reg.register(_world(2))
    assert err.value.reason == "world_version_not_increasing"


def test_current_before_register_raises():
    reg = WorldRegistry()
    with pytest.raises(OrchestratorError):
        _ = reg.current
