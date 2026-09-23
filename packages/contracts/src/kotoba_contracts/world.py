"""世界状態 — docs/handoff/contracts/world.schema.json の mirror。サーバー管理の正本。"""

from typing import Annotated, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from kotoba_contracts.intent import IdStr, SCHEMA_VERSION

SHA256_HEX = r"^[0-9a-f]{64}$"
Sha256Hex = Annotated[str, Field(pattern=SHA256_HEX)]


class WorldTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    id: IdStr
    label: str = Field(min_length=1, max_length=80)
    position_m: List[float] = Field(min_length=3, max_length=3)
    radius_m: float = Field(gt=0)


class ForbiddenRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    id: IdStr
    label: str = Field(min_length=1, max_length=128)
    polygon_xy_m: List[List[float]] = Field(min_length=3, max_length=32)


class World(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    world_version: int = Field(ge=1)
    scene_sha256: Sha256Hex
    coordinate_convention: Literal["right_handed_z_up_meters"] = (
        "right_handed_z_up_meters"
    )
    targets: List[WorldTarget] = Field(min_length=1, max_length=16)
    forbidden_regions: List[ForbiddenRegion] = Field(
        default_factory=list, max_length=16
    )
    capability_profile_sha256: Sha256Hex

    def target(self, target_id: str) -> WorldTarget:
        for t in self.targets:
            if t.id == target_id:
                return t
        raise KeyError(target_id)

    def region(self, region_id: str) -> ForbiddenRegion:
        for r in self.forbidden_regions:
            if r.id == region_id:
                return r
        raise KeyError(region_id)

    def bump_version(self, reason: str) -> "World":
        """不変コピーで版を進める。呼び出し側は理由を監査ログに残すこと。"""
        del reason
        return self.model_copy(update={"world_version": self.world_version + 1})


__all__ = ["World", "WorldTarget", "ForbiddenRegion", "Sha256Hex"]
