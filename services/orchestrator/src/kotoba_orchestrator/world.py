"""世界の版管理。bump は必ず新オブジェクトを登録し旧承認を自然失効させる。"""

from kotoba_contracts.world import World

from kotoba_orchestrator.errors import OrchestratorError


class WorldRegistry:
    def __init__(self) -> None:
        self._current: World = None  # type: ignore[assignment]

    def register(self, world: World) -> World:
        if (
            self._current is not None
            and world.world_version <= self._current.world_version
        ):
            raise OrchestratorError("world_version_not_increasing")
        self._current = world
        return world

    @property
    def current(self) -> World:
        if self._current is None:
            raise OrchestratorError("world_not_registered")
        return self._current

    def bump(self, reason: str) -> World:
        """target/カード配置等の変化で版を進める。旧版への承認は後続の検証で失効する。"""
        return self.register(self.current.bump_version(reason))
