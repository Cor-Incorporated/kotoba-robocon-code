# kotoba_contracts

`docs/handoff/contracts/*.schema.json`（提案契約）のpydantic v2実装 mirror。

| module | schema | 役割 |
|---|---|---|
| `intent.py` | intent.schema.json | LLM意味エンベロープ（execute/clarify/reject の oneOf）。数値の運動fieldは構造的に存在しない (A01) |
| `world.py` | world.schema.json | サーバー管理の世界状態（目標・禁止領域・版） |
| `snapshot.py` | state-snapshot.schema.json | Native観測の正本wire契約。四元数は wxyz (A10) |
| `approval.py` | approval-binding.schema.json | サーバー内部の承認binding（参加者から受取禁止、単回消費, A02) |
| `plan.py` | （新設・サーバー内部） | 実行計画。速度・秒数はControllerProfile由来のみ |
| `canonical.py` | （新設） | 正規JSON + 計画sha256 |

## 設計意図

- `extra="forbid"`: LLM出力に特権field（approval/safety/速度等）を混入させると ValidationError。旧実装のenvelope防御を型で継承。
- `allow_inf_nan=False`: NaN/±Inf を拒否（旧実装で確認済みの防御を維持）。
- `frozen`: 契約オブジェクトは不変。世界の更新は `World.bump_version` による新オブジェクト。
- LLMはIDと言語だけを返し、実行可能性・速度・秒数の決定はサーバーが担う。

契約例は `docs/handoff/contracts/examples/` を使用（合成データ）。テストは `tests/contracts/`。
