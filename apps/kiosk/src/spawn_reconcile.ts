// POST /api/round/spawn の応答が失われても、同じ世代のGETだけで結果を読む。
// absent は「観測時点で未生成」の意味。遅着POSTとの競合はサーバー側の
// 同一round再spawn拒否で守り、クライアントは自動再POSTしない。
export interface SpawnIdentity {
  sessionId: string;
  roundId: string;
  bootNonce: string;
}

export function spawnRequestPayload(identity: SpawnIdentity) {
  return {
    session_id: identity.sessionId,
    expected_round_id: identity.roundId,
    expected_boot_nonce: identity.bootNonce,
  };
}

export interface SpawnSnapshot<S, G> {
  state: S & { session_id?: string; round_id?: string; phase?: string };
  game: G & { round_id?: string; phase?: string; spec?: unknown;
    spec_boot_nonce?: string | number | null };
  live: { boot_nonce?: string | number | null };
}

export type SpawnOutcome = "spawned" | "absent" | "pending" | "changed" | "unknown";

export function nextAbsentReads(
  previous: number,
  outcome: SpawnOutcome,
  requestSettled: boolean,
): number {
  return outcome === "absent" && requestSettled ? previous + 1 : 0;
}

export function mayOfferManualRetry(
  absentReads: number,
  requestedAt: number,
  now: number,
): boolean {
  return absentReads >= 3 && now - requestedAt >= 3000;
}

export function classifySpawnSnapshot<S, G>(
  identity: SpawnIdentity,
  snapshot: SpawnSnapshot<S, G>,
): SpawnOutcome {
  const { state, game, live } = snapshot;
  if (state.session_id !== identity.sessionId ||
      !state.round_id || !game.round_id || live.boot_nonce == null) return "unknown";
  if (state.round_id !== game.round_id) return "pending";
  if (state.round_id !== identity.roundId ||
      String(live.boot_nonce) !== identity.bootNonce) return "changed";
  if (game.spec != null) {
    if (game.spec_boot_nonce == null) return "unknown";
    if (String(game.spec_boot_nonce) !== identity.bootNonce) return "changed";
    const spec = game.spec as { round_id?: string };
    return spec.round_id === identity.roundId ? "spawned" : "pending";
  }
  if (state.phase === "READY" && game.phase === "READY") return "absent";
  return "pending";
}

export async function readSpawnSnapshot<S, G>(
  fetcher: typeof fetch,
  sessionId: string,
): Promise<SpawnSnapshot<S, G> | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 6000);
  try {
    // bootを最後に読む。先行liveの旧bootを後のspecへ結び付けない。
    const stateResponse = await fetcher(`/api/state/${sessionId}`, {
      cache: "no-store", signal: controller.signal,
    });
    const gameResponse = await fetcher(`/api/game/state/${sessionId}`, {
      cache: "no-store", signal: controller.signal,
    });
    const liveResponse = await fetcher("/api/obs/live", {
      cache: "no-store", signal: controller.signal,
    });
    if (!stateResponse.ok || !gameResponse.ok || !liveResponse.ok) return null;
    const [state, game, live] = await Promise.all([
      stateResponse.json(), gameResponse.json(), liveResponse.json(),
    ]);
    if (!state || !game || !live || typeof state !== "object" ||
        typeof game !== "object" || typeof live !== "object") return null;
    return { state, game, live };
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
