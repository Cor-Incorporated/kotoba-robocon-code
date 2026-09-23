/**
 * reset POST の応答が途切れた場合だけ、同じ session の状態を読み戻す。
 * 再POSTすると sim を二度再起動し得るため、ここでは GET だけを行う。
 */
export type ModeReturnObservation<State, Game> =
  | { kind: "normal" | "game" | "pending"; state: State; game: Game }
  | { kind: "unknown" };

type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

export async function inspectModeReturn<
  State extends { phase: string; round_id?: string },
  Game extends { spec?: unknown; round_id?: string },
>(fetcher: Fetcher, sessionId: string): Promise<ModeReturnObservation<State, Game>> {
  try {
    const sid = encodeURIComponent(sessionId);
    const [stateResponse, gameResponse] = await Promise.all([
      fetcher(`/api/state/${sid}`, { cache: "no-store" }),
      fetcher(`/api/game/state/${sid}`, { cache: "no-store" }),
    ]);
    if (!stateResponse.ok || !gameResponse.ok) return { kind: "unknown" };
    const [state, game] = await Promise.all([
      stateResponse.json() as Promise<State>,
      gameResponse.json() as Promise<Game>,
    ]);
    if (!state || typeof state.phase !== "string" ||
        typeof state.round_id !== "string" || !state.round_id ||
        !game || typeof game.round_id !== "string" || !game.round_id ||
        !Object.hasOwn(game, "spec")) return { kind: "unknown" };
    // 並列GETがreset世代を跨いだら、古いspecで失敗と断定しない。
    if (state.round_id !== game.round_id)
      return { kind: "pending", state, game };
    if (state.phase === "RESETTING")
      return { kind: "pending", state, game };
    if (state.phase === "READY" && game.spec === null)
      return { kind: "normal", state, game };
    if (game.spec != null && state.phase !== "RESETTING")
      return { kind: "game", state, game };
    return { kind: "unknown" };
  } catch {
    return { kind: "unknown" };
  }
}
