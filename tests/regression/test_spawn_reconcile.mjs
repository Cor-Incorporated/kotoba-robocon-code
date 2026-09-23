import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  classifySpawnSnapshot, mayOfferManualRetry, nextAbsentReads, readSpawnSnapshot,
  spawnRequestPayload,
} from '../../apps/kiosk/src/spawn_reconcile.ts';

const identity = { sessionId: 'session-a', roundId: 'round-a', bootNonce: 'boot-a' };
const base = {
  state: { session_id: 'session-a', round_id: 'round-a', phase: 'READY' },
  game: { round_id: 'round-a', phase: 'READY', spec: null },
  live: { boot_nonce: 'boot-a' },
};
const response = (value, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } });

function fetchSequence(gameResponses) {
  let index = 0;
  const calls = [];
  const fetcher = async (url, init) => {
    calls.push({ url, method: init?.method ?? 'GET' });
    if (url.startsWith('/api/state/')) return response(base.state);
    if (url === '/api/obs/live') return response(base.live);
    const value = gameResponses[Math.min(index++, gameResponses.length - 1)];
    return value instanceof Response ? value : response(value);
  };
  return { fetcher, calls };
}

test('5xx後の遅延commitは同じsession/round/bootのGETで発見し、自動POSTしない', async () => {
  const { fetcher, calls } = fetchSequence([
    response({ detail: 'lost' }, 502),
    base.game,
    { ...base.game, spec: { round_id: 'round-a', seed: 7 }, spec_boot_nonce: 'boot-a' },
  ]);
  assert.equal(await readSpawnSnapshot(fetcher, identity.sessionId), null);
  assert.equal(classifySpawnSnapshot(identity, await readSpawnSnapshot(fetcher, identity.sessionId)), 'absent');
  assert.equal(classifySpawnSnapshot(identity, await readSpawnSnapshot(fetcher, identity.sessionId)), 'spawned');
  assert.equal(calls.length, 9);
  assert.equal(calls.every((call) => call.method === 'GET'), true);
  assert.deepEqual(calls.slice(0, 3).map((call) => call.url), [
    '/api/state/session-a', '/api/game/state/session-a', '/api/obs/live',
  ]);
});

test('未commitは複数回の同世代不在と待機時間を満たした時だけ手動再試行を表示する', () => {
  let count = 0;
  for (const now of [600, 1200, 1800]) {
    const outcome = classifySpawnSnapshot(identity, base);
    count = nextAbsentReads(count, outcome, true);
    assert.equal(mayOfferManualRetry(count, 0, now), false);
  }
  assert.equal(mayOfferManualRetry(count, 0, 3000), true);
  assert.equal(nextAbsentReads(count, 'unknown', true), 0);
  assert.equal(nextAbsentReads(0, 'absent', false), 0);
});

test('旧round・別boot・GET間の世代混在では再試行の不在証拠にしない', () => {
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, game: { ...base.game, round_id: 'round-b', spec: { round_id: 'round-b', seed: 9 } },
  }), 'pending');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, state: { ...base.state, round_id: 'round-b' },
    game: { ...base.game, round_id: 'round-b' },
  }), 'changed');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, live: { boot_nonce: 'boot-b' },
  }), 'changed');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, live: { boot_nonce: null },
  }), 'unknown');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, game: { ...base.game, spec: { round_id: 'round-b', seed: 9 }, spec_boot_nonce: 'boot-a' },
  }), 'pending');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, game: { ...base.game, spec: { round_id: 'round-a', seed: 9 } },
  }), 'unknown');
  assert.equal(classifySpawnSnapshot(identity, {
    ...base, game: { ...base.game, spec: { round_id: 'round-a', seed: 9 }, spec_boot_nonce: 'boot-b' },
  }), 'changed');
});

test('404や通信拒否は未生成と断定しない', async () => {
  const missing = fetchSequence([response({ detail: 'unknown_session' }, 404)]);
  assert.equal(await readSpawnSnapshot(missing.fetcher, identity.sessionId), null);
  assert.equal(await readSpawnSnapshot(async () => { throw Error('offline'); }, identity.sessionId), null);
});

test('生成POSTは照合したsession/round/bootを必須値として送る', () => {
  assert.deepEqual(spawnRequestPayload(identity), {
    session_id: 'session-a',
    expected_round_id: 'round-a',
    expected_boot_nonce: 'boot-a',
  });
});
