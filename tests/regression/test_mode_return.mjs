import assert from 'node:assert/strict';
import { test } from 'node:test';

import { inspectModeReturn } from '../../apps/kiosk/src/mode_return.ts';

const response = (value, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' },
  });

function mockFetch(state, game, status = 200) {
  const calls = [];
  const fetcher = async (url, init) => {
    calls.push({ url, method: init?.method ?? 'GET', cache: init?.cache });
    return response(url.startsWith('/api/game/') ? game : state, status);
  };
  return { fetcher, calls };
}

test('応答が途切れても現roundのREADYとspec消失から切替成功を確認する', async () => {
  const { fetcher, calls } = mockFetch(
    { phase: 'READY', round_id: 'new' },
    { phase: 'READY', round_id: 'new', spec: null },
  );
  const observed = await inspectModeReturn(fetcher, 'session-test');
  assert.equal(observed.kind, 'normal');
  assert.deepEqual(calls, [
    { url: '/api/state/session-test', method: 'GET', cache: 'no-store' },
    { url: '/api/game/state/session-test', method: 'GET', cache: 'no-store' },
  ]);
});

test('specが残る場合は切替失敗を確定し、resetを自動再送しない', async () => {
  const { fetcher, calls } = mockFetch(
    { phase: 'READY', round_id: 'old' },
    { phase: 'READY', round_id: 'old', spec: { seed: 1 } },
  );
  assert.equal((await inspectModeReturn(fetcher, 'session-test')).kind, 'game');
  assert.equal(calls.every((call) => call.method === 'GET'), true);
});

test('RESETTING中やGETの世代不一致は判定を保留する', async () => {
  const resetting = mockFetch(
    { phase: 'RESETTING', round_id: 'old' },
    { phase: 'RESETTING', round_id: 'old', spec: { seed: 1 } },
  );
  assert.equal((await inspectModeReturn(resetting.fetcher, 'session-test')).kind, 'pending');
  const crossed = mockFetch(
    { phase: 'READY', round_id: 'new' },
    { phase: 'READY', round_id: 'old', spec: { seed: 1 } },
  );
  assert.equal((await inspectModeReturn(crossed.fetcher, 'session-test')).kind, 'pending');
});

test('API断や不正応答は成功とも失敗とも断定しない', async () => {
  const badStatus = mockFetch(
    { phase: 'READY', round_id: 'new' },
    { phase: 'READY', round_id: 'new', spec: null }, 502,
  );
  assert.equal((await inspectModeReturn(badStatus.fetcher, 'session-test')).kind, 'unknown');
  assert.equal((await inspectModeReturn(async () => { throw Error('offline'); }, 'session-test')).kind,
    'unknown');
  const missingRound = mockFetch(
    { phase: 'READY', round_id: 'new' },
    { phase: 'READY', spec: null },
  );
  assert.equal((await inspectModeReturn(missingRound.fetcher, 'session-test')).kind, 'unknown');
  const missingSpec = mockFetch(
    { phase: 'READY', round_id: 'new' },
    { phase: 'READY', round_id: 'new' },
  );
  assert.equal((await inspectModeReturn(missingSpec.fetcher, 'session-test')).kind, 'unknown');
});
