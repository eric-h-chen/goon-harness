// Fixed transport. Candidate code runs only inside the network-disabled container.
import { Agent } from '@earendil-works/pi-agent-core';
import { getCurrentTools } from '@earendil-works/pi-ai';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai/utils/event-stream';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
// Keep ordinary candidate logging off the protocol channel.
console.log = (...args) => console.error(...args);
const input = createInterface({ input: process.stdin });
let start, nextId = 0;
const started = new Promise(resolve => { start = resolve; });
const pending = new Map();
input.on('line', line => {
  try {
    const message = JSON.parse(line);
    if (message.type === 'start') { start(message.task); return; }
    const call = pending.get(message.id);
    if (!call) return;
    pending.delete(message.id);
    if (message.error) call.reject(new Error(message.error));
    else call.resolve(message.result);
  } catch (error) { console.error(String(error)); process.exit(1); }
});
input.on('close', () => {
  for (const call of pending.values()) call.reject(new Error('Evaluator disconnected'));
  pending.clear();
});
function rpc(method, params) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: 'request', id, method, params });
  });
}
let done = false;
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const api = {
  Agent,
  model: { id: 'inspect-solver', name: 'Evaluator-controlled model', api: 'inspect',
    provider: 'inspect', baseUrl: '', reasoning: false, input: ['text'],
    cost: usage.cost, contextWindow: 200000, maxTokens: 16384 },
  get done() { return done; },
  generate: (messages, tools = [], config = {}) => rpc('generate', { messages, tools, config }),
  exec: (command, timeout = 180) => rpc('exec', { command, timeout }),
  readFile: path => rpc('read_file', { path }),
  writeFile: (path, content) => rpc('write_file', { path, content }),
  submit: async answer => {
    const feedback = await rpc('submit', { answer });
    done ||= feedback.attempts_left === 0;
    return feedback;
  },
  // Pi 1.4 carries system prompts and tool declarations in system messages.
  streamFn: (_model, context) => {
    const stream = createAssistantMessageEventStream();
    queueMicrotask(async () => {
      try {
        const message = await api.generate(context.messages, getCurrentTools(context.messages));
        stream.push({ type: 'start', partial: message });
        stream.push({ type: 'done', reason: message.stopReason, message });
        stream.end(message);
      } catch (error) {
        const message = { role: 'assistant', content: [], api: 'inspect', provider: 'inspect',
          model: api.model.id, usage, stopReason: 'error', errorMessage: String(error), timestamp: Date.now() };
        stream.push({ type: 'error', reason: 'error', error: message });
        stream.end(message);
      }
    });
    return stream;
  },
};
try {
  const task = await started;
  const candidate = await import(pathToFileURL('/candidate/harness.mjs'));
  if (typeof candidate.solve !== 'function') throw new Error('Export async function solve(api, task)');
  await candidate.solve(api, task);
  send({ type: 'done' });
} catch (error) {
  send({ type: 'fatal', error: String(error) });
  process.exitCode = 1;
} finally {
  input.close();
  process.stdin.destroy();
}
