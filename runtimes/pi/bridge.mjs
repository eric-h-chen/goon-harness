// Trusted Pi agent-core runtime. Model/tool requests use JSONL RPC to Python.
// No Pi provider clients, filesystem tools, saved sessions, or user extensions load.
import { Agent } from '@earendil-works/pi-agent-core';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai/utils/event-stream';
import { createInterface } from 'node:readline';

const input = createInterface({ input: process.stdin });
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
let nextId = 0;
const pending = new Map();
let start;
const started = new Promise(resolve => { start = resolve; });
input.on('line', line => {
  try {
    const msg = JSON.parse(line);
    if (msg.type === 'start') { start(msg); return; }
    const call = pending.get(msg.id);
    if (!call) return;
    pending.delete(msg.id);
    if (msg.error) call.reject(new Error(msg.error));
    else call.resolve(msg.result);
  } catch (error) {
    send({ type: 'fatal', error: String(error) });
    process.exitCode = 1;
    input.close();
  }
});
input.on('close', () => {
  for (const call of pending.values()) call.reject(new Error('Python bridge closed'));
  pending.clear();
});
function rpc(method, params) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    send({ type: 'request', id, method, params });
  });
}
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const model = { id: 'inspect-solver', name: 'Evaluator-controlled model', api: 'inspect',
  provider: 'inspect', baseUrl: '', reasoning: false, input: ['text'],
  cost: usage.cost, contextWindow: 200000, maxTokens: 16384 };
let done = false;
function streamFn(_model, context) {
  const stream = createAssistantMessageEventStream();
  queueMicrotask(async () => {
    try {
      const message = await rpc('generate', { messages: context.messages });
      stream.push({ type: 'start', partial: message });
      stream.push({ type: 'done', reason: message.stopReason, message });
      stream.end(message);
    } catch (error) {
      const message = { role: 'assistant', content: [], api: 'inspect', provider: 'inspect',
        model: model.id, usage, stopReason: 'error', errorMessage: String(error), timestamp: Date.now() };
      stream.push({ type: 'error', reason: 'error', error: message });
      stream.end(message);
    }
  });
  return stream;
}
try {
  const config = await started;
  const tools = config.tools.map(tool => ({
    ...tool, label: tool.name,
    execute: async (_id, args) => {
      if (done) return { content: [{ type: 'text', text: 'Task has ended.' }], details: {}, terminate: true };
      const result = await rpc('tool', { name: tool.name, arguments: args });
      done ||= result.done;
      return { content: [{ type: 'text', text: result.text }], details: {},
        isError: Boolean(result.isError), terminate: done };
    },
  }));
  const agent = new Agent({
    initialState: { model, systemPrompt: config.system_prompt, tools },
    streamFn, toolExecution: 'sequential',
    finishTurn: () => done ? { action: 'end' } : undefined,
  });
  await agent.prompt(config.prompt);
  // Pi normally stops on a text-only answer. This evaluator requires submit.
  while (!done && !agent.state.errorMessage) {
    await agent.prompt(config.continue_prompt);
  }
  if (agent.state.errorMessage) throw new Error(agent.state.errorMessage);
  send({ type: 'done' });
} catch (error) {
  send({ type: 'fatal', error: String(error) });
  process.exitCode = 1;
} finally {
  input.close();
  process.stdin.destroy();
}
