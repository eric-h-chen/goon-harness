// Fully editable agent implementation. The fixed evaluator supplies only api/task.
export async function solve(api, task) {
  const schema = properties => ({ type: 'object', properties,
    required: Object.keys(properties), additionalProperties: false });
  const string = { type: 'string' };
  const tool = (name, description, parameters, run) => ({ name, label: name,
    description, parameters,
    execute: async (_id, args) => {
      if (api.done) return { content: [{ type: 'text', text: 'Task ended.' }], details: {}, terminate: true };
      const result = await run(args);
      return { content: [{ type: 'text', text: String(typeof result === 'string' ? result : JSON.stringify(result)).slice(0, 20000) }],
        details: {}, terminate: api.done };
    },
  });
  const tools = [
    tool('bash', 'Execute a command in the task working directory.', schema({ command: string }),
      args => api.exec(args.command, 180)),
    tool('read_file', 'Read a task file.', schema({ path: string }), args => api.readFile(args.path)),
    tool('write_file', 'Create or overwrite a task file.', schema({ path: string, content: string }),
      args => api.writeFile(args.path, args.content)),
    tool('submit', 'Submit for grading.', schema(task.kind === 'lcb' ? { code: string } : {}),
      args => api.submit(args.code)),
  ];
  const agent = new api.Agent({
    initialState: { model: api.model, tools,
      systemPrompt: `You are an expert software engineer working through tools.
Explore the task, implement a solution, and test it when useful.
Use read_file, write_file, and bash to work in the task environment.
Call submit when your solution is ready. If feedback reports failure and you have
attempts remaining, investigate the failure and try again.
Working directory: ${task.workdir}. You have ${task.limits.max_attempts} submission attempts.
${task.kind === 'lcb' ? 'Submit complete Python code with submit(code).' : 'Modify the repository and call submit() to submit its current state.'}` },
    streamFn: api.streamFn, toolExecution: 'sequential',
    finishTurn: () => api.done ? { action: 'end' } : undefined,
  });
  await agent.prompt(task.prompt);
  while (!api.done && !agent.state.errorMessage) {
    await agent.prompt('Continue working. Call submit when your solution is ready.');
  }
  if (agent.state.errorMessage) throw new Error(agent.state.errorMessage);
}
