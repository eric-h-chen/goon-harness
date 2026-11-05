"""Exercise candidate code in real Docker; only model/benchmark responses are fake."""
import asyncio
import json
from pathlib import Path
import subprocess

import pytest
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.tool import ToolCall

from editable_pi import EditablePiAgent, preflight, validate_source
from harness_api import ExecResult, Feedback, LimitExceeded, Limits, TaskSpec


def response(name=None, args=None, text=''):
    return ModelOutput(model='mock', choices=[{'message': ChatMessageAssistant(content=text,
        tool_calls=[ToolCall(id='call', function=name, arguments=args or {})] if name else []),
        'stop_reason': 'tool_calls' if name else 'stop'}])


class Env:
    def __init__(self): self.calls = []
    async def exec(self, command, timeout):
        self.calls.append(('exec', command))
        return ExecResult(0, 'repository output', '')
    async def read_file(self, path): return 'task contents'
    async def write_file(self, path, content): self.calls.append(('write', path, content))
    async def submit(self, answer):
        self.calls.append(('submit', answer))
        return Feedback(True, 'accepted', 0)


def task():
    return TaskSpec('lcb', 'Return correct', '/task', None, True, Limits(3, 1, 30))


@pytest.fixture
def runtime():
    try: return preflight()
    except (RuntimeError, OSError) as exc: pytest.skip(str(exc))


@pytest.mark.asyncio
async def test_editable_loop_can_define_tools_and_change_workflow(runtime):
    source = '''export async function solve(api, task) {
      const tool = {name:'inspect_and_patch', description:'Combined custom tool',
        parameters:{type:'object', properties:{}, additionalProperties:false}};
      const messages=[{role:'system', content:'Custom diagnosis stage'},
                      {role:'user', content:task.prompt}];
      const plan=await api.generate(messages,[tool]);
      if(plan.content[0].name !== 'inspect_and_patch') throw Error('Expected custom call');
      const inspected=await api.exec('inspect repository');
      await api.writeFile('solution.py',inspected.stdout);
      messages.push(plan, {role:'toolResult',toolCallId:plan.content[0].id,
        toolName:'inspect_and_patch',content:[{type:'text',text:inspected.stdout}],isError:false,timestamp:Date.now()});
      const review=await api.generate(messages,[]);
      await api.submit(review.content[0].text);
    }'''
    calls=[]
    class LLM:
        async def generate(self,messages,tools,config):
            calls.append(messages)
            if len(calls)==1:
                assert tools[0].name == 'inspect_and_patch'
                assert messages[0].text == 'Custom diagnosis stage'
                return response('inspect_and_patch')
            assert messages[-1].text == 'repository output'
            assert not tools
            return response(text='correct')
    env=Env()
    await asyncio.wait_for(EditablePiAgent(source,runtime).solve(task(),env,LLM()),20)
    assert len(calls)==2
    assert env.calls==[('exec','inspect repository'),('write','solution.py','repository output'),('submit','correct')]


@pytest.mark.asyncio
async def test_seed_runs_actual_pi_and_forwards_custom_tool_schemas(runtime):
    source=Path('src/agents/pi_loop.mjs').read_text().replace("tool('bash',", "tool('custom_shell',")
    calls=[]
    class LLM:
        async def generate(self,messages,tools,config):
            calls.append(messages)
            assert 'expert software engineer' in messages[0].text
            assert {t.name for t in tools}=={'custom_shell','read_file','write_file','submit'}
            return response('custom_shell',{'command':'pwd'}) if len(calls)==1 else response('submit',{'code':'correct'})
    env=Env()
    await asyncio.wait_for(EditablePiAgent(source,runtime).solve(task(),env,LLM()),20)
    assert len(calls)==2 and env.calls[-1]==('submit','correct')


@pytest.mark.asyncio
async def test_loop_isolation_and_no_cross_task_state(runtime, tmp_path, monkeypatch):
    secret=tmp_path/'private.txt';secret.write_text('host-private')
    monkeypatch.setenv('OPENAI_API_KEY','host-secret-key')
    source='''import {readFileSync,writeFileSync,existsSync} from 'node:fs';
      export async function solve(api,task) {
        if(process.env.OPENAI_API_KEY) throw Error('credential leaked');
        if(existsSync(HOST_PATH) || existsSync('/var/run/docker.sock')) throw Error('host leaked');
        if(existsSync('/tmp/task-state')) throw Error('state leaked');
        writeFileSync('/tmp/task-state','local');
        let denied=false; try {writeFileSync('/runtime/injected','x');} catch {denied=true;}
        if(!denied) throw Error('runtime writable');
        let network=false; try {await fetch('https://example.com',{signal:AbortSignal.timeout(1000)}); network=true;} catch {}
        if(network) throw Error('network exposed');
        await api.submit('correct');
      }'''.replace('HOST_PATH',json.dumps(str(secret)))
    agent=EditablePiAgent(source,runtime)
    envs=[Env(),Env()]
    await asyncio.wait_for(asyncio.gather(*(agent.solve(task(),env,None) for env in envs)),20)
    assert all(env.calls==[('submit','correct')] for env in envs)


@pytest.mark.asyncio
async def test_budget_and_terminal_submission_enforced_outside_candidate(runtime):
    class LLM:
        def __init__(self): self.calls=0
        async def generate(self,*a,**kw):
            self.calls+=1
            raise LimitExceeded('model budget')
    # Candidate attempts to ignore errors and keep making calls.
    source='export async function solve(api,t){while(true){try{await api.generate([{role:"user",content:"x"}]);}catch{}}}'
    llm=LLM()
    with pytest.raises(LimitExceeded,match='model budget'):
        await asyncio.wait_for(EditablePiAgent(source,runtime).solve(task(),Env(),llm),20)
    assert llm.calls==1
    source='export async function solve(api,t){await api.submit("correct");await api.generate([{role:"user",content:"x"}]);}'
    with pytest.raises(LimitExceeded,match='already submitted'):
        await asyncio.wait_for(EditablePiAgent(source,runtime).solve(task(),Env(),llm),20)
    assert llm.calls==1


@pytest.mark.asyncio
async def test_cancellation_removes_actual_container(runtime):
    before=set(subprocess.check_output(['docker','ps','-q','--filter','name=goon-pi-loop-'],text=True).split())
    agent=EditablePiAgent('export async function solve(){while(true){}}',runtime)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(agent.solve(task(),Env(),None),1.5)
    after=set(subprocess.check_output(['docker','ps','-q','--filter','name=goon-pi-loop-'],text=True).split())
    assert after==before


def test_mjs_loading_does_not_execute_candidate_on_host(runtime,tmp_path):
    from adapter import load_agent
    sentinel=tmp_path/'host-execution'
    source=tmp_path/'harness.mjs'
    source.write_text('import {writeFileSync} from "node:fs"; writeFileSync('+json.dumps(str(sentinel))+',"bad"); export async function solve(){}')
    agent,hid=load_agent(source)
    assert not sentinel.exists()
    assert load_agent(agent)[1]==hid
    source.write_text('export async function solve(){ return 1; }')
    assert load_agent(source)[1]!=hid
    source.write_text('export async function solve( {')
    with pytest.raises(ValueError,match='JavaScript'):
        validate_source(source)


def test_optimizer_archives_selects_resumes_and_checkpoints_mjs(monkeypatch,tmp_path):
    import optimize as opt
    import editable_pi
    from train import write_json
    monkeypatch.setattr(editable_pi,'preflight',lambda:{'image_id':'sha256:fixed','python_sha256':'fixed'})
    rewards=iter([0,1])
    def score(path,model,batch,out,**kwargs):
        assert Path(path).suffix=='.mjs'
        result={'reward':next(rewards),'valid':True}
        write_json(Path(out)/'reward.json',result)
        return result
    async def propose(history,model,prompt,directory,calls,**kwargs):
        assert 'JavaScript' in kwargs['system_prompt']
        assert 'baseline/harness.mjs' in prompt
        assert (history/'loop_api.txt').exists()
        assert 'baseline/harness.mjs' in opt.History(history).files()
        return {'source':'export async function solve(api,task){await api.submit("correct");}',
                'reflection':'Changed execution loop'}
    evaluated=[]
    def evaluate(path,iteration,run,model,**kwargs):
        assert Path(path).suffix=='.mjs'
        evaluated.append(iteration)
        write_json(Path(run)/'eval'/f'iter_{iteration:03d}'/'metrics.json',{})
    monkeypatch.setattr(opt,'score_training',score)
    monkeypatch.setattr(opt,'propose',propose)
    monkeypatch.setattr(opt,'evaluate',evaluate)
    args=dict(model='mock',proposer_model='mock',base_harness='pi-loop',iterations=1,
              candidates=1,batch_size=1,eval_every=1)
    result=opt.optimize(tmp_path,**args)
    assert result['reward']==1 and result['harness'].endswith('best.mjs')
    assert evaluated==[0,1]
    assert opt.optimize(tmp_path,**args)==result
    monkeypatch.setattr(editable_pi,'preflight',lambda:{'image_id':'sha256:changed','python_sha256':'fixed'})
    with pytest.raises(ValueError,match='configuration'):
        opt.optimize(tmp_path,**args)


def test_editable_loop_through_inspect_metering_and_scoring(runtime,tmp_path):
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import get_model
    from inspect_ai.scorer import match
    from adapter import harness_solver, graded_on_submission
    agent=EditablePiAgent('''export async function solve(api,t){
      const result=await api.exec('printf isolated-loop');
      const answer=await api.generate([{role:'user',content:result.stdout}]);
      await api.submit(answer.content[0].text);
    }''',runtime)
    def model_response(messages,tools,tool_choice,config):
        assert messages[-1].text=='isolated-loop'
        return response(text='print(1)')
    model=get_model('mockllm/model',custom_outputs=model_response)
    benchmark=Task(dataset=[Sample(input='Print one',target='print(1)',id='synthetic')],
        solver=harness_solver(agent,'lcb_real'),scorer=graded_on_submission(match()),sandbox='local')
    [log]=inspect_eval(benchmark,model=model,log_dir=str(tmp_path),display='none')
    assert log.status=='success'
    [sample]=log.samples
    assert sample.error is None and sample.scores['pass'].value==1
    assert sample.metadata['harness']['llm_calls']==1
    assert sample.metadata['harness']['submitted']
