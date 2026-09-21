"""No-training fixtures for the paper-suite runner and checkpoint contract."""
import contextlib
import copy
import csv
import gzip
from concurrent.futures import Future
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_runner_tested',PACKAGE/'src/runner.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
PNL='PNL-causal-learn'

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='paper-runner-fixture-')
        self.root=Path(self.tmp.name);self.original_root=R.ROOT;R.ROOT=self.root;R.STOP=False
        self.signals={sig:signal.getsignal(sig) for sig in [signal.SIGTERM,signal.SIGINT]}
        R.atomic_json(self.root/'config/protocol.json',dict(method_pnl=PNL))
        self.rows={}
        for i,suite in enumerate(R.SUITES):
            row=dict(case_id=f'{suite}_fixture',domain=suite,split=suite,suite=suite,n=200,
                     truth=1,seed_namespace='fixture_namespace',data_seed=100+i,gp_seed=200+i,
                     krr_seed=300+i,pnl_seed=400+i,weight=1.)
            self.rows[suite]=row;self.manifest(suite,[row])
        self.manifest('smoke',[dict(row,case_id='smoke_'+suite,n=80,smoke_suite=suite,engineering_smoke='TRUE')
                               for suite,row in self.rows.items()])
        self.args=SimpleNamespace(command='run',suite='anm',methods='all',workers=32,rscript='fake-R',
            timeout=1,reserve_gib=6.,memory_budget_gib=None,inputs=None,partial=False)
        self.requested=[]
    def tearDown(self):
        for sig,handler in self.signals.items():signal.signal(sig,handler)
        R.ROOT=self.original_root;R.STOP=False;self.tmp.cleanup()
    def manifest(self,name,rows):
        path=self.root/f'config/manifests/{name}.csv.gz';path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def freeze(self):
        files={str(path.relative_to(self.root)):R.sha(path) for path in (self.root/'config').rglob('*') if path.is_file()}
        R.atomic_json(self.root/'MANIFEST.json',dict(experiment_id='fixture',files=files))
        return R.fingerprint()
    def jobs(self):return R.jobs_for(self.args,R.fingerprint(False))[1]
    def fake_r(self,error=None,timeout=False,completed=None,hash_value='same-observations'):
        owner=self
        class Process:
            returncode=0
            def __init__(self,command,**kwargs):
                self.terminated=False
                row=R.read_json(command[-3])[0];out=Path(command[-2])
                methods=set(filter(None,row['requested_methods'].split(',')))
                configs=set(filter(None,row['requested_configurations'].split(',')))
                owner.requested.append((methods,configs))
                (out/'1.csv').write_text('x,y\n0,0\n1,1\n')
                if error:result=dict(preparation_error=error)
                else:
                    outputs={}
                    for method in methods:
                        conf=configs if method=='RKDS' else ['D1' if method=='KCDC' else 'default']
                        for cfg in conf:
                            if completed is not None and (method,cfg) not in completed:continue
                            outputs[method+'_'+cfg]=dict(method=method,configuration=cfg,score_forward=.1,
                                score_reverse=.9,prediction=1,status='directed',seconds=.02)
                    result=dict(case_id=row['case_id'],metadata=dict(truth=1,swapped=False,n_used=row['n'],
                        data_hash=hash_value,generation_attempts=1),outputs=outputs,r_stage_incomplete=timeout)
                R.atomic_json(out/'1.json',result)
            def wait(self,timeout=None):
                if self.terminated:return self.returncode
                if owner_timeout:raise subprocess.TimeoutExpired('fake-R',timeout)
                return 0
            def terminate(self):self.terminated=True;self.returncode=-15
            def kill(self):self.terminated=True;self.returncode=-9
        owner_timeout=timeout
        return Process
    def fake_python(self,fail=None,interrupt_after=None):
        def execute(path,row,role='all',timeout=14400,requested=None,on_result=None):
            answer={}
            for method in ['ANM_GP','ANM_KRR',PNL]:
                if method not in requested:continue
                out=dict(pforward=.8,preverse=.01,prediction=1,status='directed',seconds=.03,configuration='default')
                if method==fail:out=dict(prediction=0,status='implementation_failure',seconds=.03,error='fixture failure',configuration='default')
                answer[method]=out;on_result(method,out)
                if method==interrupt_after:raise RuntimeError('fixture interruption')
            return answer
        return execute
    def execute(self,job,**kwargs):
        Path(job['scratch']).mkdir(parents=True,exist_ok=True)
        with mock.patch.object(R.subprocess,'Popen',self.fake_r(**kwargs)),mock.patch.object(R,'python_methods',side_effect=self.fake_python()):
            return R.run_job(job)
    def test_manifest_numeric_seed_and_stable_method_independent_path(self):
        row=R.csv_rows('anm')[0]
        self.assertIsInstance(row['data_seed'],int);self.assertEqual(row['seed_namespace'],'fixture_namespace')
        a=self.jobs()[0];self.args.methods='pnl';b=self.jobs()[0]
        self.assertEqual(a['checkpoint'],b['checkpoint'])
        self.assertEqual(a['id'],R.hashlib.sha256(row['case_id'].encode()).hexdigest()[:20])
        self.assertEqual(len(R.expected_keys('all','anm')),6)
        self.assertEqual(len(R.expected_keys('core','anm')),5)
        self.assertEqual(len(R.expected_keys('all','sensitivity')),12)
        self.args.suite='sensitivity';self.assertEqual(self.jobs(),[])
    def test_all_and_smoke_counts_are_isolated(self):
        self.args.suite='all';self.assertEqual(len(self.jobs()),4)
        self.assertEqual(sum(len(R.expected_keys('all',j['suite'])) for j in self.jobs()),30)
        self.args.command='smoke';self.args.suite='smoke';jobs=self.jobs()
        self.assertEqual(len(jobs),4);self.assertTrue(all('/verification/smoke_run/' in j['checkpoint'] for j in jobs))
        self.assertTrue(all(j['rows'][0]['case_id'].startswith('smoke_') for j in jobs))
    def test_native_only_codes_no_pseudo_p_and_diagnostic_mapback(self):
        row=self.rows['anm'];meta=dict(swapped=True,truth=1,data_hash='hash')
        out=dict(score_forward=.1,score_reverse=.9,prediction=0,status='screen_nonrejection',
                 raw_direction=1,prediction_forced=1,seconds=.1,condition_number_forward=2.,condition_number_reverse=3.)
        result=R.convert(row,meta,'RKDS',out,'fp')
        self.assertEqual(result['prediction_native'],0);self.assertNotIn('p_forward',result)
        self.assertNotIn('raw_direction',result['details']);self.assertNotIn('prediction_forced',result['details'])
        self.assertEqual(result['condition_number_forward'],3.)
        gp=R.convert(row,meta,'ANM_GP',dict(pforward=.9,preverse=.01,prediction=1,status='directed',seconds=.1),'fp')
        self.assertEqual(gp['prediction_native'],2);self.assertEqual(gp['p_forward'],.01)
        invalid=R.convert(row,meta,'RKDS',dict(status='zero_denominator',seconds=.1),'fp')
        self.assertFalse(invalid['success']);self.assertTrue(invalid['error'])
    def test_complete_resume_never_reexecutes_success(self):
        job=self.jobs()[0];self.execute(job);saved=R.load_checkpoint(job)
        self.assertEqual(len(saved['records']),6);self.assertEqual(saved['completed'],[job['rows'][0]['case_id']])
        with mock.patch.object(R.subprocess,'Popen') as popen:
            self.assertTrue(R.run_job(job)['skipped']);popen.assert_not_called()
    def test_split_methods_merge_in_place_and_full_marker(self):
        self.args.methods='core';job=self.jobs()[0];self.execute(job)
        self.assertEqual(len(R.load_checkpoint(job)['records']),5)
        self.assertEqual(R.load_checkpoint(job)['completed'],[])
        self.assertTrue(R.is_complete(job))
        before=copy.deepcopy(R.load_checkpoint(job)['records'])
        self.args.methods='pnl';pnl_job=self.jobs()[0];self.execute(pnl_job)
        after=R.load_checkpoint(pnl_job);self.assertEqual(len(after['records']),6)
        self.assertTrue(after['completed'])
        for record in before:self.assertIn(record,after['records'])
    def test_retry_only_failed_pnl_keeps_history_and_success_bytes(self):
        job=self.jobs()[0];Path(job['scratch']).mkdir(parents=True,exist_ok=True)
        with mock.patch.object(R.subprocess,'Popen',self.fake_r()),mock.patch.object(R,'python_methods',side_effect=self.fake_python(fail=PNL)):
            R.run_job(job)
        before=R.load_checkpoint(job);successful=[r for r in before['records'] if r['success']]
        job['retry_failed']=True;answer=self.execute(job)
        self.assertEqual(answer['recomputed_keys'],[(PNL,'default')]);self.assertEqual(self.requested[-1][0],set())
        after=R.load_checkpoint(job)
        for record in successful:self.assertIn(record,after['records'])
        self.assertEqual(len(list((Path(job['checkpoint']).parent.parent/'attempt_history').glob('*.json'))),1)
    def test_partial_interruption_fills_only_missing_python(self):
        job=self.jobs()[0];Path(job['scratch']).mkdir(parents=True,exist_ok=True)
        with mock.patch.object(R.subprocess,'Popen',self.fake_r()),mock.patch.object(R,'python_methods',side_effect=self.fake_python(interrupt_after='ANM_GP')):
            with self.assertRaises(RuntimeError):R.run_job(job)
        partial=R.load_checkpoint(job);self.assertEqual(len(partial['records']),4);self.assertEqual(partial['completed'],[])
        answer=self.execute(job);self.assertEqual(answer['recomputed_keys'],[('ANM_KRR','default'),(PNL,'default')])
    def test_sensitivity_twelve_keys_and_subset_retry(self):
        self.args.suite='sensitivity';job=self.jobs()[0]
        self.execute(job,timeout=True,completed={('RKDS','H13')})
        before=R.load_checkpoint(job);self.assertEqual(len(before['records']),12)
        self.assertEqual(sum(r['success'] for r in before['records']),1)
        baseline=next(r for r in before['records'] if r['configuration']=='H13')
        job['retry_failed']=True;self.execute(job)
        self.assertEqual(len(self.requested[-1][1]),11);self.assertNotIn('H13',self.requested[-1][1])
        after=R.load_checkpoint(job);self.assertIn(baseline,after['records']);self.assertTrue(all(r['success'] for r in after['records']))
    def test_preparation_failure_is_complete_failure_not_abstention(self):
        job=self.jobs()[0];self.execute(job,error='fixture failed preparation')
        saved=R.load_checkpoint(job);self.assertEqual(len(saved['records']),6)
        self.assertFalse(saved['inputs'][job['rows'][0]['case_id']]['prepared'])
        self.assertTrue(all(r['seconds']>=0 and not r['success'] for r in saved['records']))
        counts=R.accounting(self.root/'results',[job]);self.assertEqual(counts['failed_records'],6)
        self.assertEqual(counts['native_abstentions'],0);self.assertTrue(counts['complete'])
    def test_r_timeout_preserves_progress_and_python_still_runs(self):
        job=self.jobs()[0];self.execute(job,timeout=True,completed={('RKDS','H13')})
        records=R.load_checkpoint(job)['records']
        self.assertEqual(sum(r['success'] for r in records),4)
        self.assertTrue(all(r['timeout_scope']=='shared_R_stage' for r in records if not r['success']))
    def test_changed_data_hash_blocks_retry_before_overwrite(self):
        job=self.jobs()[0];self.execute(job,error='fixture error');job['retry_failed']=True
        self.execute(job);saved=R.load_checkpoint(job)
        next(r for r in saved['records'] if r['method']==PNL).update(success=False,prediction_native=0,error='fixture failure')
        R.atomic_json(job['checkpoint'],saved);before=Path(job['checkpoint']).read_bytes()
        with self.assertRaisesRegex(RuntimeError,'changed paired observations'):self.execute(job,hash_value='different')
        self.assertEqual(before,Path(job['checkpoint']).read_bytes())
    def test_memory_pool_phases_and_no_sample_reduction(self):
        self.args.suite='all';jobs=self.jobs()
        jobs[1]['memory_gib']=40.
        with mock.patch.object(R,'host_memory',return_value=(64.,62.)),mock.patch.object(R.os,'cpu_count',return_value=64):
            plan=R.resource_plan(self.args,jobs)
        self.assertEqual(plan['workers_effective'],3)
        self.assertEqual([p['memory_worker_limit'] for p in plan['pool_phases']],[28,1])
        self.assertEqual([p['workers'] for p in plan['pool_phases']],[3,1])
        self.assertEqual(R.job_pool_size(60.,56.,32,2.),0)
        self.assertEqual(R.job_pool_size(40.,56.,32,2.),1)
        self.assertTrue(all(j['rows'][0]['n']==200 for j in jobs))
    def test_resource_skip_never_overwrites_core_success(self):
        self.args.methods='core';job=self.jobs()[0];self.execute(job)
        before=copy.deepcopy(R.load_checkpoint(job)['records'])
        self.args.methods='all';job=self.jobs()[0];job['memory_gib']=100.
        R.resource_skip(job,10.);saved=R.load_checkpoint(job)
        for record in before:self.assertIn(record,saved['records'])
        fail=next(r for r in saved['records'] if r['method']==PNL)
        self.assertEqual(fail['status'],'resource_unavailable');self.assertEqual(fail['seconds'],0.)
    def test_atomic_json_failure_preserves_original(self):
        path=self.root/'atomic.json';R.atomic_json(path,dict(old=True));before=path.read_bytes()
        with self.assertRaises(ValueError):R.atomic_json(path,dict(bad=float('nan')))
        self.assertEqual(path.read_bytes(),before);self.assertEqual(list(self.root.glob('*.tmp')),[])
    def test_deadline_and_lock_contract(self):
        previous=signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with R.deadline(.02):signal.pause()
        self.assertEqual(signal.getsignal(signal.SIGALRM),previous)
        with R.campaign_lock(['anm']):
            with self.assertRaises(RuntimeError):
                with R.campaign_lock(['anm','tcep']):pass
            with R.campaign_lock(['nonanm']):pass
    def test_export_partial_snapshot_and_restore_merge(self):
        self.freeze();self.args.methods='core';job=self.jobs()[0];self.execute(job)
        before=copy.deepcopy(R.load_checkpoint(job)['records'])
        with contextlib.redirect_stdout(io.StringIO()):archive=R.export(self.args)
        with zipfile.ZipFile(archive) as zipped:
            manifest=json.loads(zipped.read(R.ARCHIVE_ROOT+'/EXPORT_MANIFEST.json'))
        self.assertFalse(manifest['complete']);self.assertEqual(manifest['missing_records'],1)
        Path(job['checkpoint']).unlink();self.args.methods='pnl';pnl_job=self.jobs()[0];self.execute(pnl_job)
        self.args.methods='all';self.args.inputs=[str(archive)]
        with contextlib.redirect_stdout(io.StringIO()):R.restore(self.args)
        after=R.load_checkpoint(job)
        for record in before:self.assertIn(record,after['records'])
        self.assertEqual(len(after['records']),6);self.assertTrue(after['completed'])
    def test_export_restore_preparation_failure_metadata(self):
        self.freeze();job=self.jobs()[0];self.execute(job,error='fixture preparation failure')
        with contextlib.redirect_stdout(io.StringIO()):archive=R.export(self.args)
        Path(job['checkpoint']).unlink();self.args.inputs=[str(archive)]
        with contextlib.redirect_stdout(io.StringIO()):R.restore(self.args)
        saved=R.load_checkpoint(job)
        self.assertFalse(saved['inputs'][job['rows'][0]['case_id']]['prepared'])
        self.assertEqual(len(saved['records']),6)
    def test_shell_syntax_and_version_isolation(self):
        subprocess.run(['bash','-n',str(PACKAGE/'scripts/start.sh')],check=True)
        script=(PACKAGE/'scripts/start.sh').read_text()
        self.assertIn('rkds-paper-',script);self.assertIn('start_new_session=True',script)
        self.assertNotIn('nonanm_h13_randomcause_v2',script)
        self.assertIn('KillMode=mixed',script)
    def test_root_entrypoint_and_shell_location(self):
        result=subprocess.run([sys.executable,str(PACKAGE/'run.py'),'--help'],
                              cwd=self.root,capture_output=True,text=True,check=True)
        self.assertIn('preflight',result.stdout)
        self.assertIn('--decision-rule',result.stdout)
        subprocess.run(['bash','-n',str(PACKAGE/'scripts/setup.sh')],check=True)
    def test_shell_launcher_without_options(self):
        launcher=self.root/'scripts/start.sh';launcher.parent.mkdir(parents=True)
        launcher.write_text((PACKAGE/'scripts/start.sh').read_text())
        python=self.root/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        python.chmod(0o755)
        result=subprocess.run(['bash',str(launcher),'verify'],cwd=PACKAGE,
                              capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),['-u',str(self.root/'run.py'),'verify'])
    def test_run_uses_joined_fresh_spawn_pools(self):
        self.freeze();self.args.suite='all';jobs=self.jobs();jobs[1]['memory_gib']=40.
        sizes=[];contexts=[]
        class FakePool:
            def __init__(self,max_workers,initializer,mp_context):sizes.append(max_workers);contexts.append(mp_context.get_start_method())
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def submit(self,fn,job):
                future=Future()
                try:future.set_result(fn(job))
                except BaseException as error:future.set_exception(error)
                return future
        with mock.patch.object(R,'verify'),mock.patch.object(R,'check_versions'),mock.patch.object(R,'jobs_for',return_value=(self.root/'results',jobs)), \
             mock.patch.object(R,'host_memory',return_value=(64.,62.)),mock.patch.object(R.os,'cpu_count',return_value=64), \
             mock.patch.object(R,'ProcessPoolExecutor',FakePool),mock.patch.object(R.subprocess,'Popen',self.fake_r()), \
             mock.patch.object(R,'python_methods',side_effect=self.fake_python()),contextlib.redirect_stdout(io.StringIO()):
            R.run(self.args)
        self.assertEqual(sizes,[3,1]);self.assertEqual(contexts,['spawn','spawn'])
        info=R.read_json(R.control_path(self.root/'results',self.args));self.assertEqual(info['state'],'complete')
        self.assertEqual(info['full_campaign']['records'],30)

if __name__=='__main__':unittest.main()
