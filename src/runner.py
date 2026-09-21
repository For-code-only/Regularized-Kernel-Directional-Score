#!/usr/bin/env python3
"""Frozen, native-decision paper experiments with case-level resumable checkpoints."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, ExitStack
import csv
import gzip
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = 'rkds_paper_experiments'
THREAD_ENV = ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']
for name in THREAD_ENV: os.environ[name] = '1'
CORE = ['RKDS','ANM_GP','ANM_KRR','RECI_MON3','KCDC']
R_METHODS = {'RKDS','RECI_MON3','KCDC'}
STOP = False
MODULES = {}
PROTOCOLS = {}

def utc(): return datetime.now(timezone.utc).isoformat()
def read_json(path): return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def protocol():
    # Frozen during a process lifetime; do not reread this file per method/record.
    path=ROOT/'config/protocol.json'
    if path not in PROTOCOLS:PROTOCOLS[path]=read_json(path)
    return PROTOCOLS[path]
def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w') as f:
            json.dump(value,f,allow_nan=False,separators=(',',':'))
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if hasattr(value,'item'):return clean(value.item())
    return value
def load_module(name,path):
    if name not in MODULES:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);MODULES[name]=module
    return MODULES[name]

SUITES = ('anm','tcep','sensitivity','nonanm')
SENSITIVITY_CONFIGS = ('H13','H13_theta_1e-08','H13_theta_1e-07','H13_theta_1e-05',
    'H13_theta_1e-04','H13_theta_1e-03','H13_sigma1_1e-01','H13_sigma1_4e-01',
    'H13_sigma2_1e-01','H13_sigma2_4e-01','H13_sigma3_8e-01','H13_sigma3_3.2e+00')

def fingerprint(require=True):
    path=ROOT/'MANIFEST.json'
    if path.exists():return sha(path)
    if require:raise RuntimeError('MANIFEST.json is missing; production execution is disabled.')
    inputs=['run.py','src/runner.py','src/r/worker.R','src/r/scientific_core.R','config/protocol.json','config/manifests/smoke.csv.gz']
    return 'engineering-'+hashlib.sha256(''.join(sha(ROOT/p) for p in inputs if (ROOT/p).exists()).encode()).hexdigest()
def verify():
    frozen=read_json(ROOT/'MANIFEST.json')
    changed=[]
    for name,digest in frozen['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe frozen manifest path')
        if not (ROOT/name).is_file() or sha(ROOT/name)!=digest:changed.append(name)
    if changed:raise RuntimeError('Frozen input/code mismatch: '+', '.join(changed[:10]))
    return frozen
def csv_rows(name):
    if name not in (*SUITES,'smoke'):raise ValueError('Unknown manifest')
    with gzip.open(ROOT/'config/manifests'/f'{name}.csv.gz','rt',encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    integers={'n','replicate','function_id','cause_id','noise_id','truth','generation_attempts',
              'structure_id','pair_id','subsample_seed','cap'}
    for row in rows:
        for key,value in list(row.items()):
            if value!='' and (key in integers or key.endswith('_seed') or key=='method_seed_base'):
                try:row[key]=int(value)
                except ValueError:
                    if key.endswith('_seed') or key=='method_seed_base':raise
            elif key in ['rho','weight','noise_level'] and value!='':row[key]=float(value)
        if name!='smoke':row['suite']=name
        else:
            row['suite']=row.get('smoke_suite',row.get('suite'))
            if row.get('suite') not in SUITES:raise ValueError('Smoke rows must identify their suite')
    if len({row['case_id'] for row in rows})!=len(rows):raise ValueError('Duplicate manifest case IDs')
    return rows
def selected_suites(args):
    suites=list(SUITES) if args.suite in ['all','smoke'] else [args.suite]
    return [suite for suite in suites if args.methods!='pnl' or suite!='sensitivity']
def is_smoke(args):return args.command=='smoke' or args.suite=='smoke'
def method_names(methods='all',suite='anm'):
    if suite=='sensitivity':return [] if methods=='pnl' else ['RKDS']
    return CORE+[protocol()['method_pnl']] if methods=='all' else CORE if methods=='core' else [protocol()['method_pnl']]
def expected_keys(methods='all',suite='anm'):
    if suite=='sensitivity':
        return [] if methods=='pnl' else [('RKDS',cfg) for cfg in SENSITIVITY_CONFIGS]
    return [(method,'H13' if method=='RKDS' else 'D1' if method=='KCDC' else 'default')
            for method in method_names(methods,suite)]
def host_memory():
    if Path('/proc/meminfo').exists():
        fields={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
        total=fields['MemTotal'];available=fields.get('MemAvailable',total)
        limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
        if limit.exists() and limit.read_text().strip().isdigit():
            cap=int(limit.read_text());total=min(total,cap)
            available=min(available,max(0,cap-int(current.read_text()) if current.exists() else cap))
        return total/2**30,available/2**30
    try:
        total=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))/2**30
        return total,total
    except Exception:return 8.,8.
def peak_gib(children=False):
    raw=resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    return raw/(2**30 if sys.platform=='darwin' else 2**20)

def worker_baseline(methods='all'):
    return 2.0 if methods=='all' else 1.5 if methods=='core' else 1.25
def reservation(n,methods='all'):
    return max(worker_baseline(methods),(1.6 if methods=='all' else .75 if methods=='core' else 1.25)+96*int(n)**2/2**30)
def job_pool_size(memory_gib,budget,workers,baseline):
    if max(memory_gib,baseline)>budget:return 0
    if memory_gib>budget/2:return 1
    return max(1,min(workers,int(budget//baseline),int((budget-max(memory_gib,baseline))//baseline)+1))
@contextmanager
def deadline(seconds):
    def expired(*_):raise TimeoutError(f'Method exceeded {seconds} second limit')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
@contextmanager
def campaign_lock(suites,smoke=False):
    import fcntl
    with ExitStack() as stack:
        for suite in sorted(set(suites)):
            path=ROOT/('verification/smoke_run' if smoke else 'results')/suite/'run.lock'
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open('a+'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError(f'The {suite} suite is already running; wait for it to stop.')
        yield
def init_worker():
    for key in THREAD_ENV:os.environ[key]='1'
    signal.signal(signal.SIGINT,signal.SIG_IGN)
def convert(row,meta,method,out,fp):
    # Native decisions only. Do not manufacture p values for score-only methods.
    out={k:v for k,v in out.items() if 'forced' not in k and k!='raw_direction'}
    sf=out.get('score_forward',out.get('pforward'));sr=out.get('score_reverse',out.get('preverse'))
    try:sf=float(sf);sr=float(sr);valid=math.isfinite(sf) and math.isfinite(sr)
    except (ValueError,TypeError):sf=sr=None;valid=False
    error=str(out.get('error') or '');state=out.get('status','directed')
    if not valid and not error:
        error=f'Missing or nonfinite directional scores; status={state}'
        out['error']=error
    success=valid and not error and state not in ['implementation_failure','software_failure','invalid_input','timeout','resource_unavailable','algorithm_error']
    native=int(out.get('prediction',out.get('decision',0)) or 0) if success else 0
    if native not in [0,1,2]:raise ValueError('Invalid native direction encoding')
    swapped=bool(meta.get('swapped',False));mapped=lambda x:3-x if swapped and x else x
    configuration=out.get('configuration','H13' if method=='RKDS' else 'D1' if method=='KCDC' else 'default')
    result=dict(row);result.pop('requested_methods',None);result.pop('requested_configurations',None)
    result.update(method=method,configuration=configuration,direction_encoding='12',
        truth=int(meta.get('truth',row.get('truth',1))),prediction_native=mapped(native),
        presented_prediction_native=native,score_forward=sr if swapped else sf,
        score_reverse=sf if swapped else sr,presented_score_forward=sf,presented_score_reverse=sr,
        success=success,status=state,error=error,undirected_reason='' if native else out.get('undirected_reason',state),
        data_hash=meta.get('data_hash'),swapped=swapped,n_used=meta.get('n_used',row.get('n')),
        seconds=out.get('seconds'),timing_scope=out.get('timing_scope','method_elapsed'),
        peak_python_gib=peak_gib(),peak_child_gib=peak_gib(True),protocol_hash=fp,details=out)
    for k,v in out.items():
        if any(s in k for s in ['denominator','condition_number','lambda_min','lambda_max','self_hsic','screen_p']):result[k]=v
    if swapped:
        for k in list(result):
            reverse=k[:-8]+'_reverse'
            if k.endswith('_forward') and reverse in result and k not in ['score_forward','presented_score_forward']:
                result[k],result[reverse]=result[reverse],result[k]
    result['diagnostic_coordinate_system']='original; nested details retain presented coordinates'
    if 'pforward' in out or 'preverse' in out:
        result['p_forward']=out.get('preverse') if swapped else out.get('pforward')
        result['p_reverse']=out.get('pforward') if swapped else out.get('preverse')
    return clean(result)

def python_methods(path,row,role='all',timeout=14400,requested=None,on_result=None):
    import numpy as np
    requested=set(['ANM_GP','ANM_KRR',protocol()['method_pnl']] if requested is None else requested)
    output={}
    for method in ['ANM_GP','ANM_KRR',protocol()['method_pnl']]:
        if method not in requested:continue
        start=time.monotonic();caught=[]
        try:
            with deadline(timeout),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if method==protocol()['method_pnl']:
                    adapter=load_module('paper_selected_pnl',ROOT/'src/methods/pnl.py')
                    pair=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
                    result=adapter.run_pair(pair,int(row['pnl_seed']))
                else:
                    gp=load_module('paper_old_gp',ROOT/'src/methods/anm_gp.py')
                    pair=gp.load_pair(path,False,np)
                    seed=int(row['gp_seed'] if method=='ANM_GP' else row['krr_seed'])
                    random.seed(seed);np.random.seed(seed)
                    if method=='ANM_GP':
                        from causallearn.search.FCMBased.ANM.ANM import ANM
                        pf,pr=map(float,ANM().cause_or_effect(pair[:,[0]],pair[:,[1]]))
                        if not all(math.isfinite(p) and 0<=p<=1 for p in [pf,pr]):raise ValueError('Invalid residual p values')
                        prediction,state=gp.decision(pf,pr)
                        result=dict(prediction=prediction,status=state,pforward=pf,preverse=pr,error='')
                    else:
                        from scipy.spatial.distance import pdist
                        from sklearn.kernel_ridge import KernelRidge
                        from causallearn.utils.KCI.KCI import KCI_UInd
                        krr=load_module('paper_krr',ROOT/'src/methods/anm_krr.py')
                        result=krr.run_pair(pair,seed,krr.load_settings(ROOT/'config/krr_settings.json'),np,pdist,KernelRidge,KCI_UInd)
        except Exception as e:
            result=dict(prediction=0,status='timeout' if isinstance(e,TimeoutError) else 'implementation_failure',error=f'{type(e).__name__}: {e}')
        result.update(seconds=time.monotonic()-start,warnings=[str(w.message) for w in caught],configuration='default')
        output[method]=result
        if on_result is not None:on_result(method,result)
    return output


def case_complete(saved,row,methods='all'):
    keys={(record['method'],record['configuration']) for record in saved['records']}
    return set(expected_keys(methods,row['suite'])).issubset(keys)
def validate_checkpoint(saved,job):
    row=job['rows'][0];case=row['case_id'];expected=set(expected_keys('all',job['suite']))
    if (saved.get('fingerprint')!=job['fingerprint'] or saved.get('job_id')!=job['id']
        or saved.get('role')!='all' or saved.get('suite')!=job['suite']):
        raise ValueError('Checkpoint protocol/job/suite mismatch')
    keys=[]
    for record in saved.get('records',[]):
        key=(record.get('method'),record.get('configuration'));keys.append(key)
        if record.get('case_id')!=case or key not in expected or record.get('protocol_hash')!=job['fingerprint']:
            raise ValueError('Checkpoint record outside frozen case/method keys')
        if record.get('prediction_native') not in [0,1,2] or not isinstance(record.get('success'),bool):
            raise ValueError('Invalid checkpoint result encoding')
        duration=record.get('seconds')
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
            raise ValueError('Invalid checkpoint elapsed-time record')
        if record['success']:
            if not record.get('data_hash') or any(not isinstance(record.get(k),(int,float)) or not math.isfinite(record[k]) for k in ['score_forward','score_reverse']):
                raise ValueError('Successful checkpoint lacks finite scores or paired-data hash')
            if record['method'] in ['ANM_GP','ANM_KRR',protocol()['method_pnl']] and any(
                not isinstance(record.get(k),(int,float)) or not math.isfinite(record[k]) or not 0<=record[k]<=1 for k in ['p_forward','p_reverse']):
                raise ValueError('Successful residual-test checkpoint lacks valid p values')
        elif record['prediction_native']!=0 or not record.get('error'):
            raise ValueError('Failed methods require direction zero and an explicit error')
        canonical=saved.get('inputs',{}).get(case,{}).get('data_hash')
        if canonical and record.get('data_hash') and canonical!=record['data_hash']:
            raise ValueError('Checkpoint paired-data hash mismatch')
    if len(keys)!=len(set(keys)) or set(saved.get('completed',[]))-{case} or set(saved.get('inputs',{}))-{case}:
        raise ValueError('Checkpoint duplicate/unexpected case keys')
    if bool(saved.get('completed'))!=(set(keys)==expected):
        raise ValueError('Checkpoint completion marker disagrees with all-method keys')
    return saved
def load_checkpoint(job):
    path=Path(job['checkpoint'])
    if path.exists():return validate_checkpoint(read_json(path),job)
    return dict(job_id=job['id'],case_id=job['rows'][0]['case_id'],fingerprint=job['fingerprint'],
                suite=job['suite'],role='all',records=[],inputs={},completed=[])
def target_keys(job,saved):
    previous={(r['method'],r['configuration']):r for r in saved['records']}
    return {key for key in expected_keys(job['methods'],job['suite'])
            if key not in previous or job.get('retry_failed') and not previous[key]['success']}
def is_complete(job):
    return not target_keys(job,load_checkpoint(job))
def persist_case(job,saved,previous,meta):
    case=job['rows'][0]['case_id']
    saved['records']=[previous[key] for key in sorted(previous)]
    saved['inputs'][case]=meta
    saved['completed']=[case] if set(previous)==set(expected_keys('all',job['suite'])) else []
    saved.update(updated_utc=utc(),peak_python_gib=peak_gib(),peak_child_gib=peak_gib(True))
    atomic_json(job['checkpoint'],saved)
def replace_record(job,saved,previous,result,meta):
    key=(result['method'],result['configuration'])
    if key in previous and previous[key]['success']:
        raise RuntimeError('Successful results must never be overwritten')
    if key in previous:
        path=Path(job['checkpoint']).parent.parent/'attempt_history'
        atomic_json(path/f"{job['id']}_{result['method']}_{result['configuration']}_{time.time_ns()}.json",
                    dict(case_id=result['case_id'],previous=previous[key],replacement_utc=utc(),fingerprint=job['fingerprint']))
    previous[key]=result;persist_case(job,saved,previous,meta)
def run_job(job):
    saved=load_checkpoint(job);row=job['rows'][0];case=row['case_id']
    previous={(r['method'],r['configuration']):r for r in saved['records']}
    targets=target_keys(job,saved)
    if not targets:return dict(id=job['id'],suite=job['suite'],skipped=True)
    target_methods={key[0] for key in targets};started=time.monotonic()
    def record_result(method,out,meta):
        result=convert(row,meta,method,out,job['fingerprint']);key=(method,result['configuration'])
        if key not in targets:raise RuntimeError(f'Worker returned unrequested scientific key {key}')
        if method in R_METHODS:
            result['r_stage_time_limit_seconds']=job['timeout'];result['timeout_scope']='shared_R_stage'
        else:
            result['method_time_limit_seconds']=job['timeout'];result['timeout_scope']='per_Python_method'
        replace_record(job,saved,previous,result,meta)
    with tempfile.TemporaryDirectory(prefix='paper-case-',dir=job['scratch']) as temp:
        temp=Path(temp)
        request=dict(row,requested_methods=','.join(sorted(target_methods&R_METHODS)),
                     requested_configurations=','.join(sorted(cfg for method,cfg in targets if method=='RKDS')))
        atomic_json(temp/'case.json',[request])
        command=[job['rscript'],str(ROOT/'src/r/worker.R'),str(temp/'case.json'),str(temp),job['methods']]
        log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True);r_started=time.monotonic()
        with log.open('a') as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);timed_out=False
            try:code=proc.wait(timeout=job['timeout'])
            except subprocess.TimeoutExpired:
                timed_out=True;proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code=proc.returncode
            except BaseException:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        elapsed=time.monotonic()-r_started
        prepared=read_json(temp/'1.json') if (temp/'1.json').exists() else dict(
            preparation_error=f'R preparation did not finish (exit={code}, timeout={timed_out}); see {log}',
            preparation_status='timeout' if timed_out else 'implementation_failure')
        meta=prepared.get('metadata') or saved['inputs'].get(case,{})
        old_hash=saved['inputs'].get(case,{}).get('data_hash')
        if old_hash and meta.get('data_hash') and old_hash!=meta['data_hash']:
            raise RuntimeError('Retry changed paired observations')
        if not prepared.get('preparation_error') and (not meta.get('data_hash') or not (temp/'1.csv').exists()):
            prepared['preparation_error']='R preparation is incomplete: canonical observations/hash unavailable'
            prepared['preparation_status']='timeout' if timed_out else 'implementation_failure'
        if prepared.get('preparation_error'):
            meta=dict(meta,preparation_error=prepared['preparation_error'],preparation_failed_this_attempt=True,
                      prepared=bool(meta.get('data_hash')),generation_attempts=1)
            for method,cfg in sorted(targets):
                record_result(method,dict(configuration=cfg,status=prepared.get('preparation_status','invalid_input'),
                    error=prepared['preparation_error'],seconds=elapsed,
                    timing_scope='shared_preparation_failure_elapsed_not_additive'),meta)
        else:
            found=set()
            for name,out in (prepared.get('outputs') or {}).items():
                method=out.get('method',name)
                cfg=out.get('configuration','H13' if method=='RKDS' else 'D1' if method=='KCDC' else 'default')
                found.add((method,cfg));record_result(method,out,meta)
            for method,cfg in sorted({key for key in targets if key[0] in R_METHODS}-found):
                record_result(method,dict(configuration=cfg,
                    status='timeout' if timed_out else 'implementation_failure',
                    error=f'R method/configuration did not finish (exit={code})',seconds=elapsed,
                    timing_scope='shared_R_stage_failure_elapsed_not_additive'),meta)
            persist_case(job,saved,previous,meta)
            wanted=target_methods-R_METHODS
            if wanted:python_methods(temp/'1.csv',row,job['methods'],job['timeout'],requested=wanted,
                on_result=lambda method,out:record_result(method,out,meta))
    return dict(id=job['id'],suite=job['suite'],seconds=time.monotonic()-started,
                recomputed_keys=sorted(targets),peak_python_gib=peak_gib())
def resource_skip(job,budget):
    saved=load_checkpoint(job);row=job['rows'][0]
    previous={(r['method'],r['configuration']):r for r in saved['records']}
    meta=saved['inputs'].get(row['case_id'],dict(prepared=False,resource_not_executed=True))
    for method,cfg in sorted(target_keys(job,saved)):
        out=dict(configuration=cfg,status='resource_unavailable',seconds=0.,timing_scope='not_executed',
                 error=f"Estimated {job['memory_gib']:.2f} GiB exceeds {budget:.2f} GiB task budget; exact full data not run")
        replace_record(job,saved,previous,convert(row,meta,method,out,job['fingerprint']),meta)

def selected_rows(args):
    if is_smoke(args):
        suites=selected_suites(args);return [row for row in csv_rows('smoke') if row['suite'] in suites]
    return [row for suite in selected_suites(args) for row in csv_rows(suite)]
def jobs_for(args,fp):
    base=ROOT/'verification/smoke_run'/fp[-12:] if is_smoke(args) else ROOT/'results'
    jobs=[]
    for row in selected_rows(args):
        suite=row['suite'];jid=hashlib.sha256(row['case_id'].encode()).hexdigest()[:20];directory=base/suite
        jobs.append(dict(id=jid,rows=[row],suite=suite,methods=args.methods,role='all',fingerprint=fp,
            checkpoint=str(directory/'checkpoints'/f'{jid}.json'),scratch=str(directory/'scratch'),
            log=str(directory/'logs'/f'{jid}.log'),rscript=args.rscript,timeout=args.timeout,
            retry_failed=args.command=='retry',memory_gib=reservation(row['n'],args.methods)))
    return base,jobs
def resource_plan(args,jobs):
    total,available=host_memory();budget=max(0.,min(total,available)-args.reserve_gib)
    if args.memory_budget_gib is not None:budget=min(budget,args.memory_budget_gib)
    requested=min(args.workers or max(1,(os.cpu_count() or 2)//2),os.cpu_count() or 1)
    baseline=worker_baseline(args.methods);phases={};skipped=[]
    for job in jobs:
        size=job_pool_size(job['memory_gib'],budget,requested,baseline)
        if not size:skipped.append(dict(suite=job['suite'],case_id=job['rows'][0]['case_id'],n=job['rows'][0]['n'],estimated_gib=job['memory_gib']))
        else:
            phase=phases.setdefault(size,dict(workers=size,jobs=0,pool_baseline_gib=size*baseline,largest_job_gib=0.))
            phase['jobs']+=1;phase['largest_job_gib']=max(phase['largest_job_gib'],job['memory_gib'])
    for phase in phases.values():
        phase['memory_worker_limit']=phase['workers']
        phase['workers']=min(phase['workers'],phase['jobs'])
        phase['pool_baseline_gib']=phase['workers']*baseline
    return dict(host_total_gib=total,available_gib=available,reserve_gib=args.reserve_gib,
        task_budget_gib=budget,workers_requested=requested,workers_effective=max((p['workers'] for p in phases.values()),default=0),
        persistent_worker_baseline_gib=baseline,pool_phases=[phases[size] for size in sorted(phases,reverse=True)],
        resource_unavailable=skipped,resource_policy='Fresh spawn pool per phase; reserve idle baselines plus active dense extras; never reduce sample size',
        memory_estimate_caveat='Conservative accounting estimate, not a hard OS memory limit; inspect observed peak memory.')
def accounting(base,jobs,validate=True,methods='all'):
    counts=dict(expected_cases=len(jobs),expected_records=sum(len(expected_keys(methods,j['suite'])) for j in jobs),
                checkpoints=0,records=0,completed_cases=0,successful_cases=0,
                failed_records=0,native_abstentions=0,directed_records=0)
    for job in jobs:
        if not Path(job['checkpoint']).exists():continue
        saved=load_checkpoint(job) if validate else read_json(job['checkpoint']);counts['checkpoints']+=1
        expected=set(expected_keys(methods,job['suite']))
        rows=[r for r in saved['records'] if (r['method'],r['configuration']) in expected]
        keys={(r['method'],r['configuration']) for r in rows};complete=keys==expected
        counts['records']+=len(rows);counts['completed_cases']+=complete
        counts['successful_cases']+=complete and all(r['success'] for r in rows)
        counts['failed_records']+=sum(not r['success'] for r in rows)
        counts['native_abstentions']+=sum(r['success'] and r['prediction_native']==0 for r in rows)
        counts['directed_records']+=sum(r['success'] and r['prediction_native'] in [1,2] for r in rows)
    counts['missing_records']=counts['expected_records']-counts['records']
    counts['complete']=counts['completed_cases']==len(jobs)
    counts['all_successful']=counts['complete'] and counts['failed_records']==0
    return counts
def control_path(base,args):
    return base/'control'/f'{args.suite}_{args.methods}.json'
def live_process(info):
    if info.get('state') not in ['running','stopping']:return False
    try:
        pid=int(info['pid']);os.kill(pid,0);cmd=Path(f'/proc/{pid}/cmdline')
        return not cmd.exists() or str(ROOT/'run.py').encode() in cmd.read_bytes()
    except (OSError,ValueError,KeyError):return False
def run(args):
    global STOP
    STOP=False
    if not is_smoke(args):verify();check_versions(args.methods)
    fp=fingerprint(not is_smoke(args));base,jobs=jobs_for(args,fp)
    if not jobs:
        print(json.dumps(dict(state='nothing_selected',suite=args.suite,methods=args.methods)));return
    for job in jobs:Path(job['scratch']).mkdir(parents=True,exist_ok=True)
    with campaign_lock(selected_suites(args),is_smoke(args)):
        pending=[job for job in jobs if not is_complete(job)];plan=resource_plan(args,pending)
        budget=plan['task_budget_gib'];baseline=plan['persistent_worker_baseline_gib'];phases={};attempts={}
        if pending and budget<baseline:raise RuntimeError('Available task memory is below a persistent-worker reservation; no task started.')
        def stopping(*_):
            global STOP
            STOP=True
        signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
        info=dict(suite=args.suite,suites=selected_suites(args),methods=args.methods,pid=os.getpid(),
                  state='running',started_utc=utc(),fingerprint=fp,**plan,total_jobs=len(jobs),
                  completed_jobs=len(jobs)-len(pending),active_jobs=0,pending_jobs=len(pending),
                  infrastructure_errors=[],resource_skipped_jobs=0)
        path=control_path(base,args);atomic_json(path,info)
        for job in pending:
            size=job_pool_size(job['memory_gib'],budget,plan['workers_requested'],baseline)
            if not size:
                resource_skip(job,budget);info['resource_skipped_jobs']+=1;info['completed_jobs']+=1
            else:phases.setdefault(size,[]).append(job)
        phase_sizes=sorted(phases,reverse=True)
        for phase_index,size in enumerate(phase_sizes):
            if STOP:break
            pending=phases[size];size=min(size,len(pending));active={};extra_used=0.;pool_baseline=size*baseline
            later=sum(len(phases[key]) for key in phase_sizes[phase_index+1:])
            info.update(pool_workers=size,pool_baseline_gib=pool_baseline,
                        reserved_task_gib=pool_baseline,pending_jobs=len(pending)+later)
            atomic_json(path,info)
            # Joining all previous workers releases imported Torch and allocator arenas
            # before a large full-data phase. Idle processes are never free memory.
            with ProcessPoolExecutor(max_workers=size,initializer=init_worker,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                while pending or active:
                    if STOP and not active:break
                    index=0
                    while pending and not STOP and index<len(pending) and len(active)<size:
                        job=pending[index];extra=max(0.,job['memory_gib']-baseline)
                        if pool_baseline+extra_used+extra<=budget+1e-9:
                            pending.pop(index);active[pool.submit(run_job,job)]=job;extra_used+=extra
                        else:index+=1
                    if not active:
                        if pending and not STOP:raise RuntimeError('Memory scheduler cannot admit its phase; no busy wait.')
                        continue
                    done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=active.pop(future);extra_used-=max(0.,job['memory_gib']-baseline)
                        try:
                            print(json.dumps(future.result()),flush=True);info['completed_jobs']+=1
                        except Exception as error:
                            key=(job['suite'],job['id']);attempts[key]=attempts.get(key,0)+1
                            info['infrastructure_errors'].append(dict(suite=job['suite'],job=job['id'],error=str(error),attempt=attempts[key],utc=utc()))
                            if attempts[key]<=1 and not STOP:pending.append(job)
                            else:atomic_json(base/job['suite']/'failed_jobs'/f"{job['id']}.json",dict(error=str(error),job=job))
                    if not active:extra_used=0.
                    info.update(active_jobs=len(active),pending_jobs=len(pending)+later,updated_utc=utc(),
                                reserved_task_gib=pool_baseline+max(0.,extra_used),state='stopping' if STOP else 'running')
                    atomic_json(path,info)
        counts=accounting(base,jobs,methods=args.methods)
        state='stopped' if STOP else 'complete' if counts['all_successful'] else 'complete_with_method_failures' if counts['complete'] else 'incomplete_infrastructure_errors'
        info.update(counts,state=state,finished_utc=utc(),active_jobs=0,pool_workers=0,
                    pool_baseline_gib=0.,reserved_task_gib=0.,full_campaign=accounting(base,jobs))
        atomic_json(path,info);print(json.dumps(info,indent=2))
def check_versions(methods='all'):
    from importlib import metadata
    expected={'numpy':'2.3.5','scipy':'1.17.0','scikit-learn':'1.8.0','causal-learn':'0.1.4.8'}
    if methods!='core':expected['torch']='2.8.0'
    actual={name:metadata.version(name) for name in expected}
    if any(actual[key].split('+')[0]!=value for key,value in expected.items()):
        raise RuntimeError(f'Pinned scientific environment mismatch: {actual}')
    if methods!='core':
        adapter=load_module('paper_selected_pnl',ROOT/'src/methods/pnl.py')
        if hasattr(adapter,'check_environment'):adapter.check_environment()
    return actual
def setup(args):
    versions=check_versions(args.methods)
    # Prevent multiple launchers from racing the shared compiled-permutation cache.
    with campaign_lock(['setup']):
        subprocess.run([args.rscript,str(ROOT/'src/r/worker.R'),'setup'],cwd=ROOT,check=True)
    report=dict(utc=utc(),python=sys.version,platform=sys.platform,versions=versions,cpus=os.cpu_count(),
        memory_gib=host_memory(),threads={name:os.environ[name] for name in THREAD_ENV},
        r=subprocess.check_output([args.rscript,'--version'],text=True,stderr=subprocess.STDOUT).strip())
    atomic_json(ROOT/'logs'/f'environment_{args.methods}.json',report);print(json.dumps(report,indent=2))
def preflight(args):
    base,jobs=jobs_for(args,fingerprint(False))
    report=dict(suite=args.suite,methods=args.methods,cases=len(jobs),
        selected_records=sum(len(expected_keys(args.methods,j['suite'])) for j in jobs),
        full_records=sum(len(expected_keys('all',j['suite'])) for j in jobs),cases_per_job=1,
        r_stage_timeout_seconds=args.timeout,python_method_timeout_seconds=args.timeout,
        **resource_plan(args,jobs))
    print(json.dumps(report,indent=2));return report
def status(args):
    base,jobs=jobs_for(args,fingerprint(False));runs=[]
    for path in sorted((base/'control').glob('*.json')) if (base/'control').exists() else []:
        info=read_json(path)
        if not set(info.get('suites',[]))&set(selected_suites(args)):continue
        if info.get('state') in ['running','stopping'] and not live_process(info):
            info['state']='interrupted_resume_required'
        runs.append(info)
    report=dict(suite=args.suite,methods=args.methods,runs=runs,
                selected=accounting(base,jobs,methods=args.methods),full_campaign=accounting(base,jobs))
    print(json.dumps(report,indent=2));return report
def stop(args):
    base=ROOT/'verification/smoke_run'/fingerprint(False)[-12:] if is_smoke(args) else ROOT/'results';stopped=[]
    for path in sorted((base/'control').glob('*.json')) if (base/'control').exists() else []:
        info=read_json(path)
        if set(info.get('suites',[]))&set(selected_suites(args)) and live_process(info):
            os.kill(int(info['pid']),signal.SIGTERM);stopped.append(info['pid'])
    print(json.dumps(dict(stop_requested=sorted(set(stopped)),message='Current cases finish; no new cases start. Wait for stopped before resume.')))

def export(args):
    frozen=verify();fp=fingerprint();base,jobs=jobs_for(args,fp)
    allowed={str(Path(job['checkpoint']).relative_to(ROOT)):job for job in jobs}
    names=set(frozen['files'])|{'MANIFEST.json'}
    for directory in [base/suite for suite in selected_suites(args)]+[ROOT/'logs',base/'control']:
        if not directory.exists():continue
        for path in directory.rglob('*'):
            if path.is_file() and 'scratch' not in path.parts and not path.name.endswith(('.tmp','.lock')):
                names.add(str(path.relative_to(ROOT)))
    target=ROOT/'exports';target.mkdir(exist_ok=True)
    path=target/f'{ROOT.name}_{args.suite}_results_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{time.time_ns()%1000000:06d}.zip'
    temporary=path.with_suffix('.zip.tmp');hashes={};analysis_state={}
    completed=records=failed=abstentions=0
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            def write_bytes(name,data):
                hashes[name]=hashlib.sha256(data).hexdigest()
                archive.writestr(ARCHIVE_ROOT+'/'+name,data)
            # Read one atomic checkpoint at a time: never retain every record or
            # all archive bytes in RAM. Counts and hashes describe these exact bytes.
            for name in sorted(names):
                if '/checkpoints/' in name and name.startswith('results/') and name not in allowed:
                    raise ValueError(f'Unexpected checkpoint in result snapshot: {name}')
                data=(ROOT/name).read_bytes()
                if name in allowed:
                    job=allowed[name];saved=validate_checkpoint(json.loads(data),job)
                    completed+=case_complete(saved,job['rows'][0])
                    records+=len(saved['records']);failed+=sum(not r['success'] for r in saved['records'])
                    abstentions+=sum(r['success'] and r['prediction_native']==0 for r in saved['records'])
                write_bytes(name,data)
            if (ROOT/'analysis').exists():
                for summary_path in sorted((ROOT/'analysis').rglob('summary.json')):
                    directory=summary_path.parent
                    try:
                        summary_bytes=summary_path.read_bytes();summary=json.loads(summary_bytes)
                        analyzed={item['path']:item['sha256'] for item in summary['checkpoint_snapshot']['files']}
                        selected=set(summary.get('suites',[summary['suite']] if summary.get('suite') else []))
                        observed={name:digest for name,digest in hashes.items()
                                  if name.startswith('results/') and '/checkpoints/' in name
                                  and (not selected or name.split('/')[1] in selected)}
                        current=bool(analyzed) and summary['protocol_fingerprint']==fp and analyzed==observed
                    except (KeyError,ValueError,TypeError):current=False
                    analysis_state[str(directory.relative_to(ROOT))]='included_current_snapshot' if current else 'omitted_stale_or_unverifiable'
                    if current:
                        for item in directory.rglob('*'):
                            if item.is_file() and not item.name.endswith('.tmp'):
                                name=str(item.relative_to(ROOT))
                                if name not in hashes:write_bytes(name,summary_bytes if item==summary_path else item.read_bytes())
            expected=sum(len(expected_keys('all',job['suite'])) for job in jobs)
            report=dict(exported_utc=utc(),fingerprint=fp,suites=selected_suites(args),
                complete=completed==len(jobs),all_successful=completed==len(jobs) and failed==0,
                expected_cases=len(jobs),completed_cases=completed,expected_records=expected,records=records,
                missing_records=expected-records,failed_records=failed,native_abstentions=abstentions,
                snapshot=True,analysis_state=analysis_state,files=hashes)
            archive.writestr(ARCHIVE_ROOT+'/EXPORT_MANIFEST.json',(json.dumps(report,indent=2)+'\n').encode())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    print(json.dumps(dict(path=str(path),sha256=sha(path),complete=report['complete'],
        all_successful=report['all_successful'],missing_records=report['missing_records'],analysis_state=analysis_state),indent=2))
    return path
def restore(args):
    verify();fp=fingerprint();base,jobs=jobs_for(args,fp)
    allowed={(job['suite'],job['id']):job for job in jobs}
    if not args.inputs:raise ValueError('restore requires --inputs result.zip [...]')
    with campaign_lock(selected_suites(args)), tempfile.TemporaryDirectory(prefix='paper-restore-') as staging:
        staged={};histories={};preserved=0
        for archive_path in args.inputs:
            with zipfile.ZipFile(archive_path) as archive:
                prefix=ARCHIVE_ROOT+'/'
                if hashlib.sha256(archive.read(prefix+'MANIFEST.json')).hexdigest()!=fp:
                    raise ValueError('Restore frozen protocol differs from local package')
                manifest=json.loads(archive.read(prefix+'EXPORT_MANIFEST.json'))
                if manifest.get('fingerprint')!=fp:raise ValueError('Restore export fingerprint mismatch')
                if len(archive.namelist())!=len(set(archive.namelist())):raise ValueError('Duplicate ZIP members')
                for name in archive.namelist():
                    member=PurePosixPath(name)
                    if member.is_absolute() or '..' in member.parts or not name.startswith(prefix):
                        raise ValueError('Unsafe ZIP member')
                    relative=name[len(prefix):];parts=PurePosixPath(relative).parts
                    if len(parts)!=4 or parts[0]!='results' or parts[1] not in selected_suites(args) or parts[2] not in ['checkpoints','attempt_history']:continue
                    raw=archive.read(name)
                    if manifest['files'].get(relative)!=hashlib.sha256(raw).hexdigest():
                        raise ValueError('Restore member checksum mismatch')
                    if parts[2]=='attempt_history':
                        stage=Path(staging)/'history'/hashlib.sha256(relative.encode()).hexdigest()
                        atomic_json(stage,json.loads(raw));histories[relative]=stage;continue
                    data=json.loads(raw);key=(parts[1],data.get('job_id'))
                    if key not in allowed or parts[3]!=f"{key[1]}.json":raise ValueError('Unexpected restore checkpoint path/job')
                    job=allowed[key];validate_checkpoint(data,job);local=read_json(staged[key]) if key in staged else load_checkpoint(job)
                    case=job['rows'][0]['case_id'];a=local['inputs'].get(case,{});b=data['inputs'].get(case,{})
                    if a.get('data_hash') and b.get('data_hash') and a['data_hash']!=b['data_hash']:
                        raise ValueError('Restore paired observations conflict')
                    indexed={(record['method'],record['configuration']):record for record in local['records']}
                    for record in data['records']:
                        record_key=(record['method'],record['configuration']);old=indexed.get(record_key)
                        if old and old['success']:
                            science=['truth','prediction_native','score_forward','score_reverse','p_forward','p_reverse','data_hash','success','status']
                            if record['success'] and any(old.get(field)!=record.get(field) for field in science):
                                raise ValueError('Conflicting successful restore record')
                            preserved+=1;continue
                        if not old or record['success']:indexed[record_key]=record
                    local['records']=[indexed[item] for item in sorted(indexed)]
                    if not a or not a.get('data_hash'):local['inputs'][case]=b if b else a
                    local['completed']=[case] if set(indexed)==set(expected_keys('all',job['suite'])) else []
                    local['restored_utc']=utc()
                    stage=Path(staging)/'checkpoints'/job['suite']/f"{job['id']}.json"
                    atomic_json(stage,local);staged[key]=stage
        # Validate all archives before any checkpoint changes.
        for key,stage in staged.items():
            data=read_json(stage)
            job=allowed[key];path=Path(job['checkpoint'])
            if path.exists():atomic_json(path.parent.parent/'attempt_history'/f'{job["id"]}_pre_restore_{time.time_ns()}.json',read_json(path))
            atomic_json(path,data)
        for relative,stage in histories.items():
            target=ROOT/relative
            if not target.exists():atomic_json(target,read_json(stage))
    print(json.dumps(dict(restored_checkpoints=len(staged),successful_records_preserved=preserved,
                          **accounting(base,jobs)),indent=2))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['setup','preflight','run','resume','retry','smoke','status','stop','export','restore','analyze','verify'])
    parser.add_argument('--suite',choices=[*SUITES,'all','smoke'],default='all')
    parser.add_argument('--methods',choices=['all','core','pnl'],default='all')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--rscript',default='Rscript');parser.add_argument('--reserve-gib',type=float,default=6.)
    parser.add_argument('--memory-budget-gib',type=float)
    parser.add_argument('--timeout',type=int,default=int(os.getenv('RKDS_TIMEOUT_SECONDS','14400')))
    parser.add_argument('--inputs',nargs='*');parser.add_argument('--partial',action='store_true')
    parser.add_argument('--decision-rule',choices=['native','score'],default='native',help='Analysis only: method rules or direct score comparison')
    args=parser.parse_args()
    if args.decision_rule != 'native' and args.command != 'analyze':
        parser.error('--decision-rule score is only valid with analyze')
    if args.workers<0 or args.reserve_gib<0 or args.timeout<1 or args.memory_budget_gib is not None and args.memory_budget_gib<=0:
        parser.error('Invalid resource option')
    if args.command in ['run','resume','retry','smoke']:run(args)
    elif args.command=='setup':setup(args)
    elif args.command=='preflight':preflight(args)
    elif args.command=='status':status(args)
    elif args.command=='stop':stop(args)
    elif args.command=='export':export(args)
    elif args.command=='restore':restore(args)
    elif args.command=='verify':
        frozen=verify();print(json.dumps(dict(status='verified',fingerprint=fingerprint(),files=len(frozen['files']))))
    elif args.command=='analyze':
        verify()
        command=[sys.executable,str(ROOT/'src/evaluate.py'),'--suite',args.suite,'--decision-rule',args.decision_rule]
        if args.partial:command.append('--allow-partial')
        with campaign_lock(selected_suites(args)):subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
