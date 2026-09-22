"""CPU consistency check; this cannot reproduce GPU lifecycle behavior."""
def validate(d):
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    need(d['original_full_split_retries']==0,'historical retry scope')
    rows=d['ledger'];keys=[(r['config'],r['arm'],r['path'],r['repeat']) for r in rows]
    need(len(keys)==len(set(keys)) and len(rows)<=24,'attempt coverage')
    run=[r for r in rows if r['status']!='NOT_RUN_BUDGET'];need(d['attempts']==len(run),'attempt denominator')
    for r in run:
        ok=r['returncode']==0 and r['result_valid'] and r['worker_clean'] and r['worker_exitcodes'] and all(x==0 for x in r['worker_exitcodes']) and not r['timeout'] and not r['cleanup_owned_workers'] and r['stages'][-1]=='complete'
        need((r['status']=='PASS')==bool(ok),'pass propagation')
    need(d['clean']==sum(r['status']=='PASS' for r in run),'clean count')
    need(d['failed']==sum(r['status']=='FAIL' for r in run),'failed count')
    need(d['not_run']==sum(r['status']=='NOT_RUN_BUDGET' for r in rows),'not-run count')
    need(d['walltime_seconds']<=3600+30,'bounded walltime')
    need(d['status']!='HISTORICAL_FAILURE_FIXED','unsupported lifecycle conclusion')
    return True
