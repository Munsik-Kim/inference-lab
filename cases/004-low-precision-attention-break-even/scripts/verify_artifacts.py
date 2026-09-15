"""Read-only publication audit; independent scalar summary calculations."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CASE, fingerprint, sha, verify_spec, write_json


def percentile(values, q):
    ordered=sorted(values)
    index=(len(ordered)-1)*q
    low=math.floor(index);high=math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)


def same(a,b):
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--model-only',action='store_true')
    a=p.parse_args();checks={}
    spec,spec_sha=verify_spec();checks['frozen_protocol']='PASS'
    if a.snapshot:
        model=json.loads((CASE/'provenance/model.json').read_text())
        assert a.snapshot.name==model['revision']
        for item in model['files']:
            assert sha(a.snapshot/item['name'])==item['verified_sha256'],item['name']
        checks['model_files_sha256']={'status':'PASS','count':len(model['files'])}
    if a.model_only:
        assert a.snapshot,'Model-only requires --snapshot'
        print(json.dumps(checks));return
    manifest=json.loads((CASE/'inputs/manifest.json').read_text());docs=manifest['documents']
    assert len({d['document_id'] for d in docs})==24
    assert len({d['text_sha256'] for d in docs})==24
    assert len({d['prefix_sha256']['512'] for d in docs})==24
    assert sum(d['split']=='confirmation' for d in docs)==16
    for d in docs:
        assert sha(CASE/d['text_file'])==d['text_sha256']
        for n in [512,2048,4096]:assert fingerprint(d['token_ids_4096'][:n])==d['prefix_sha256'][str(n)]
    checks['input_hashes_and_split']='PASS'
    cap=json.loads((CASE/'provenance/confirmation_capture.json').read_text())
    assert len(cap['traces'])==48 and cap['started_at_utc']>spec['frozen_at_utc']
    for t in cap['traces']:
        assert t['all_query_positions'] and t['query_count']==t['length']
        assert t['hq']==16 and t['hkv']==8 and t['layer']==13
        assert t['status']=='VALID' and t['native_replay_relative_error']<=.01
        assert t['spec_sha256']==spec_sha
    checks['full_capture_geometry_and_native_replay']='PASS'
    summary=json.loads((CASE/'results/selection_table.json').read_text())
    assert summary['spec_sha256']==spec_sha
    rounds=[];rows=[];unit_count=0
    for n in range(5):
        path=CASE/f'results/confirmation/round-{n}.json';r=json.loads(path.read_text());rounds.append(r)
        assert r['round']==n and r['status']=='COMPLETED' and not r['errors']
        assert r['evidence_kind']=='gpu_measurement' and not r['mock']
        assert r['spec_sha256']==spec_sha and r['environment_fingerprint']==spec['environment_fingerprint']
        assert summary['source_hashes'][str(path.relative_to(CASE))]==sha(path)
        assert len(r['measurements'])==288
        for m in r['measurements']:
            assert m['status']=='OK' and m['inputs_unchanged'] and not m['mock']
            assert len(m['wall_ms'])==len(m['event_ms'])==20
            assert all(math.isfinite(v) and v>0 for v in m['wall_ms']+m['event_ms'])
            if m['backend']=='kernel_only':assert m['public_wrapper_bitwise_equal']
            for u in m.get('errors_by_head',[]):
                assert u['invalid_rows']==0 and not u['near_zero']
                err=math.fsum(x*x for x in u['row_error_norms'])
                ref=math.fsum(x*x for x in u['row_reference_norms'])
                same(math.sqrt(err/ref),u['relative_output_error'])
                same(math.sqrt(err/(32*m['shape']['dim'])),u['absolute_rms_error'])
                same(u['dot_sum']/math.sqrt(u['actual_squared_sum']*ref),u['cosine'])
                unit_count+=1
            rows.append(m)
    index={(m['shape']['id'],m['round'],m['document_id'],m['backend']):m for m in rows}
    assert len(index)==len(rows)
    comparisons=0
    for e in summary['entries']:
        sid=e['shape']['id'];pairs=[]
        for m in rows:
            if m['shape']['id']!=sid or m['backend']!=e['baseline']:continue
            other=index[sid,m['round'],m['document_id'],'sage']
            assert m['input_sha256']==other['input_sha256']
            pairs.extend(x/y for x,y in zip(m['wall_ms'],other['wall_ms']))
        same(statistics.median(pairs),e['speed']['speedup'])
        for backend,z in e['backend_summaries'].items():
            selected=[m for m in rows if m['shape']['id']==sid and m['backend']==backend]
            for key,outkey in [('wall_ms','wall_block_mean_ms'),('event_ms','event_block_mean_ms')]:
                values=[v for m in selected for v in m[key]]
                same(statistics.median(values),z[outkey]['median']);same(percentile(values,.95),z[outkey]['p95']);comparisons+=2
            units=[u for m in selected if m['round']==0 for u in m.get('errors_by_head',[])]
            if units:
                values=[u['relative_output_error'] for u in units]
                same(statistics.median(values),z['numerical']['median']);same(percentile(values,.95),z['numerical']['p95']);comparisons+=2
                assert z['numerical']['units']==len(units)
        if e['shape']['family']=='qwen':
            assert e['confirmation_document_count']==16 and e['primary_unit_count']==256
    checks['raw_records']={'status':'PASS','processes':5,'records':len(rows),'unit_recalculations':unit_count}
    checks['independent_scalar_aggregates']={'status':'PASS','comparisons':comparisons,'speedup_shapes':len(summary['entries'])}
    report={'scope':'Static evidence audit; no new GPU measurements','spec_sha256':spec_sha,'checks':checks}
    if a.output:write_json(a.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
