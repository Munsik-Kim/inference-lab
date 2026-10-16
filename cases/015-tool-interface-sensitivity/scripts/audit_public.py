"""Model-free aggregate/cell/source/public-table verification for review copies."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,rows,digest,require,atomic,sha
from interface_sensitivity.prepare import validate
from interface_sensitivity.analysis import summarize
from interface_sensitivity.audit import audit

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1];result=case/'results/p1_r1'
 frozen=validate(case);records=rows(result/'episodes.jsonl');plan=read(result/'planned_cells.json');reg=read(case/'specs/protocol.json')
 manifest=case/'publication/PUBLIC_MANIFEST.json'
 if manifest.exists():
  entries=read(manifest)['files']
  actual={str(f.relative_to(case)) for f in case.rglob('*') if f.is_file() and '__pycache__' not in f.parts and str(f.relative_to(case)) not in ('publication/PUBLIC_MANIFEST.json','ARCHIVE_CONTENTS.json')}
  require(actual==set(entries),'closed public file set mismatch')
  for name,e in entries.items():
   require((case/name).stat().st_size==e['bytes'] and sha(case/name)==e['sha256'],'public file identity mismatch: '+name)
 methods=read(case/'specs/methods.json');source=read(case/'provenance/source_freeze.json')
 for row in records:
  require(row['model_revision']==methods['revision'] and row['model_id']==methods['model_id'],'model identity mismatch')
  require(row['artifact_hash']==methods['methods'][row['method']]['artifact_id'],'method artifact identity mismatch')
  require(row['generation_config_hash']==digest(reg['generation']),'generation contract mismatch')
  require(row['execution_source_sha256']==source['execution_source_sha256'],'mixed execution source')
  require(row['model_calls']<=reg['generation']['max_model_calls'] and row['tool_attempts']<=reg['generation']['max_tool_calls'],'episode budget exceeded')
  require(all(r['output_tokens']<=reg['generation']['max_new_tokens'] and r['input_tokens']+reg['generation']['max_new_tokens']<=reg['generation']['context_limit'] for r in row['responses']),'request limit exceeded')
 cells=[read(f) for f in sorted((result/'workers').glob('*/attempt01/cells/*.json'))]
 require(len(cells)==len(records)==96,'incomplete primary matrix')
 require({r['episode_id']:digest(r) for r in cells}=={r['episode_id']:digest(r) for r in records},'cell/aggregate mismatch')
 expected_events=[{'episode_id':r['episode_id'],**e} for r in records for e in r['tool_events']]
 require(expected_events==rows(result/'tool_events.jsonl'),'event sidecar mismatch')
 summary=summarize(records,plan,reg['bootstrap_replicates'],reg['bootstrap_seed'])
 require(summary==read(result/'summary.json'),'raw-derived summary mismatch')
 audit_receipt=audit(case,result/'episodes.jsonl')
 require(audit_receipt==read(result/'independent_audit.json'),'raw replay audit mismatch')
 artifacts=read(case/'provenance/artifact_identity.json')
 for m,e in artifacts.items():
  require(e['projection_count']==len(e['per_projection_identity'])==252,'external artifact metadata coverage')
  require(sum(r['bytes'] for r in e['per_projection_identity'])==e['payload_bytes'],'artifact byte metadata mismatch')
 demo=read(case/'demo/data.json');require(demo['summary']==summary and demo['episodes']==records,'explorer data mismatch')
 for lang in ['ko','en']:
  text=(case/'demo'/(lang+'.html')).read_text();require(text.count('class="episode"')==96,'explorer episode count mismatch')
 value={'status':'PASS','primary_cells_and_aggregate':96,'tool_events':len(expected_events),'task_vectors':8,'methods':3,'interfaces':4,
  'independent_replay':audit_receipt,'frozen_source':frozen,'external_artifact_metadata_files':504,'external_weights_decoded':False,
  'bilingual_explorer_matches_records':True,'source_scope':'CPU stored raw response, fixture state and paired scalar audit; no new model forward'}
 atomic(a.output,value,True);print(json.dumps(value))
if __name__=='__main__':main()
