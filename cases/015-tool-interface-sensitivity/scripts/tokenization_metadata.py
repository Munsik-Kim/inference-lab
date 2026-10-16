"""Post-hoc input metadata only: context-dependent name BPE and TASK positions."""
import argparse,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,inputs,token_hash,atomic,require
from interface_sensitivity.transform import make_interface
from transformers import AutoTokenizer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1];rt=read(a.private/'runtime.json');tok=AutoTokenizer.from_pretrained(rt['model']['path'],local_files_only=True);saved=read(a.private/'initial_token_ids.json');tasks,messages,tools=inputs(case);out=[]
 for task in tasks:
  for interface in ['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME']:
   spec=make_interface(task,messages[task['id']],tools,interface);text=tok.apply_chat_template(spec['messages'],tools=spec['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];offsets=encoded['offset_mapping'];require(ids==saved[task['id']+'/'+interface],'rendered-text tokenizer mismatch')
   def span(start,end):
    indexes=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    return {'character_start':start,'character_end':end,'overlapping_token_indexes':indexes,'context_token_ids':[ids[i] for i in indexes]}
   names=[]
   for t in spec['tools']:
    name=t['function']['name'];match=re.search(r'"name"\s*:\s*"('+re.escape(name)+')"',text);require(match is not None,'typed tool declaration not found');names.append({'exposed_name':name,**span(*match.span(1))})
   start=text.rindex('TASK: ')
   out.append({'task_id':task['id'],'interface_id':interface,'input_token_sha256':token_hash(ids),'total_input_tokens':len(ids),'task_reference_span':span(start,start+len('TASK: ')),'name_context_spans':names,'fact_evidence_initial_position':None,'fact_evidence_reason':'facts are provided by later tool results, not present in initial archive filler'})
 atomic(case/'provenance/tokenization.posthoc.json',{'phase':'POST_HOC_INPUT_METADATA','model_forwards':0,'tokenizer_revision':rt['model']['revision'],'full_rendered_token_ids_match_registered':32,'records':out,'BPE_scope':'tokens overlapping the name span in full rendered prompt; boundary tokens can include quotes. Not separately tokenized name IDs.'},True)
 print('PASS32context-dependent tokenization metadata records; no model forward')
if __name__=='__main__':main()
