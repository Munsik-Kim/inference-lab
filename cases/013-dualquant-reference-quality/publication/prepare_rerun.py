"""Prepare a new local rerun directory; no model loading, conversion or inference."""
from pathlib import Path
import argparse, hashlib, json, shutil
CASE=Path(__file__).resolve().parents[1]

def prepare(output,models,wiki_tokens):
    output=Path(output);recorded=json.loads((CASE/'records/experiment_manifest.json').read_text())
    if output.exists() or output.resolve().is_relative_to(CASE.resolve()):raise ValueError('Choose a new external directory')
    for key in ('06b','4b'):
        model=Path(models[key]); tokens=Path(wiki_tokens[key])
        for name,row in recorded['models'][key]['tokenizer_files'].items():
            p=model/name
            if not p.is_file() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Model config/tokenizer identity mismatch: '+key+'/'+name)
        row=recorded['inputs']['wiki_'+key+'.npy']
        if not tokens.is_file() or tokens.stat().st_size!=row['bytes'] or hashlib.sha256(tokens.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Original WikiText token-array identity required: '+key)
        recorded['models'][key]['path']=str(model.resolve())
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists():raise ValueError('Partial directory exists; inspect it first')
    (temporary/'inputs').mkdir(parents=True)
    for name,row in recorded['inputs'].items():
        if name.startswith('wiki_'):
            source=Path(wiki_tokens['06b' if '06b' in name else '4b'])
        elif name=='wikitext_source.json':
            continue # public path-redacted copy is provenance, not a rerun input
        else:source=CASE/'records/inputs'/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen input mismatch: '+name)
        shutil.copyfile(source,temporary/'inputs'/name)
    recorded['inputs'].pop('wikitext_source.json')
    (temporary/'experiment_manifest.json').write_text(json.dumps(recorded,indent=2)+'\n')
    shutil.copyfile(CASE/'records/reference_contract.json',temporary/'reference_contract.json')
    temporary.rename(output)
    return {'status':'PREPARED_ONLY','model_loaded':False,'inference':False,'output':str(output)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model06b',required=True,type=Path);p.add_argument('--model4b',required=True,type=Path)
    p.add_argument('--wiki06b',required=True,type=Path);p.add_argument('--wiki4b',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.output,{'06b':a.model06b,'4b':a.model4b},{'06b':a.wiki06b,'4b':a.wiki4b})))
