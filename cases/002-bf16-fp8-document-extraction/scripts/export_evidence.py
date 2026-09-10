"""Create public evidence copies without changing measured text or numeric results."""
import argparse,hashlib,json,pathlib,sys
from data import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--work',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--model-paths',type=pathlib.Path,required=True);a=p.parse_args()
    paths=json.loads(a.model_paths.read_text())
    replacements={str(a.root.resolve()):'<CASE_ROOT>',str(a.work.resolve()):'<RUN_ROOT>',str(a.cache.resolve()):'<JIT_CACHE>',str(pathlib.Path(sys.prefix)):'<CASE_ENV>',**{v:f'<{k.upper()}_MODEL>' for k,v in paths.items()}}
    # One user-home replacement catches incidental paths in compiler diagnostics.
    replacements[str(pathlib.Path.home())]='<USER_HOME>'
    def clean(text):
        for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])):text=text.replace(old,new)
        return text
    manifest=[]
    for run in sorted(a.work.iterdir()):
        if not run.is_dir() or not (run/'server.json').exists():continue
        m=json.loads((run/'server.json').read_text())
        target=a.root/'results'/('development' if m['stage']=='dev' else '')/run.name;target.mkdir(parents=True,exist_ok=True)
        for name in ['server.json','server.log','memory.jsonl','requests.jsonl']:
            source=run/name
            if not source.exists():continue
            raw=source.read_bytes();public=clean(raw.decode()).encode()
            (target/name).write_bytes(public)
            manifest.append({'file':str((target/name).relative_to(a.root)),'raw_sha256':hashlib.sha256(raw).hexdigest(),'public_sha256':hashlib.sha256(public).hexdigest(),'path_labels_replaced':public!=raw})
    dump(a.root/'results/export_manifest.json',{'redaction':'Literal local path prefixes replaced by stable labels. Model-generated text, timing, errors, warnings, process exit codes and request IDs preserved. Original files remain in the private run directory.','files':manifest})
    print('Exported',len(manifest),'evidence files.')

if __name__=='__main__':main()
