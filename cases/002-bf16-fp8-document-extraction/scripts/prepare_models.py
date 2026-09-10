"""Download only the two pinned official checkpoints into a user-chosen cache."""
import argparse,hashlib,json,pathlib
from huggingface_hub import snapshot_download

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--verify-weights',action='store_true');a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];models=json.loads((root/'provenance/models.json').read_text());paths={}
    for label,m in models.items():
        paths[label]=snapshot_download(m['repo_id'],revision=m['revision'],cache_dir=a.cache_dir,allow_patterns=[f['path'] for f in m['files']],token=False,max_workers=3)
        for f in m['files']:
            file=pathlib.Path(paths[label])/f['path'];assert file.stat().st_size==f['size'],f['path']
            if a.verify_weights and f.get('lfs'):
                with file.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                assert actual==f['lfs']['sha256'],f['path']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(paths,indent=2)+'\n')
    print('Pinned checkpoints ready. Local paths are in',a.output)

if __name__=='__main__':main()
