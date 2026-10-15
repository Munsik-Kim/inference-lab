"""Copy an explicit set of immutable observations into a fresh analysis folder."""
import argparse
import shutil
from pathlib import Path

INPUTS=['input_manifest.json','reused_artifact_manifest.json','source_freeze.json','selection_frozen.json','completion.json','forward_calls.jsonl','events.jsonl','old_CAL_reference.json','cells']

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.run);dst=Path(a.output)
    missing=[n for n in INPUTS if not (src/n).exists()]
    if missing:raise FileNotFoundError(missing)
    dst.mkdir(parents=True,exist_ok=False)
    for name in INPUTS:
        source=src/name
        if source.is_dir():shutil.copytree(source,dst/name)
        else:shutil.copyfile(source,dst/name)
    print('Copied raw-only analysis inputs; no summary/derived tables copied.')

if __name__=='__main__':main()
