"""Copy only raw observations/receipts into a new CPU-analysis directory."""
import argparse,shutil
from pathlib import Path

ITEMS=['metrics','checkpoints','input_manifest.json','source_freeze.json','model_manifest.json','runtime_manifest.json','artifact_frozen.json','artifacts.jsonl','build_ledger.jsonl','CAL_diagnostics.jsonl',
       'completion.json','worker_receipt.json','forward_calls.jsonl','events.jsonl','fresh_process_reload.json','smoke_matrix_verification.json','smoke_matrix_receipt.json','supplemental_smoke_registration.json',
       'evaluation_reload.jsonl','RTN.build_complete.json','COMP_QUARTER.build_complete.json','COMP_HALF.build_complete.json','preservation.json']
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.run);dst=Path(a.output)
    missing=[n for n in ITEMS if not (src/n).exists()]
    if missing:raise FileNotFoundError(missing)
    dst.mkdir(parents=True,exist_ok=False)
    for name in ITEMS:
        if (src/name).is_dir():shutil.copytree(src/name,dst/name)
        else:shutil.copyfile(src/name,dst/name)
    print('Copied immutable inputs/receipts only; summary, decision and paired tables were not copied.')
if __name__=='__main__':main()
