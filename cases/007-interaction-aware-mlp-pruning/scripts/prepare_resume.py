"""Copy hash-checked completed private stages into a NEW private run; no GPU calls."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--completed-run',type=Path,required=True);p.add_argument('--new-run',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 if a.new_run.resolve().is_relative_to(case) or a.new_run.exists():raise ValueError('New external directory required')
 read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();selection=read(a.completed_run/'selection.json');dev=read(a.completed_run/'development/summary.json')
 if dev['status']!='PASS' or dev['selection_sha256']!=sha(a.completed_run/'selection.json'):raise ValueError('Invalid completed development/selection')
 if selection['protocol_sha256']!=sha(case/'configs/protocol.json'):raise ValueError('Protocol changed')
 for stem,h in selection['calibration_raw_hashes'].items():
  if sha(a.completed_run/'calibrate'/(stem+'.json'))!=h:raise ValueError('Calibration file mismatch')
 if list((a.completed_run/'heldout').glob('c007-*.json')):raise ValueError('This resume preparation is only for pre-heldout block')
 a.new_run.mkdir(parents=True)
 for name in ['smoke','calibrate','development']:
  for f in (a.completed_run/name).rglob('*'):
   if f.is_symlink():raise ValueError('Unexpected private symlink')
  shutil.copytree(a.completed_run/name,a.new_run/name)
  for f in (a.completed_run/name).rglob('*'):
   if f.is_file() and sha(f)!=sha(a.new_run/name/f.relative_to(a.completed_run/name)):raise ValueError('Copy differs')
 shutil.copyfile(a.completed_run/'selection.json',a.new_run/'selection.json')
 with (a.new_run/'resume.json').open('x') as f:json.dump({'kind':'SAME_FROZEN_STUDY_RESUME','copied_stages':['smoke','calibrate','development'],'selection_sha256':sha(a.new_run/'selection.json'),'heldout_retries':0,'gpu_calls':0},f,indent=2)
 print('Private completed stages preserved/copied; next stage is heldout, only when GPU is free.')
if __name__=='__main__':main()
