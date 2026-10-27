"""Final explicit-root comparison, independent output receipt, no overwriting earlier receipts."""
import argparse,subprocess
from pathlib import Path
from common import read,write,identity
from preserve import snapshot,git
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;before=read(r/'records/preservation_before.json');changes=[];counts={}
for key,old in before['roots'].items():
 current=snapshot(Path(old['root']));counts[key]=len(current['files'])
 if current!=old:changes.append(key)
assert git(Path(before['roots']['original_case']['root']).parents[1])==before['git'],'GIT_STATE_CHANGED'
for path,expected in before['selected_external_files'].items():assert identity(path)==expected,path
for item in before['parallel_tracked_states']:
 q=subprocess.run(['git','-C',item['path'],'status','--porcelain=v1','--untracked-files=no'],capture_output=True,text=True);assert q.stdout==item['status'] and q.returncode==item['returncode'],'PARALLEL_TRACKED_STATE_CHANGED'
write(r/'records/preservation_final.json',{'changed_roots':changes,'counts':counts,'selected_external_equal':True,'Git_and_parallel_tracked_equal':True,'scope':'Same explicit roots as before; no model-shard or unrelated-project content scan'})
assert not changes,changes
print({'changed_roots':changes,'counts':counts})
