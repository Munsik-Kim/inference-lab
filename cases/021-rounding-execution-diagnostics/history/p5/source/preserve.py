"""Explicit bounded roots, no symlink traversal and no model-shard sweep."""
import argparse,os,json,subprocess
from pathlib import Path
from common import read,write,identity
def snapshot(base):
    regular={};links={};physical=set()
    for folder,dirs,files in os.walk(base,followlinks=False):
        for name in dirs+files:
            p=Path(folder)/name
            if p.is_symlink():links[str(p.relative_to(base))]=os.readlink(p)
        for name in files:
            p=Path(folder)/name
            if p.is_file() and not p.is_symlink():
                stat=p.stat();physical.add((stat.st_dev,stat.st_ino));regular[str(p.relative_to(base))]=identity(p)
    return {'root':str(base),'files':regular,'symlinks':links,'unique_physical_regular_files':len(physical)}
def git(work):
    commands={'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--porcelain=v1','--untracked-files=all'],'staged':['diff','--cached','--binary'],'tracked':['diff','--binary'],'worktrees':['worktree','list','--porcelain']}
    return {k:subprocess.run(['git','-C',str(work),*v],check=True,capture_output=True,text=True).stdout for k,v in commands.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['before','after'],required=True);a=p.parse_args();r=a.root;loc=read(r/'references/locations.json');p4=Path(loc['p4_root'])
    roots={key:snapshot(Path(loc[key])) for key in ['original_case','original_private','p4_root','p4_replay']}
    g=git(loc['original_worktree']);parallel=[]
    for entry in g['worktrees'].strip().split('\n\n'):
        path=Path(entry.splitlines()[0].removeprefix('worktree '))
        if path.exists():
            q=subprocess.run(['git','-C',str(path),'status','--porcelain=v1','--untracked-files=no'],capture_output=True,text=True);parallel.append({'path':str(path),'status':q.stdout,'returncode':q.returncode})
    ext=read(p4/'references/external_inputs.json');selected={}
    for row in ext['artifacts']:
        if row['projection'] in ['model.layers.0.self_attn.q_proj','model.layers.25.self_attn.q_proj']:
            selected[row['path']]=identity(row['path']);assert selected[row['path']]=={k:row[k] for k in ['bytes','sha256']}
    for path,expected in ext['existing_runtime_sources'].items():selected[path]=identity(path);assert selected[path]==expected
    record={'roots':roots,'git':g,'parallel_tracked_states':parallel,'selected_external_files':selected,'model_weight_scope':'NO model weight read/hash in P5; only two RPI4 artifacts and selected runtime source/binary','external_paths_are_references_not_copies':True}
    write(r/f'records/preservation_{a.phase}.json',record)
    if a.phase=='after':
        before=read(r/'records/preservation_before.json');assert before==record,'PRESERVATION_MISMATCH'
        write(r/'summary/preservation.json',{'status':'UNCHANGED','regular_file_counts':{k:len(v['files']) for k,v in roots.items()},'unique_physical_counts_by_nonoverlapping_root':{k:v['unique_physical_regular_files'] for k,v in roots.items()},'symlink_counts':{k:len(v['symlinks']) for k,v in roots.items()},'Git_and_parallel_tracked_status_equal':True,'selected_external_files':selected,'model_weight_current_hash':'NOT_RUN','scope':'Explicit roots only; no global disk or other-project content audit'})
    print(json.dumps({'phase':a.phase,'counts':{k:len(v['files']) for k,v in roots.items()},'bytes':{k:sum(x['bytes'] for x in v['files'].values()) for k,v in roots.items()}}))
if __name__=='__main__':main()
