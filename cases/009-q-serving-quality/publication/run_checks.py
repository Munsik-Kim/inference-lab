"""Bounded CPU publication checks, also usable from the standalone ZIP."""
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from restore_review import restore
from verify_publication import verify,CASE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    def run(name,args,extra=None):
        e=env.copy();e.update(extra or {})
        with (out/(name+'.log')).open('w') as f:cp=subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,cwd=out,env=e)
        results.append({'name':name,'exit_code':cp.returncode});(out/'checks.json').write_text(json.dumps({'status':'RUNNING' if not cp.returncode else 'FAIL','checks':results},indent=2)+'\n')
        if cp.returncode:raise RuntimeError(name+' failed; see external log')
    v=verify(root);(out/'identity.json').write_text(json.dumps(v,indent=2)+'\n')
    restore(root,out/'historical');old=out/'historical'/CASE
    run('historical-scalars',[sys.executable,'-B',old/'scripts/verify_public.py','--case',old,'--output',out/'historical-scalars.json'])
    run('historical-tests',[sys.executable,'-B','-m','unittest','discover','-s',old/'tests','-v'])
    run('publication-tests',[sys.executable,'-B','-m','unittest','discover','-s',root/CASE/'publication/tests','-v'])
    run('source-package-tests',[sys.executable,'-B','-m','unittest','discover','-s',root/'packages/diova-compare/tests','-v'],{'PYTHONPATH':str(root/'packages/diova-compare/src')})
    run('posthoc',[sys.executable,'-B',root/CASE/'publication/posthoc/analyze.py','--case',root/CASE,'--output',out/'posthoc.json'])
    assert json.loads((out/'posthoc.json').read_text())==json.loads((root/CASE/'publication/posthoc/analysis.json').read_text())
    shutil.copytree(root/'packages/diova-compare',out/'package-build-source')
    run('wheel-build',[sys.executable,'-B','-m','pip','wheel','--no-deps','--no-build-isolation',out/'package-build-source','--wheel-dir',out/'wheels'])
    wheel=next((out/'wheels').glob('diova_compare-0.1.1-*.whl'))
    with zipfile.ZipFile(wheel) as z,zipfile.ZipFile(root/'downloads/diova_compare-0.1.1-py3-none-any.whl') as public:
        assert z.testzip() is None and 'Version: 0.1.1\n' in z.read('diova_compare-0.1.1.dist-info/METADATA').decode()
        for name in z.namelist():
            if name.startswith('diova_compare/') and name.endswith('.py'):assert z.read(name)==public.read(name)
    run('create-install-env',[sys.executable,'-m','venv',out/'installed'])
    py=out/'installed/bin/python'
    run('wheel-install',[py,'-B','-m','pip','install','--no-index','--no-deps',wheel])
    run('wheel-origin',[py,'-B','-c',"import diova_compare;print(diova_compare.__version__);print(diova_compare.__file__)"])
    origin=(out/'wheel-origin.log').read_text();assert str(out/'installed') in origin and str(root/'packages') not in origin
    run('installed-package-tests',[py,'-B','-m','unittest','discover','-s',root/'packages/diova-compare/tests','-v'])
    run('official-cli',[py,'-B',root/CASE/'publication/check_cli.py','--root',root,'--python',py,'--output',out/'official'])
    (out/'wheel-checksum.json').write_text(json.dumps({'filename':wheel.name,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'bytes':wheel.stat().st_size,'scope':'Fresh CI/local build; ZIP timestamps can differ from published wheel. Runtime source bytes matched.'},indent=2)+'\n')
    verify(root)
    (out/'checks.json').write_text(json.dumps({'status':'PASS','checks':results,'new_gpu_runs':0,'historical_full_split_retries':0,'source_and_installed_unique_tests':37,'publication_tests':10,'historical_study_tests':24,'scope':'CPU identity, original restoration, point metrics, posthoc, wheel build/install and lifecycle ledger. Not GPU reproduction.'},indent=2)+'\n')
    print('PASS',len(results),'CPU check stages')
if __name__=='__main__':main()
