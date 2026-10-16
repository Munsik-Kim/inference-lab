"""Display-only repair around the byte-preserved frozen A1 renderer.

The original Python triple-quoted template emitted a literal newline inside a
JavaScript single-quoted string. Keep its source/raw data intact, escape that
separator and emit a separately checkable external script for the review view.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from order_regression.render import render
from order_regression.common import read,sha,require,atomic


def repair_html(text):
    broken=".join('\n\n')";fixed=".join('\\n\\n')"
    require(text.count(broken)==1,'unexpected frozen display template; refuse broad replacement')
    repaired=text.replace(broken,fixed,1)
    scripts=re.findall(r'<script>(.*?)</script>',repaired,re.S)
    require(len(scripts)==1,'one executable script required')
    result=repaired.replace('<script>'+scripts[0]+'</script>','<script src="explorer.js" defer></script>',1)
    result=result.replace('</style>','.turn p{overflow-wrap:anywhere}</style>',1)
    return result,scripts[0]


def render_review(root,output):
    root=Path(root);output=Path(output);render(root,output)
    data_before=sha(output/'data.json');originals={};scripts=[]
    for lang in ['ko','en']:
        path=output/(lang+'.html');originals[lang]=sha(path)
        html,script=repair_html(path.read_text());scripts.append(script);path.write_text(html,encoding='utf-8')
    require(scripts[0]==scripts[1],'bilingual script mismatch')
    (output/'explorer.js').write_text(scripts[0],encoding='utf-8')
    require(sha(output/'data.json')==data_before,'raw display data changed')
    value={'status':'PASS','display_version':'A1_REVIEW_VIEWER_V2','source_run_id':'a1_r1','new_model_forwards':0,
        'frozen_renderer_sha256':sha(root/'src/order_regression/render.py'),'wrapper_sha256':sha(Path(__file__)),
        'execution_source_sha256':read(root/'provenance/source_freeze.json')['execution_source_sha256'],
        'original_generated_html_sha256':originals,'raw_data_sha256_unchanged':data_before,
        'transformation':'escape one JS newline separator; externalize executable script; wrap long observation hashes; no record/score changes',
        'files':{name:sha(output/name) for name in ['ko.html','en.html','explorer.js','data.json']}}
    atomic(output/'viewer_manifest.json',value,True);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(render_review(ROOT,a.output)))
