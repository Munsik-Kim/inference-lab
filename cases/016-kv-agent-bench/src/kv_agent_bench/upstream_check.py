"""Run the exact pinned upstream NVFP4 test body with its required context.

This is not a pytest-suite run. Only a trusted, SHA-verified upstream function
is compiled; no model-generated code is ever passed here.
"""
def main():
    import argparse
    import ast
    import pathlib
    import time
    from .common import sha_file,save
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if sha_file(args.source)!=args.sha256:raise ValueError('upstream test hash mismatch')
    import torch
    import flashinfer
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import KVCacheLayout
    started=time.monotonic()
    if not any(current_platform.is_device_capability_family(f)for f in (80,90,120)):
        save(args.output,{'status':'SKIP','reason':'exact upstream device condition'});return
    tree=ast.parse(pathlib.Path(args.source).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='test_flashinfer_nvfp4_kv_cache_update')
    node.decorator_list=[];namespace={'torch':torch,'KVCacheLayout':KVCacheLayout,'flashinfer':flashinfer}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_vllm_test_flashinfer.py','exec'),namespace)
    rows=[]
    for layout in ('LBNHC','LBHNC','BLNHC','BLHNC'):
        torch.manual_seed(6101903)
        with set_current_vllm_config(VllmConfig()):namespace[node.name](None,layout)
        rows.append({'layout':layout,'status':'PASS','upstream_query_heads':4,'upstream_kv_heads':2,'head_dim':128})
    torch.cuda.synchronize();save(args.output,{'status':'PASS','rows':rows,'source_sha256':args.sha256,'execution_scope':'exact trusted upstream function body; manual parameter/context harness','full_pytest_suite':'NOT_RUN','qwen_shape_test':'separate kernel_check.json uses32/8 heads','model_forwards':0,'wall_seconds':time.monotonic()-started})

if __name__=='__main__':main()
