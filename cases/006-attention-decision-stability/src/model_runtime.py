"""Pinned Qwen inference and detached diagnostics. No diagnostics in timing calls."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
from .intervention import ScopedAttention
from .metrics import norm_evidence, pool_norms, score_full
from .reference import sampled_fp32, positions
from .validity import assess

ARMS=('B','A_PUBLIC','V4')
BOUNDARIES=(13,14,18,22,27)


def tensor_hash(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def array(t):
    return t.detach().float().cpu().numpy().astype(np.float64)


class Runtime:
    def __init__(self,snapshot:Path):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.torch=torch;self.registry=ALL_ATTENTION_FUNCTIONS
        torch.set_num_threads(4);torch.manual_seed(606160)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(0).total_memory)
        start=time.perf_counter()
        self.model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                                                       dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-start
        cfg=self.model.config
        if (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)!=(28,16,8,128):
            raise ValueError('Wrong pinned model geometry')
        self.model.eval()

    def ids(self,tokens):
        return self.torch.tensor([tokens],device='cuda',dtype=self.torch.long)

    def forward(self,tokens,arm='B',cache=True):
        """Uninstrumented model boundary. Native own-cache return; no hidden fallback."""
        with self.torch.inference_mode(),ScopedAttention(self.registry,arm):
            return self.model(input_ids=self.ids(tokens),use_cache=cache,logits_to_keep=1)

    def diagnose(self,item,gold_index,arm,semantic,baseline=None):
        torch=self.torch
        finite_attention=[];finite_blocks=[];hidden={};hooks=[];qkv_before=None;qkv_after=None
        ref=None;native_samples=None;actual_samples=None;last_operator=None;layout_checks=[]
        captured_qkv=None;operator_out=None
        def observer(when,layer,q,k,v,out,selected):
            nonlocal qkv_before,qkv_after,ref,native_samples,actual_samples,last_operator,captured_qkv,operator_out
            if when=='before' and layer==13:
                qkv_before=[tensor_hash(x) for x in (q,k,v)]
                layout_checks.append(q.shape==(1,16,item['length'],128) and k.shape==v.shape==(1,8,item['length'],128)
                                     and all(x.dtype==torch.bfloat16 for x in (q,k,v)))
                if baseline is None:
                    ref_gpu,_=sampled_fp32(q,k,v,q.shape[-1]**-.5)
                    ref=array(ref_gpu);del ref_gpu
                else:ref=baseline['reference']
                # Tiny smoke only needs full operands for standalone/capture correspondence.
                if item['split']=='smoke':captured_qkv=[x.detach().cpu().clone() for x in (q,k,v)]
            if when=='after':
                finite_attention.append(bool(torch.isfinite(out).all()))
                if not finite_attention[-1]:raise ValueError(f'BLOCKED_NUMERICAL_VALIDITY full attention {arm} layer{layer}')
                if layer==13:
                    qkv_after=[tensor_hash(x) for x in (q,k,v)]
                    hnd=out.transpose(1,2)
                    layout_checks.append(out.dtype==torch.bfloat16 and out.shape==(1,item['length'],16,128))
                    actual_samples=array(hnd[:,:,positions(item['length'])])
                    last_operator=array(hnd[:,:,-1])
                    if captured_qkv is not None:operator_out=out.detach().cpu().clone()
        def block_hook(index):
            def hook(_module,_args,out):
                finite_blocks.append(bool(torch.isfinite(out).all()))
                if not finite_blocks[-1]:raise ValueError(f'BLOCKED_NUMERICAL_VALIDITY full block {arm} layer{index}')
                if index in BOUNDARIES:hidden[f'block_{index}_output_last']=array(out[:,-1])
            return hook
        def residual_hook(_module,args):
            hidden['layer13_post_attention_residual_last']=array(args[0][:,-1])
        norm_finite=[]
        def norm_hook(_module,_args,out):
            norm_finite.append(bool(torch.isfinite(out).all()))
            if not norm_finite[-1]:raise ValueError('BLOCKED_NUMERICAL_VALIDITY full final norm')
            hidden['final_norm_last']=array(out[:,-1])
        for i,layer in enumerate(self.model.model.layers):hooks.append(layer.register_forward_hook(block_hook(i)))
        hooks.append(self.model.model.layers[13].post_attention_layernorm.register_forward_pre_hook(residual_hook))
        hooks.append(self.model.model.norm.register_forward_hook(norm_hook))
        try:
            with torch.inference_mode(),ScopedAttention(self.registry,arm,observer) as scoped:
                result=self.model(input_ids=self.ids(item['token_ids']),use_cache=False,logits_to_keep=1)
                z=array(result.logits[0,-1]);calls=list(scoped.calls)
        finally:
            for hook in hooks:hook.remove()
        hidden['layer13_attention_before_o_proj_last']=last_operator
        hidden['final_logits']=z
        expected_route={'B':'native_BF16','A_PUBLIC':'sageattn_qk_int8_pv_fp8_cuda','V4':'sageattn_qk_int8_pv_fp16_cuda'}[arm]
        routed=(len(calls)==28 and all(c['q_len']==item['length'] and c['kv_len']==item['length'] for c in calls)
                and [c['layer'] for c in calls]==list(range(28))
                and all(c['route']==(expected_route if c['layer']==13 else 'native_BF16') for c in calls))
        checks={'full_attention_outputs_finite':len(finite_attention)==28 and all(finite_attention),
                'full_block_outputs_finite':len(finite_blocks)==28 and all(finite_blocks),
                'full_final_hidden_finite':len(norm_finite)==1 and all(norm_finite),
                'full_logits_finite':bool(np.isfinite(z).all()),'inputs_unchanged':qkv_before==qkv_after,
                'same_layer13_qkv':qkv_before==(baseline['qkv_hashes'] if baseline is not None else qkv_before),
                'routing_valid':routed,'dtype_layout_valid':bool(layout_checks) and all(layout_checks),
                'answer_interface_valid':item['label_token_ids']==[32,33,34,35] and not item['future_answer_tokens_present'],
                'native_B_validated':semantic.get('native_B_validated'),
                'effective_backend_verified':semantic.get('verified_arms',{}).get(arm)}
        status=assess(checks)
        if not all(checks[k] for k in checks if k not in ('native_B_validated','effective_backend_verified')):
            raise ValueError(f'BLOCKED_SEMANTICS {status}')
        units=[{'head':h,**norm_evidence(actual_samples[:,h],ref[:,h])} for h in range(16)]
        for h,u in enumerate(units):u['last_query']=norm_evidence(actual_samples[:,h,-1],ref[:,h,-1])
        score=score_full(z,item['label_token_ids'],gold_index)
        record={'arm':arm,'option_logits':score['option_logits'],'gold_index':gold_index,'full_lse':score['full_lse'],
                'full_argmax':score['full_argmax'],'label_ids':item['label_token_ids'],'native_logits_dtype':str(result.logits.dtype),
                'score':score,'local_units':units,'local_pooled':pool_norms(units),
                'local_last_query':pool_norms([u['last_query'] for u in units]),'validity':checks,'validity_status':status,
                'qkv_hashes':qkv_before,'attention_routes':calls,'full_finite_check_counts':{'attention':len(finite_attention),'blocks':len(finite_blocks),'final_norm':len(norm_finite)},
                'hidden_differences':{},'native_pair_local':None,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        if baseline is not None:
            record['hidden_differences']={name:norm_evidence(values,baseline['hidden'][name]) for name,values in hidden.items()}
            record['native_pair_local']=norm_evidence(actual_samples,baseline['native_samples'])
        private={'reference':ref,'native_samples':actual_samples,'hidden':hidden,'qkv_hashes':qkv_before,
                 'full_logits':z,'qkv':captured_qkv,'operator_out':operator_out}
        del result
        return record,private
