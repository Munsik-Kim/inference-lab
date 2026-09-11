"""Prepare self-authored synthetic input documents and a pre-development plan."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import random
import sys
import torch
from transformers import AutoTokenizer

CASE = Path(__file__).resolve().parents[1]
TOPICS = ['river survey', 'archive migration', 'queue scheduler', 'orchard irrigation',
          'museum inventory', 'graph traversal', 'ferry timetable', 'library renewal',
          'solar inspection', 'warehouse routing', 'event ledger', 'coastal mapping',
          'theatre maintenance', 'cache eviction', 'grain storage', 'water sampling',
          'schema validation', 'railway signage', 'school equipment', 'interval merging',
          'forest monitoring', 'recycling collection', 'dependency resolution', 'weather station']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def make_document(topic, language, number, seed):
    rng = random.Random(seed)
    paragraphs = [f'Synthetic document {number:02d}: {topic}. All events and identifiers are fictional.']
    for section in range(90):
        a, b, c = [rng.randrange(5, 950) for _ in range(3)]
        stage = rng.choice(['inspection', 'allocation', 'review', 'delivery', 'revision', 'measurement'])
        if language == 'en':
            variants = [
                f'Section {section} records the {stage} for {topic}. The north team counted {a} units, while the southern register showed {b}. These figures use different reporting windows and should not be added. A separate sample of {c} entries will be reconciled after the original records are checked.',
                f'The {topic} working group reviewed batch {number}-{section}. An earlier estimate of {a} was retained for comparison, but the accepted count is {b}. The remaining {c} items have not been classified. The coordinator asked that provisional notes remain visible beside the confirmed entry so a later reviewer can reconstruct the decision.',
                f'During {stage}, the instrument recorded {a}, {b}, and {c} in that order. The middle observation used a replacement component. Staff therefore kept the measurements as separate entries. For {topic}, the next report should state the collection method, the period covered, and any missing observations before drawing a comparison.',
                f'Appendix {section} describes a disagreement about {topic}. One proposal used a limit of {a}; another suggested {b}. Neither proposal was approved. The existing threshold of {c} remains in effect until a signed revision is recorded. A request to investigate the difference does not itself change the operating rule.'
            ]
            paragraphs.append(rng.choice(variants))
        elif language == 'ko':
            variants = [
                f'{section}번 기록은 {topic}의 점검 내역이다. 첫 조사에서 {a}개를 확인했고 재검사에서는 {b}개가 나왔다. 조사 구역이 달라 두 수치를 합산하지 않았다. 누락된 {c}개 항목은 다음 검토에 포함한다. 담당자는 원본 자료와 변경 사유를 함께 보관하기로 했다.',
                f'문서 {number}-{section}의 검토 회의에서는 {a}라는 예상치를 논의했다. 최종 승인된 값은 {b}이며 나머지 {c}건은 아직 제안 상태다. 이전 기록은 삭제하지 않는다. 승인되지 않은 제안이 현재 규칙으로 오해되지 않도록 상태와 날짜를 구분해서 적었다.',
                f'{topic} 담당 조는 {stage} 과정에서 장비를 교체했다. 교체 전 값 {a}와 교체 후 값 {b}는 동일한 조건의 측정이 아니다. 검증용 표본 {c}개를 별도로 확보했다. 자료를 인용할 때는 측정 조건과 예외 처리 이유를 함께 설명해야 한다.',
                f'이번 {section}차 작업에는 상반된 의견이 남아 있다. 일부는 기준을 {a}로 바꾸자고 했고 다른 조는 {b}를 제안했다. 변경은 확정되지 않았으므로 기존 기준 {c}를 유지한다. 추가 조사 요청과 변경 승인은 서로 다른 절차라는 점을 회의록에 명시했다.'
            ]
            paragraphs.append(rng.choice(variants))
        else:
            variants = [
                f'def collect_{number}_{section}(records):\n    # {topic}: keep accepted records and preserve their original order.\n    limit = {a}\n    result = []\n    for record in records:\n        if record.get("status") != "accepted":\n            continue\n        value = record.get("value", {b})\n        if value <= limit:\n            result.append((record["id"], value + {c}))\n    return result\n',
                f'def reconcile_{number}_{section}(left, right):\n    # {topic}: missing is different from zero.\n    merged = dict(left)\n    for key, value in right.items():\n        if value is None:\n            continue\n        old = merged.get(key, {a})\n        merged[key] = min(old + value, {b})\n    assert len(merged) <= len(left) + len(right)\n    return sorted(merged.items(), key=lambda item: (item[1], item[0]))\n',
                f'class Window_{number}_{section}:\n    """Bounded ledger for {topic}; entries are fictional."""\n    def __init__(self):\n        self.capacity = {a}\n        self.entries = []\n    def append(self, item):\n        self.entries.append(item)\n        if len(self.entries) > self.capacity:\n            self.entries.pop(0)\n    def total(self):\n        return sum(x.get("amount", {b}) for x in self.entries if x.get("confirmed", False))\n'
            ]
            paragraphs.append(rng.choice(variants))
    return '\n\n'.join(paragraphs) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-cache', type=Path, required=True)
    args = p.parse_args()
    model = json.loads((CASE / 'provenance/model.json').read_text())
    snapshot = args.model_cache / ('models--' + model['model_id'].replace('/', '--')) / 'snapshots' / model['revision']
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    records = []
    for i, topic in enumerate(TOPICS):
        split = 'dev' if i < 8 else 'eval'
        language = ['en', 'ko', 'code'][i % 3]
        docid = f'{split}-{i if i < 8 else i - 8:02d}'
        seed = (310000 if split == 'dev' else 320000) + i
        text = make_document(topic, language, i, seed)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) >= 4096
        (CASE / 'inputs' / f'{docid}.txt').write_text(text)
        records.append({'document_id': docid, 'split': split, 'language': language,
                        'topic': topic, 'seed': seed, 'text_sha256': sha(text.encode()),
                        'full_token_count': len(ids), 'token_ids_4096': ids[:4096],
                        'prefix_sha256': {str(n): sha(json.dumps(ids[:n], separators=(',', ':')).encode())
                                          for n in [512, 2048, 4096]}})
    assert len({r['text_sha256'] for r in records}) == 24
    assert len({tuple(r['token_ids_4096'][:512]) for r in records}) == 24
    write_json(CASE / 'inputs/manifest.json', {'origin': 'Self-authored synthetic documents generated by prepare.py; Apache-2.0. No external customer documents.',
               'tokenizer': model['model_id'], 'revision': model['revision'],
               'input_format': 'Plain text token prefixes, no chat template or generated completion.',
               'limitation': 'Distinct scenario/document IDs and seeds; shared generation rules. Prefix lengths of a document are dependent.',
               'documents': records})
    plan = {'paper': 'https://arxiv.org/html/2609.09721v1', 'model_id': model['model_id'], 'model_revision': model['revision'],
            'lengths': [512, 2048, 4096], 'layers': [0, 13, 27], 'heads': [0, 5, 10, 15],
            'query_positions': {str(n): [i * (n - 1) // 15 for i in range(16)] for n in [512, 2048, 4096]},
            'tile': 128, 'primary_block': 32, 'sensitivity_blocks': [16, 64],
            'methods': ['dense_fp32', 'online_fp32', 'nearest_ceil', 'nearest_efq_scale', 'efq_mmlu', 'efq_mean', 'efq_balance', 'efq_lut', 'efq_calibrated'],
            'primary_methods': ['efq_mean', 'efq_calibrated'], 'sensitivity_methods': ['nearest_efq_scale', 'efq_mean', 'efq_calibrated'],
            'calibration': {'tau': [-3.6, -3.3, -3.06, -2.9, -2.6, -2.3, -2.1], 'h': [1.7, 2., 2.3, 2.5, 2.7, 3., 3.3],
                'candidate_count': 49, 'length': 2048, 'layer': 13, 'heads': [0, 10], 'documents': 8,
                'objective': 'sqrt(sum squared output errors / sum squared FP32 reference outputs), pooled over the fixed development subset',
                'ties': 'First minimum in ascending listed tau then h order; no refinement.'},
            'dtype': {'checkpoint_and_trace': 'bfloat16', 'audit_qk_probability_v_and_accumulators': 'float32', 'small_oracle': 'float64', 'tf32': False},
            'scale': {'layout': 'Consecutive key elements within one head/query row, reset at each attention tile.',
                'exponent_range': [-127, 127], 'out_of_range': 'Clamp exponent, recompute code with that scale; count below-range blocks and represented zero scales.',
                'nearest_ceil': 'ceil((local shifted maximum - ln(6))/ln(2)); headroom scale. Independent MXFP4 numerical reference, not verified paper baseline.',
                'nearest_efq_scale': 'Nearest E2M1 under exactly paper Eq. 8; midpoint ties to even code index.',
                'efq': 'floor((local shifted maximum + ln(2/9))/ln(2))',
                'masked_or_padding': 'Masked entries get zero codes; all-masked blocks do not contribute; partial blocks padded with masked entries.'},
            'metrics': {'unit': 'document x length x layer x query head; Frobenius norm across 16 sampled query outputs, all valid keys per query',
                'relative_error': 'norm(O-Oref)_F / norm(Oref)_F; no epsilon denominator',
                'reference_norm_floor_rms': 1e-6, 'near_zero': 'Relative error null when reference RMS <= 1e-6; retain absolute Frobenius and count.',
                'js': 'Natural logs, FP64-renormalized historical-rescaled effective probabilities; stable log1p/even-series calculation. Attention arithmetic stays FP32.',
                'top_k': 8, 'top_k_ties': 'Stable descending probability sort, ascending key index; k=min(8, number of valid keys).',
                'tie_fraction': 'Adjacent equal effective probabilities after sorting / (valid keys - 1), includes zero ties; 0 for one valid key.',
                'zero_fraction': 'Generated E2M1 zero codes among valid keys, before historical rescaling.',
                'removed_mass': 'Dense reference probability mass at generated zero-code keys.',
                'max_jump': 'Largest finite increase after a previous finite online row maximum; excludes initialization.',
                'aggregation': 'Median/p95/max over head units, also per length/layer/head. Repeated prefixes/rows are not independent documents.',
                'bootstrap': {'resamples': 2000, 'seed': 330011, 'unit': '16 paired evaluation document IDs', 'statistic': 'Mean per-document unit error difference calibrated minus nearest_same_scale, separately per length.'}},
            'screen': {'baseline': 'nearest_efq_scale', 'median_ratio_limit': 1.10, 'p95_ratio_limit': 1.25,
                       'layer_length_median_ratio_limit': 2.0, 'baseline_relative_error_floor': 1e-5,
                       'numeric_failures_allowed': 0, 'near_zero_or_missing_policy': 'Count/report; cannot silently pass. Hold ratio when baseline is at floor.',
                       'meaning': 'Project engineering screen only; no quality preservation or acceleration claim.'},
            'errors': {'input_nonfinite': 'Reject NaN/+Inf scores; -Inf is masking.', 'fully_masked_row': 'Return zero output and valid=false; count separately.',
                       'valid_zero_denominator': 'Preserve failure/NaN, do not repair; screen fails.', 'execution_error': 'Save stage and traceback; no silent retries.'},
            'synthetic': {'dev_seed': 340011, 'eval_seed': 350011, 'lengths': [512, 2048, 4096], 'rows': 16,
                'families': ['uniform', 'gaussian_low', 'gaussian_medium', 'gaussian_high', 'single_peak', 'multiple_peaks', 'heavy_tail', 'causal', 'max_jump'],
                'replicates_per_family': 2, 'not_model_quality': True},
            'trace_validation': {'small_length': 128, 'same_backend_relative_tolerance': 0.01, 'fp32_vs_native_bf16_relative_tolerance': 0.02},
            'split_policy': 'Extract only development until numerical tests, small native trace check and calibration pass; freeze spec before first evaluation extraction.'}
    write_json(CASE / 'configs/development_plan.json', plan)
    versions = {name: md.version(name) for name in ['torch', 'transformers', 'numpy', 'huggingface_hub', 'safetensors', 'matplotlib']}
    versions['torch_runtime'] = torch.__version__
    versions['cuda_build'] = torch.version.cuda
    versions['python'] = sys.version.split()[0]
    write_json(CASE / 'provenance/environment.json', {'versions': versions, 'environment': 'CASE003 venv with system-site-packages from unchanged CASE002 environment; plotting dependencies installed only in CASE003.',
               'python_role_path': '$CASE003_ENV/bin/python', 'torch_import_source': 'CASE002/lib/python3.12/site-packages/torch',
               'modeling_source_sha256': sha((Path(__import__('transformers').__file__).parent / 'models/qwen3/modeling_qwen3.py').read_bytes())})
    pins = sorted({d.metadata['Name']: md.version(d.metadata['Name']) for d in md.distributions()}.items())
    (CASE / 'provenance/package_versions.txt').write_text(''.join(f'{n}=={v}\n' for n, v in pins))
    for name in ['config.json', 'generation_config.json', 'LICENSE']:
        (CASE / 'provenance' / ('model_' + name)).write_bytes((snapshot / name).read_bytes())
    print(json.dumps({'documents': len(records), 'dev': 8, 'eval': 16, 'minimum_full_tokens': min(r['full_token_count'] for r in records), 'plan_sha256': sha((CASE/'configs/development_plan.json').read_bytes()), 'versions': versions}))


if __name__ == '__main__':
    main()
