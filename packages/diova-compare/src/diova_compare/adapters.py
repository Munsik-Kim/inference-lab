"""Adapters keep the official harness metric and its recorded prompt provenance."""
import hashlib
import json
import math
from .core import require


def harness_sample(sample, *, task, version, model_id, artifact_id, metric,
                   evidence_kind='measurement'):
    """Convert a logged lm-eval sample. acc_norm needs explicit option lengths.

    This adapter never replaces official result aggregation. It exposes pairs for
    one explicitly selected metric; MMLU subjects remain separate task names.
    """
    args = sample['arguments']
    prompt_hash = hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    base = dict(schema_version=1, sample_id=str(sample['doc_id']), task=task,
                task_version=version, prompt_hash=prompt_hash, model_id=model_id,
                artifact_id=artifact_id, gold_definition='lm-eval:' + metric,
                evidence_kind=evidence_kind)
    if metric == 'acc':
        responses = sample['filtered_resps']
        scores = [float(x[0]) for x in responses]
        require(all(math.isfinite(x) for x in scores), 'nonfinite harness score')
        gold = str(int(sample['target']))
        prediction = str(max(range(len(scores)), key=scores.__getitem__))
        require(int(sample[metric]) == int(prediction == gold), 'harness prediction/metric mismatch')
        return dict(base, output_type='choice', answer=prediction, correct=bool(sample[metric]), gold=[gold])
    if metric.startswith('exact_match'):
        answer = sample['filtered_resps'][0]
        require(isinstance(answer, str), 'missing extracted final answer')
        require(sample[metric] in (0, 1, 0.0, 1.0), 'invalid exact match')
        # Official extraction/normalization owns correctness; this is not a new parser.
        return dict(base, output_type='extracted_answer', answer=answer or '[EMPTY_EXTRACTION]',
                    correct=bool(sample[metric]), gold=[str(sample['target'])])
    raise ValueError('Unsupported adapter metric; use official aggregate for acc_norm/PPL')
