"""Completion is derived from evidence, never a substitute for output validity."""
from .validity import assess


def completion_status(records, semantic, numerical_audit, timing_audit, secondary):
    problems=[]
    if semantic.get('status')!='PASS':problems.append('SEMANTIC_EVIDENCE_MISSING_OR_FAILED')
    if not records or any(not assess(r.get('validity',{}))['valid'] for r in records):
        problems.append('FULL_OUTPUT_VALIDITY_MISSING_OR_FAILED')
    expected={('dev',4096):96,('standard',4096):192,('standard',512):48,('standard',2048):48,
              ('boundary_pool',4096):46} # Exact retained EVAL_MANIFEST_FREEZE selection, not a refill target.
    for (split,length),count in expected.items():
        for arm in ('B','A_PUBLIC','V4'):
            rows=[r for r in records if r['split']==split and r['length']==length and r['arm']==arm]
            if len(rows)!=count or len({r['base_id'] for r in rows})!=count:
                problems.append(f'INCOMPLETE_{split}_{length}_{arm}')
    if numerical_audit.get('status')!='PASS_CPU_MEASUREMENT_CONSISTENCY':problems.append('SCALAR_AUDIT_NOT_PASSED')
    if timing_audit.get('status')!='PASS':problems.append('TIMING_AUDIT_NOT_PASSED')
    if len(secondary)!=24 or len({r['base_id'] for r in secondary})!=24:problems.append('SECONDARY_INCOMPLETE')
    for item in secondary:
        for arm in ('B','A_PUBLIC','V4'):
            row=item.get('arms',{}).get(arm,{})
            required=[row.get('validity',{})]
            if arm!='B':required.append(row.get('common_prefix',{}).get('validity',{}))
            if any(v.get('full_output_finite') is not True or v.get('all_decode_BF16') is not True for v in required):
                problems.append('SECONDARY_FULL_OUTPUT_OR_DECODE_INVALID')
    invalid=any('VALIDITY' in s or 'INVALID' in s for s in problems)
    return {'execution_status':'BLOCKED_NUMERICAL_VALIDITY' if invalid else 'PARTIAL_TECHNICAL_BLOCK' if problems else 'COMPLETED_CONTROLLED_STUDY',
            'semantic_validity_status':'PASS' if semantic.get('status')=='PASS' and not invalid else 'BLOCKED',
            'evidence_completeness':'INCOMPLETE' if problems else 'ALL_FIXED_STAGES_COMPLETE',
            'deployment_verdict':'NOT_ASSESSED','problems':sorted(set(problems)),
            'publication_readiness':'PENDING_PACKAGE_AND_HYGIENE_CHECKS'}
