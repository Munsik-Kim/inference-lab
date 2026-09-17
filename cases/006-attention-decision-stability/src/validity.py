"""Evidence-derived gates; missing checks can never imply validity."""
from __future__ import annotations

REQUIRED = ("full_attention_outputs_finite", "full_block_outputs_finite",
            "full_final_hidden_finite", "full_logits_finite", "inputs_unchanged",
            "same_layer13_qkv", "routing_valid", "dtype_layout_valid", "answer_interface_valid",
            "native_B_validated", "effective_backend_verified")


def assess(checks: dict) -> dict:
    missing = [name for name in REQUIRED if name not in checks or checks[name] is None]
    failed = [name for name in REQUIRED if name in checks and checks[name] is not None and checks[name] is not True]
    status = "BLOCKED_NUMERICAL_VALIDITY" if any("finite" in x for x in failed) else "BLOCKED_SEMANTICS" if failed else "INCOMPLETE_VALIDITY" if missing else "VALID"
    return {"status": status, "valid": not missing and not failed, "missing": missing, "failed": failed}


def study_status(records: list[dict], expected_cells: int, semantic: dict | None) -> dict:
    if not records:
        return {"execution_status": "CPU_ONLY_HARNESS_READY", "semantic_validity_status": "NOT_RUN",
                "evidence_completeness": "MEASUREMENT_NOT_RUN", "observable_effect_summary": "NOT_ESTIMABLE",
                "deployment_verdict": "NOT_ASSESSED", "publication_readiness": "HARNESS_ONLY_REQUIRES_GPU_VALIDATION"}
    statuses = [assess(r.get("validity", {})) for r in records]
    good = all(s["valid"] for s in statuses) and semantic is not None and semantic.get("status") == "PASS"
    complete = len(records) == expected_cells
    decision = "COMPLETED_CONTROLLED_STUDY" if good and complete else "PARTIAL_TECHNICAL_BLOCK"
    if any(s["status"] == "BLOCKED_NUMERICAL_VALIDITY" for s in statuses):
        decision = "BLOCKED_NUMERICAL_VALIDITY"
    return {"execution_status": decision, "semantic_validity_status": "VALID" if good else "BLOCKED_OR_INCOMPLETE",
            "evidence_completeness": "COMPLETE" if complete else "PARTIAL",
            "observable_effect_summary": "ESTIMATE_FROM_RETAINED_PAIRS" if good else "NO_SUCCESSFUL_VERDICT",
            "deployment_verdict": "NOT_ASSESSED", "publication_readiness": "REQUIRES_ARTIFACT_AUDIT"}
