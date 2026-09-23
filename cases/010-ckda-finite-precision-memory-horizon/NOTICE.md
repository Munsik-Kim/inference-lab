# Attribution, licenses and evidence / 기여·라이선스·근거

[English introduction](README.md) · [한국어 소개](README.ko.md)

This integration combines two preserved DIOVA research stages. Project-authored code follows the repository's [Apache-2.0 license](../../LICENSE). The full original notices are retained in [v1](versions/v1/NOTICE.md) and [v2](versions/v2/NOTICE.md). Their old relative repository links are interpreted in the historical workspace restored by [restore_workspace.py](scripts/restore_workspace.py); the original text is not edited to change scientific identity.

통합본은 두 DIOVA 연구 단계를 보존해 연결합니다. 프로젝트 작성 코드는 저장소 Apache-2.0 라이선스를 따르고 두 버전의 원래 고지를 유지합니다. 과거 고지의 저장소 상대경로는 복원한 원래 workspace에서 해석합니다.

## Implementation and upstream roles

Complex KDA supplies the group task, model, optimizers, native recurrence and original readout. The upstream checkout is pinned to [OpenEuroLLM/ComplexKDA ef9d108d…](https://github.com/OpenEuroLLM/ComplexKDA/tree/ef9d108d1692387cae37f5b2d539a71826a127c1) and retains its [MIT license](https://github.com/OpenEuroLLM/ComplexKDA/blob/ef9d108d1692387cae37f5b2d539a71826a127c1/LICENSE). The checkout and trained checkpoint weights are not distributed here.

DIOVA implements packed-state adapters, online residual handling, failure-aware serialization, checkpoint/restart contracts, fixed-budget comparisons, scalar audits and presentation tools. Codex assisted source inspection, implementation, testing and documentation. An independent arithmetic path in this repository is not external independent reproduction. Residual transport, corrected-state feedback and byte packing are not claimed as new algorithms or theorems.

Complex KDA의 모델·과제·원래연산과 DIOVA의 저장·재시작·비교·검산 구현을 구분합니다. Codex 지원은 구현·검사·문서화를 포함합니다. 새 프로세스검사나 별도 계산경로가 외부 독립 연구 재현을 뜻하지는 않습니다.

## Bibliography inherited from the original notices

The following titles and roles are carried from the preserved v1 notice; v2 and this integration do not claim a new paper search or numerical reproduction of these works.

| Reference | Relationship to this project |
| --- | --- |
| Julien Siems et al., *Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention* (2026), [arXiv2609.24797v1](https://arxiv.org/html/2609.24797v1) | Chosen model, signed gates, extended beta and finite-group task |
| Tao Zhang et al., *DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization* (2026), [arXiv2608.27513v1](https://arxiv.org/html/2608.27513v1) | Related error/persistence-based state precision; the CAL mixed comparator here is generic, not a DAMP reproduction |
| Jiwan Chung, Heechan Choi and Seon Joo Kim, *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics* (2026), [arXiv2605.07755v1](https://arxiv.org/html/2605.07755v1) | Related state separation/readability analysis; no transferred guarantee for the present MLP |
| Ismail Erbas, Xavier Intes and Vikas Pandey, *When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference* (2026), [arXiv2609.04490v1](https://arxiv.org/html/2609.04490v1) | Related write-back/residual-memory baselines with different recurrence, clipping and residual-grid conventions |

문헌의 제목·역할은 원래 v1 고지에서 이어받았습니다. 이번 통합에서 읽지 않은 정리·수치 결과를 새 근거로 추가하지 않습니다. 공식 task, 자체 CAL 기준선, affine 잔차항등식, 학습 MLP 판독의 범위는 [통합보고서](REPORT.ko.md)에 설명합니다.

## Review evidence availability

The retained v2 receipt states that the exact requested reviewer files and probe were not found in its accessible search scope. The integration likewise does not claim execution of unavailable `CASE010_REVIEW_KO.md`, `CASE010_V2_REVIEW_KO.md` or independent-review scripts. The request-recreated synthetic probe and actual historical model-failure receipts are labelled separately. [Original availability receipt](versions/v2/results/historical-reanalysis/review_artifact_availability.json).

확보하지 못한 독립 검토 문서·script의 검사 결과는 주장하지 않습니다. 요청 설명으로 재구성한 합성 probe, 원래 실제 모델 실패 receipt, 이번 통합의 CPU 재검산은 별도 근거입니다.
