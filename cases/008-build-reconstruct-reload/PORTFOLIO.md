# Portfolio card

## English

**Build, Reconstruct, Reload.** Implemented a local checkpoint pipeline that turns existing Qwen weights into a standard GPTQ W4 artifact or a standalone model with one smaller reconstructed MLP. The code checks actual packed runtime execution, strict per-layer reload, native teacher reconstruction, gold scores and fixed-work cost. Measured Q safetensors are 2,651,839,568 bytes versus 8,044,982,000 native bytes. The three R recovery estimates and task-score changes are in [the results](README.md). Scope: two pinned models, short synthetic inputs, one R layer, no deployment verdict.

## 한국어

**모델을 변환·보정하고 다시 실행하기.** 기존 Qwen 가중치를 표준 GPTQ W4 저장본 또는 한 층 MLP를 축소·보정한 독립 모델로 만드는 로컬 도구를 구현했습니다. 실제 압축 실행 경로, 층별 구조 재로딩, 원래 출력 오차의 회복, 정답 점수와 같은 작업량의 비용을 검사합니다. Q의 safetensors는 2,651,839,568바이트이며 BF16 원본은 8,044,982,000바이트입니다. 세 R 구조의 회복률과 과제 점수는 [결과](README.ko.md)에서 확인합니다. 고정 모델 두 개·짧은 합성 입력·R의 한 층을 다루며 배포는 평가하지 않았습니다.

## Code tour and demonstration

1. Inspect [artifact loading](../../tools/modelpack/artifact.py): meta skeleton, per-layer override, strict safetensors assignment and tied weights.
2. Follow [fixed-set reconstruction](../../tools/modelpack/r_study.py) and [Cholesky fitting](../../tools/modelpack/numerics.py): same native input, DEV-only eta, represented BF16 weights.
3. Open [the item explorer](demo/index.html), select Q or R and a task, then compare baseline and candidate scores. Run [CPU reanalysis](REPRODUCTION.md) to check the aggregate.

Qwen/Transformers, GPTQ/LLM Compressor, compressed-tensors and vLLM/Marlin supply upstream models and algorithms; Codex assisted the project. The engineering contribution is the controlled conversion/reconstruction/reload and evidence path. CPU consistency checks are separate from GPU replication. [Attribution](NOTICE.md).

## Structured reading path / 구조화 보고서

[English report](REPORT.md) · [한국어 보고서](REPORT.ko.md): background, hypothesis provenance, numerical theory, implementation, dataset/design, results and interpretation. Q connects conversion, storage and fresh-runtime execution; R fits the existing smaller down weight without adding inference matrices. Q saves file/resident-weight bytes; R reduces local squared output error. Request-speed intervals include 1 and gold-score changes are mixed.

Q는 변환·저장·새 실행기 통합을, R은 추가 추론 행렬 없이 작은 down 가중치를 보정하는 구현을 보여줍니다. 파일·상주 가중치 감소와 국소 제곱오차 복구를 각각 확인했고, 요청 시간 구간은 1을 포함하며 정답 점수 변화는 혼재했습니다. 설정의 뜻부터 [한국어 보고서](REPORT.ko.md)에서 읽을 수 있습니다.
