# Recurrent state storage and restart / 반복 상태 저장과 재시작

[English home](README.md) · [한국어 홈](README.ko.md) · [EN report](REPORT.md) · [KO 보고서](REPORT.ko.md)

## English

Built a low-bit recurrent-state codec with transported residuals, byte-budget accounting and first-error survival analysis. The failure-aware adapter serializes cursor, numerical terminal reason and the first failed write. It retains the last committed state/RNG body and resumes the same suffix in a fresh process.

The small S3 CKDA study connects implementation and measured behavior. INT8 reduces serialized state per stream from 12,305 to 3,137 bytes (74.5%), with less than one token difference from native in mean consecutive-correct-length point estimates for each of three checkpoints. At the Rank2 N=128 storage cap, comparison with Mixed5/6 gives checkpoint-dependent effects; the full means, intervals, horizons and CPU costs are linked in the [report](REPORT.md#fresh).

A short code tour:

1. [PackedCodec](versions/v2/source/v1_reference/codec/packed.py): real bit streams, scales, padding and stochastic RNG.
2. [OnlineAdapter](versions/v2/source/online_v2.py): row-local commit, terminal persistence and body/RNG rollback; [fresh-process tests](versions/v2/tests/test_online_v2.py).
3. [Evaluation checkpoint](versions/v2/source/checkpoint.py): keep runtime bytes separate from gold-dependent history; [independent auditor](versions/v2/analysis/audit_v2.py) reconstructs first error from recorded predictions.

To demonstrate it, open the [five-setting table](REPORT.md#fresh), compare native with INT8 storage, then compare Rank2 with Mixed5/6 at the same N=128 cap. Open the adapter's terminal contract and run the [model-free CPU checks](REPRODUCTION.md). The recorded learned results and synthetic restart example have distinct roles.

Complex KDA provides the task, model and original readout. DIOVA implements the storage, integration, diagnostics and verification tools; Codex assisted implementation and writing. [Attribution](NOTICE.md).

## 한국어

저비트 recurrent-state codec에 잔차 전달, 실제 바이트 장부, 최초 판독 실패 분석을 연결했습니다. Failure-aware adapter는 cursor·수치 terminal 원인·최초 실패 write를 직렬화합니다. 마지막 정상 상태·RNG body를 보존하고 새 프로세스에서 같은 나머지 입력을 이어 실행합니다.

작은 S3 CKDA 실험에서 구현과 관측을 연결했습니다. INT8은 stream별 직렬화 상태를 12,305 → 3,137 bytes(74.5%) 줄였고, 세 checkpoint 각각 평균 연속 정답 길이의 native 대비 점추정 차이는 1 token 미만이었습니다. Rank2의 N=128 예산 안에서 Mixed5/6과 비교한 효과는 checkpoint별로 달랐습니다. 평균·구간·horizon·CPU 비용은 [전체 결과](REPORT.ko.md#fresh)에서 확인할 수 있습니다.

코드는 세 곳을 순서대로 읽을 수 있습니다.

1. [PackedCodec](versions/v2/source/v1_reference/codec/packed.py): 실제 bitstream·scale·padding·stochastic RNG.
2. [OnlineAdapter](versions/v2/source/online_v2.py): 행별 commit, terminal 보존, body·RNGrollback; [새 프로세스 테스트](versions/v2/tests/test_online_v2.py).
3. [평가 checkpoint](versions/v2/source/checkpoint.py): 실행 bytes와 정답 의존 이력 분리; [독립 검산기](versions/v2/analysis/audit_v2.py)는 저장 prediction에서 최초 실패를 재계산합니다.

시연할 때 [다섯 설정 표](REPORT.ko.md#fresh)를 열고 native와 INT8의 저장량을 먼저 비교합니다. 이어 같은 N=128 cap의 Rank2와 Mixed5/6을 비교하고 adapter의 terminal 계약과 [모델 없는 CPU 검사](REPRODUCTION.ko.md)로 이동합니다. 저장된 학습 모델 결과와 합성 재시작 예제는 역할이 다릅니다.

Complex KDA는 과제·모델·원래 판독기를 제공합니다. DIOVA는 저장·통합·진단·검증 도구를 구현했으며 Codex가 구현·문서화를 지원했습니다. [출처 고지](NOTICE.md).
