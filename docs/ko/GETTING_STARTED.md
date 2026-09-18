# 읽기·탐색·재계산·재현 안내

[English](../en/GETTING_STARTED.md) | 한국어 · [홈](../../README.ko.md)

개별 실험을 모은 저장소이며 하나의 설치형 서비스나 공통 모델 서버 CLI는 아닙니다. 필요한 확인 수준을 선택하면 됩니다. 읽기와 저장된 결과 탐색에는 GPU가 필요하지 않습니다.

## 1. 근거 읽기

[사례 안내](CASEBOOK.md)를 읽고 각 사례의 결과·방법 링크로 이동합니다. [포트폴리오 안내](PORTFOLIO.md)는 구현 역량을 실제 소스와 연결합니다. 기존 [문서 추출 결과 해설 — 영어](../../notes/case002-results-and-discussion.md)도 짧은 사례별 읽을거리로 유지합니다. 기존 사례 문서는 기술 기록이며, 이 안내가 protocol이나 측정 자료를 대체하지 않습니다.

<a id="offline-explorers"></a>
## 2. 저장된 결과 탐색기 열기

<!-- claims: c006-package c006-browser -->
[현행 Case006 통합 ZIP](../../downloads/case006_decision_stability_with_readout_20260917_docfix1.zip)과 [크기·해시 JSON](../../downloads/case006_decision_stability_with_readout_20260917_docfix1.json)을 사용합니다. GitHub 파일 화면에서 다운로드 기능을 선택합니다. HTML 소스 미리보기를 실행 중인 데모로 해석하지 않습니다. `docfix1`이 없는 이전 통합 ZIP은 역사적 버전으로 보존돼 있으며, 현재판은 과학적 근거를 바꾸지 않고 재현 경로 안내를 수정했습니다. Cases001·002·004의 기존 ZIP은 각 사례 페이지에 연결돼 있습니다. 여기에는 별도 Case003·Case005 ZIP 링크를 만들지 않았습니다.

로컬 디렉터리에 압축을 풀고 다음 파일 중 하나를 브라우저에서 엽니다.

```text
006-attention-decision-stability/demo/index.html
006-attention-decision-stability/supplemental/readout-ties-v1/demo/index.html
```

Readout은 모델 내부 상태를 어휘 점수로 바꾸는 마지막 출력 계산입니다. 첫 탐색기는 원래 출력 계산(native)을 보여주며, 두 번째는 같은 입력의 별도 진단용 출력 계산(shadow)과 비교하는 사후 진단입니다. **기존 탐색기 UI는 영어**입니다. 둘 다 저장된 데이터를 포함하며 모델을 실행하지 않습니다. GitHub의 HTML 소스 화면은 실행형 데모가 아니고 Pages 서버도 필요하지 않습니다.

B는 BF16 기준선이며, A_PUBLIC과 V4는 가중치가 BF16인 같은 모델 안의 두 저정밀 attention 설정입니다. Gold는 별도로 계산한 정답입니다.

**짧게 확인하기:** STANDARD에서 과제와 항목 ID를 고르고, gold와 B/A_PUBLIC/V4 점수 및 결과 유형을 비교합니다. 보충 탐색기에서는 native와 shadow를 비교하되 독립 표본으로 세지 않습니다. 스트레스셋으로 옮길 때에는 별도 선정 규칙부터 확인합니다. 비어 있는 유형은 관측 사례가 없다는 뜻이지 앞으로도 위험이 0이라는 뜻은 아닙니다.

보충 탐색기는 Windows Edge의 `file://`에서 데스크톱과 태블릿 크기 에뮬레이션 검사를 기록했습니다. 실제 iPad 시험은 아니므로 **real iPad NOT_TESTED**입니다. JavaScript를 막는 파일 미리보기에서는 조작 기능이 작동하지 않을 수 있습니다. Markdown·그림·JSON은 별도로 볼 수 있습니다. [실제 브라우저 검사 기록 JSON](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/provenance/browser_validation.json)을 확인하고 모든 기기에서의 동작 보증으로 확대하지 않습니다.

<a id="case007-explorer"></a>
### Case007: 압축과 절충 관계 확인하기

<!-- claims: c007-package -->
[검토된 Case007 ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.zip)을 받고 [크기·해시 metadata](../../downloads/case007_mlp_pruning_reviewed_publication_v1.json)를 확인합니다. 압축을 풀고 `007-interaction-aware-mlp-pruning/demo/index.html`을 로컬에서 엽니다. 기존 UI는 영어이며 저장된 측정만 사용합니다. 방법과 삭제 예산을 선택하고 입력 ID와 제거 그룹을 살펴본 뒤, 국소 재구성 오차와 정답 전환을 구분합니다. 첫 화면의 **COMPLETED_NO_CLEAR_TRANSFER**는 원래 동결 판정으로 배포 승인이 아닙니다. [결과 해석](CASEBOOK.md#case-007).

원래 manifest와 현재 공개본 manifest는 검사 범위가 다릅니다. 기존 Case007 패키지 검사기는 Python 표준 라이브러리를 사용합니다. 저장소 루트에서 다음 명령을 실행하고 새 외부 출력 파일을 지정합니다. ZIP을 풀었다면 case 디렉터리에서 `cd`를 제외하고 실행합니다.

```bash
cd cases/007-interaction-aware-mlp-pruning
python -B scripts/verify_publication.py --case . --output /tmp/inference-lab-case007-docs-check-new.json
```

이는 원본 보존과 공개 스칼라·표의 일관성 검사이며 비공개 벡터나 GPU 실행 검사가 아닙니다. 전체 CPU 재분석과 별도 GPU 절차는 [Case007 재현 안내 — 기술 원문(영어)](../../cases/007-interaction-aware-mlp-pruning/REPRODUCTION.md)를 따릅니다. case 전용 ZIP은 저장소 전체 백업이 아니므로 이전 사례 링크에는 저장소가 필요합니다. 이 안내에서 과거 브라우저 검사를 재실행하지 않았고 실제 iPad는 **NOT_TESTED**입니다.

## 3. CPU로 Case006 패키지를 확인한 뒤 수치를 재계산하기

<!-- claims: c006-verification -->
저장소 checkout 루트에서 사용 가능한 CPU Python 환경으로 시작합니다. 아래 최소 패키지 검사는 Python 표준 라이브러리를 사용하며 GPU·모델·새 설치가 필요하지 않습니다. 출력은 **case 밖의 새 파일**이어야 하며 이미 있으면 다른 이름을 선택합니다.

```bash
cd cases/006-attention-decision-stability
python -B scripts/verify_publication.py --output /tmp/inference-lab-case006-docs-check-new.json
```

ZIP만 풀었다면 먼저 그 안의 `006-attention-decision-stability` 디렉터리로 들어간 뒤 같은 verifier 명령을 실행합니다. 이 검사는 파일 식별과 목록을 확인하며 과학적 수치 계산 자체를 검사하는 것은 아닙니다.

점수와 구간을 다시 계산하려면 [CPU 재분석 명령 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#cpu-analysis-of-retained-measurements)을 따릅니다. 기록된 분석 환경은 Python 3.12.14와 NumPy 2.3.5이며, 그림 생성에는 Matplotlib 3.10.8과 Pillow 12.3.0도 사용했습니다. 이 안내는 설치 명령이나 의존성 변경을 추가하지 않습니다. 이번 문서 개편에서는 대형 재계산과 GPU 예시를 실행하지 않았고 기존 CPU 환경으로 최소 패키지 verifier만 실행했습니다.

`SHA256SUMS`는 원래 검토 snapshot을, `PUBLICATION_SHA256SUMS`는 자기 자신을 제외한 현재 통합 공개 파일을 설명합니다. 보충 자료의 고정 manifest도 별도로 유지합니다. 역사적 59개 exact-inventory 테스트는 원실험 전용 archive의 목록을 전제로 하므로 통합 트리에 그대로 적용하거나 새 파일에 맞추어 완화하지 않습니다. [검증 범위 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#publication-package-verification).

## 4. GPU 재현은 별도 실험으로 다루기

커널 출력을 다시 얻으려면 고정 모델·소스·binary와 호환 하드웨어 환경이 필요합니다. 공개 스칼라 기록으로 제외된 hidden vector나 전체 Q/K/V를 복원할 수는 없습니다. [고정 로컬 GPU 안내 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#pinned-local-gpu-replay)와 [원실험·readout 경로 연결 — 영어](../../cases/006-attention-decision-stability/REPRODUCTION.md#connecting-original-replay-to-the-readout-supplement)가 조건을 설명합니다. 이 안내 작성 중 실행한 절차가 아닌 향후 재현 지침입니다. CPU 재계산은 저장된 근거의 계산 일관성을 확인하며, 외부 GPU 재현이나 배포 적합성을 입증하지 않습니다.
