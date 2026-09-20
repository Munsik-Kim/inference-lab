# 읽기·탐색·재계산·재현 안내

[English](../en/GETTING_STARTED.md) | 한국어 · [홈](../../README.ko.md)

[처음 보는 분](START_HERE.md) · [쉬운 용어집](GLOSSARY.md)

사례를 읽고, 기록된 입력을 탐색하거나, 작은 CPU 검사를 실행할 수 있습니다. GPU 재현에는 별도 환경과 명령이 있습니다.


<a id="case008-paths"></a>
## Case008: 목적에 맞는 실행 경로

| 목적 | 시작할 곳 |
|---|---|
| 결과를 보고 기록된 점수를 재계산하기 | [Q/R 결과 화면](https://munsik-kim.github.io/inference-lab/ko/case008.html)을 연 뒤 아래 CPU 근거 검사 실행. |
| 사전학습 모델 없이 구현을 실행해 보기 | [작은 CPU 저장·재로딩 예제](#tiny-model-demo). |
| 큰 Q 또는 R 모델 저장본 만들기 | [고정 환경과 전체 제작 절차·영어](../../cases/008-build-reconstruct-reload/REPRODUCTION.md). 모델 가중치는 별도로 준비하는 로컬 입력입니다. |
| 원래 실험을 같은 조건으로 재현하기 | [Protocol·영어](../../cases/008-build-reconstruct-reload/configs/protocol_draft.json), 고정 입력과 [역사적 snapshot 복원·영어](../../cases/008-build-reconstruct-reload/publication/README.md). |

저장소 루트에서 Python 3.12.14·NumPy 2.3.5가 있는 CPU 환경과 **새 외부 출력 디렉터리**를 사용합니다.

```bash
OUT=$(mktemp -d)
python3 -B tools/case008_checks/run.py --mode evidence --output "$OUT/evidence"
```

공개본 테스트, 원래 분석, 스칼라·시간 감사, 같은 기록의 NLL·복구율 진단을 실행합니다. 재생성한 summary, 추가·국소 결과표, 사후 JSON을 보존된 파일과 바이트 단위로 비교합니다. 모델 추론이나 제외된 전체 벡터를 다시 계산하지 않습니다. `Case 008 CPU contracts` workflow는 이 검사와 modelpack 테스트를 별도 job으로 실행합니다. Portfolio CPU checks는 문서·사이트 검사로 유지합니다.

<a id="tiny-model-demo"></a>
## 작은 모델의 저장·재로딩 기능 실행하기

별도 CPU 환경에서 [예제 의존성 버전](../../tools/modelpack_demo/requirements-cpu.txt)을 사용합니다. Python 3.12.14, PyTorch 2.13.0, Transformers 5.17.0, safetensors 0.8.0, NumPy 2.3.5로 확인했습니다. CI는 CPU 전용 PyTorch wheel을 설치하며, 기존의 호환 환경도 사용할 수 있습니다. 사전학습 가중치·tokenizer·GPU는 필요 없습니다. 환경 준비는 [예제 README·영어](../../tools/modelpack_demo/README.md)에 있습니다.

```bash
OUT=$(mktemp -d)
python3 -B tools/modelpack_demo/roundtrip.py --output "$OUT/tiny-roundtrip"
python3 -B -m unittest discover -s tests/modelpack_demo -v
```

CPU에서 무작위 초기화한 2층 Qwen 모델을 만들고, 1번 층의 중간 폭을 64에서 48로 줄여 저장한 뒤 제작 프로세스를 종료합니다. 다른 디렉터리의 복사본을 두 번째 프로세스에서 읽으며, 원래 제작 디렉터리는 접근을 차단합니다. 전체 출력의 유효성과 일치, 행렬 shape, embedding과 head의 가중치 공유를 확인합니다. 고정된 Case008 로더를 재사용하며 연구 코드를 바꾸지 않습니다.

검사 결과에는 `TINY_RANDOM_CPU_FIXTURE`, 폭 `[64,48]`, gate/up `[48,32]`, down `[32,48]`, 같은 logits와 공유 가중치가 남습니다. 작은 모델 파일과 프로세스 로그는 외부 출력 디렉터리에 저장됩니다. 직렬화 기능을 검사하는 예제이며 학습된 모델의 품질·속도 결과는 아닙니다. 원래 modelpack 테스트까지 함께 실행하려면 다음을 사용합니다.

```bash
OUT=$(mktemp -d)
python3 -B tools/case008_checks/run.py --mode modelpack --output "$OUT/modelpack"
```

Wrapper는 해시를 확인한 역사적 트리를 복원한 뒤 exact-inventory 테스트를 실행합니다. 누락·잘못된 artifact 거절도 검사하며, 현재 공개 파일 목록에 맞춰 동결 테스트를 완화하지 않습니다.


## 설치 없이 공개 결과 화면 열기

[한국어 홈](https://munsik-kim.github.io/inference-lab/ko/index.html) · [Case007: 구조화 압축](https://munsik-kim.github.io/inference-lab/ko/case007.html) · [Case006: 답변 선택과 동률](https://munsik-kim.github.io/inference-lab/ko/case006.html). Pages 화면은 저장된 측정값을 보여주며 설치나 GPU가 필요 없습니다. 오프라인에서 보려면 아래처럼 로컬 빌드를 사용합니다.

<a id="local-showcase"></a>
## 한·영 결과 화면을 로컬에서도 열기

새 화면에서 Case007의 그룹 선택과 Case006의 답변 점수를 볼 수 있습니다. 저장된 측정값을 읽으며, 원래 출력과 사후 출력 진단을 따로 보여줍니다. 이 저장소의 루트에서 Python 3.12로 실행하고 새 외부 출력 디렉터리를 사용합니다.

```bash
OUT=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$OUT/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/site-check.json"
```

브라우저에서 `$OUT/site/index.html`을 열어 한국어 또는 English를 고릅니다. 빌드는 Python 표준 라이브러리를 사용합니다. 스크립트와 표시 데이터가 로컬에 있어 서버 없이 `file://`로 열 수 있습니다. 같은 상대 리소스 경로는 `/inference-lab/` 하위 경로에서도 동작합니다. 공개된 Pages 화면과 로컬 빌드는 같은 정적 화면 코드를 사용합니다.

상세 화면은 **목표 → 데이터셋 → 가정과 이론 → 실험 설계 → 검증 내용 → 결과 → 결과 해석 → 결론** 순서로 읽습니다. 목차에서 필요한 항목으로 이동하거나 **입력별 결과 바로 보기**를 선택할 수 있습니다. 위의 전체 실험 요약과 아래 필터의 현재 부분집합 통계는 별도입니다. [GitHub 사례 안내](CASEBOOK.md)도 여덟 사례를 같은 순서로 설명합니다.

Case007에서 25% 또는 50% 삭제를 고른 뒤 과제나 입력 ID를 선택합니다. 표는 같은 예산에서 B와 두 선택법을 비교합니다. Case006에서는 표준 또는 선정 스트레스, 원래 계산 또는 표시된 FP32 진단을 선택합니다. 후보는 항상 같은 출력 계산의 B와 대응합니다. 필터와 입력 ID는 URL에 남아 새로고침·언어 전환 후에도 유지됩니다. JSON 다운로드에는 원래 수치 정밀도의 비교값을 넣습니다. 안내 사례는 과제별 첫 고정 ID이며 성공 사례만 고른 것이 아닙니다.

화면에서는 결정론적인 무관한 채움 문장과 토큰 배열을 생략하고 정확한 공개 입력·점수 기록으로 연결합니다. Case007 시간은 별도 입력 6개의 집계이며 화면의 각 입력별 측정값이 아닙니다. 네 선택지 점수로 선택지 지표를 재계산할 수 있습니다. 저장된 전체 어휘 KL과 전체 출력 유효성은 원래 측정에서 더 넓은 자료를 확인한 기록입니다.

기존 Windows Edge를 CDP로 제어해 390·768·1280픽셀 화면에서 `file://`, 필터, 빈 결과, URL 복원, 언어 전환, JSON 다운로드를 실행했습니다. 화면 크기 에뮬레이션이며 **실제 iPad NOT_TESTED**입니다. 아래 기존 탐색기의 브라우저 기록은 과거 검사로 유지합니다. [선택적 브라우저 검사 코드](../../tools/showcase/browser_check.cjs) · [빌드·표시 데이터 검사](../../tools/showcase/check.py).

<a id="cpu-selector"></a>
## GPU 없이 Case007 선택기 다시 실행하기

Python 3.12.14·NumPy 2.3.5가 있는 기존 CPU 환경을 사용합니다. Torch, Transformers, 모델 파일은 필요 없습니다. 아래 예제를 해당 환경에서 실행해 확인했습니다. 저장소 루트에서 `CPU_PY`를 그 인터프리터로 지정하고 새 출력 파일을 사용합니다.

```bash
CPU_PY=python3
OUT=$(mktemp -d)
"$CPU_PY" -B tools/showcase/replay_selection.py --output "$OUT/selection.json"
"$CPU_PY" -B -m unittest discover -s tests/showcase -v
```

Wrapper는 원래 선택기를 고유한 모듈 이름으로 불러와 보정 행렬 Q에서 실행합니다. Source hash, 유한하고 대칭인 16×16 Q, 연속 그룹 대응, 고정 예산, 원래 사전식 동점 규칙을 검사합니다. 선택에 미관측 입력의 통계를 사용하지 않습니다.

| 삭제 예산 | 다시 계산한 그룹과 저장 결과 일치 여부 |
|---|---|
| 4/16 | INDEPENDENT `[8,12,13,14]`, PAIRWISE `[8,13,14,15]` — 모두 일치 |
| 8/16 | 두 방법 모두 `[7,8,9,11,12,13,14,15]` — 모두 일치 |

방법별로 1,820개 또는 12,870개의 작은 CPU 목적값을 계산합니다. **모델 forward는 0회**입니다. JSON에는 목적값과 정의를 담습니다. 독립 선택은 선택된 대각 원소 합, 상호작용 선택은 선택된 부분행렬 전체 합을 사용하므로 목적함수가 다릅니다. 이 예제는 저장된 선택 로직을 확인하며 GPU 품질·시간은 원실험의 결과입니다.


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

<a id="cpu-checks"></a>
## 3. CPU로 Case006 패키지를 확인한 뒤 수치를 재계산하기

<!-- claims: c006-verification -->
저장소 checkout 루트에서 사용 가능한 CPU Python 환경으로 시작합니다. 아래 최소 패키지 검사는 Python 표준 라이브러리를 사용하며 GPU·모델·새 설치가 필요하지 않습니다. 출력은 **case 밖의 새 파일**이어야 하며 이미 있으면 다른 이름을 선택합니다.

```bash
cd cases/006-attention-decision-stability
python -B scripts/verify_publication.py --output /tmp/inference-lab-case006-docs-check-new.json
```

ZIP만 풀었다면 먼저 그 안의 `006-attention-decision-stability` 디렉터리로 들어간 뒤 같은 verifier 명령을 실행합니다. 이 검사는 파일 식별과 목록을 확인하며 과학적 수치 계산 자체를 검사하는 것은 아닙니다.

점수와 구간을 다시 계산하려면 [CPU 재분석 명령 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#cpu-analysis-of-retained-measurements)을 따릅니다. 기록된 분석 환경은 Python 3.12.14와 NumPy 2.3.5이며, 그림 생성에는 Matplotlib 3.10.8과 Pillow 12.3.0도 사용했습니다. 이 안내는 설치 명령이나 의존성 변경을 추가하지 않습니다. 화면 검사는 선택기 재실행, 패키지 검증과 일부 스칼라 감사를 포함합니다. 전체 bootstrap·그림 재생성과 GPU 예시는 별도 절차입니다.

`SHA256SUMS`는 원래 검토 snapshot을, `PUBLICATION_SHA256SUMS`는 자기 자신을 제외한 현재 통합 공개 파일을 설명합니다. 보충 자료의 고정 manifest도 별도로 유지합니다. 역사적 59개 exact-inventory 테스트는 원실험 전용 archive의 목록을 전제로 하므로 통합 트리에 그대로 적용하거나 새 파일에 맞추어 완화하지 않습니다. [검증 범위 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#publication-package-verification).

## 4. GPU 재현은 별도 실험으로 다루기

커널 출력을 다시 얻으려면 고정 모델·소스·binary와 호환 하드웨어 환경이 필요합니다. 공개 스칼라 기록으로 제외된 hidden vector나 전체 Q/K/V를 복원할 수는 없습니다. [고정 로컬 GPU 안내 — 기술 원문(영어)](../../cases/006-attention-decision-stability/REPRODUCTION.md#pinned-local-gpu-replay)와 [원실험·readout 경로 연결 — 영어](../../cases/006-attention-decision-stability/REPRODUCTION.md#connecting-original-replay-to-the-readout-supplement)가 조건을 설명합니다. 이 안내 작성 중 실행한 절차가 아닌 향후 재현 지침입니다. CPU 재계산은 저장된 근거의 계산 일관성을 확인하며, 외부 GPU 재현이나 배포 적합성을 입증하지 않습니다.

과거 입문 안내의 검사는 문서·링크·파일 보존에 한정됐습니다. Case008의 현재 CPU 명령은 별도로 실행해 확인했으며, GPU 재현과 큰 모델 제작 명령은 이번에 실행하지 않았습니다.

## Case 008: 보고서와 독립 검산 묶음

[구조화 보고서](../../cases/008-build-reconstruct-reload/REPORT.ko.md)를 읽고 [코드·recipe·측정 자료 ZIP](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip)과 [메타데이터](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.json)를 내려받으세요. 압축 해제 루트에는 tools/modelpack과 사례 폴더가 함께 있습니다. `cases/008-build-reconstruct-reload/demo/index.html`을 열면 원래 영어 탐색기를 볼 수 있습니다. 한·영 사이트 요약은 Q의 파일·메모리와 R의 보정 전후를 보여주고, 원래 탐색기는 더 자세한 기록으로 이어집니다. [CPU 검사·원본 복원 명령(영어)](../../cases/008-build-reconstruct-reload/REPRODUCTION.md)은 향후 GPU 명령과 분리되어 있으며 모델 가중치는 포함하지 않습니다.
