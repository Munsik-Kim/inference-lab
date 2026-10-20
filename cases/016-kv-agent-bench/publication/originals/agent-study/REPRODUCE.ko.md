# 모델 없이 결과를 확인하는 방법

이 사례 폴더에서 실행합니다. CPU 검산에는 Python 표준 라이브러리만 필요합니다. 모델·Torch·Transformers를 불러오거나 서버에 접속하지 않습니다.

```bash
CASE016_CHECK_ROOT="$(mktemp -d /tmp/case016-cpu.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m kv_agent_bench.cli validate --case .
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m unittest discover -s publication/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m kv_agent_bench.cli oracle-check --case . --output "$CASE016_CHECK_ROOT/oracle"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m kv_agent_bench.cli audit --tasks data/tasks.json --results results/p1_r1 --output "$CASE016_CHECK_ROOT/replay"
python3 -B scripts/audit_results.py --output "$CASE016_CHECK_ROOT/independent"
python3 -B scripts/verify_sources.py --output "$CASE016_CHECK_ROOT/source-views"
```

`validate`는 등록 자료와 도구 메뉴를 확인합니다. `oracle-check`는 정답 행동 경로를 합성 fixture에서 실행합니다. 향후 EVAL 자료의 CPU 경로 검사도 포함되며 모델 성능 표본은 아닙니다. `audit`는 저장된 SSE에서 호출을 다시 조립하고 새 fixture에 실행합니다. 별도 `audit_results.py`는 token·호출·종료 이유·상태·두 endpoint 계산·paired 표까지 재계산한 뒤에 저장 점수와 대조합니다.

공개 원기록의 token ID를 원문으로 다시 해독하는 데는 등록된 tokenizer가 필요합니다. 이 CPU 경로만으로 모델 추론·logits·GPU 실행 자체를 독립 재현한 것으로 표시하지 않습니다. 모든 출력 디렉터리는 새 경로여야 합니다.

[한국어 결과 탐색기](demo/ko.html), [영어 탐색기](demo/en.html)는 정적 파일로 열 수 있습니다. 재생성할 때는 case를 새로운 scratch에 복사한 다음, 실제 구현된 `PYTHONPATH=src python3 -B -m kv_agent_bench.cli render --case <사본>` 명령을 사용합니다. 원 기록을 덮어쓰지 않습니다.

## GPU 재실행과 비공개 입력

실제 실행한 진입점은 다음과 같습니다. 두 변수는 기존 격리 환경의 Python 절대경로와 비공개 runtime JSON을 가리킵니다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$CASE016_GPU_PYTHON" -B -m kv_agent_bench.cli run --case . --runtime "$CASE016_PRIVATE_RUNTIME" --stage p0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$CASE016_GPU_PYTHON" -B -m kv_agent_bench.cli run --case . --runtime "$CASE016_PRIVATE_RUNTIME" --stage p1
```

runtime의 실제 필드는 `python`, `snapshot`, `private_root`, `prepared_root`, `p0_run_id`, `p1_run_id`, `pool_bytes`, `p1_max_model_len`, `p1_kvs`, `source_commit`입니다. 이번 P0의 복구 이력에는 `p0_kvs`·`p0_reuse_results`와 계측 필드도 있습니다. 새 재현은 별도 출력 경로와 자체 P0 지원·계측 검사를 사용해야 합니다. 기존 완료 cell은 덮어쓰기를 거절하며, 게이트를 수동으로 PASS로 바꾸는 방법은 재현 경로가 아닙니다.

모델은 기존 WSL의 Qwen3-4B 고정 revision을 읽기 전용으로 사용했습니다. [모델 identity](provenance/model_identity.json)는 12개 실제 로컬 파일과 모든 weight shard의 해시입니다. 별도로 확보한 공식 전체 weight digest와 비교했다는 뜻은 아닙니다. D 원본 탐색·복사는 실행 조건이 아니었습니다.

의존성은 [환경 기록](provenance/environment.json)과 [고정 목록](provenance/resolved-lock.txt)을 확인합니다. vLLM0.32.0 공식 wheel과 FlashInfer 공식 커널을 새 Python3.12 환경에 설치했습니다. compiler/CRT/NVVM13.0.88 정렬과 라이브러리 링크 경로 수정은 새 환경 내부에서만 했습니다. 시스템 driver나 기존 연구 환경은 업그레이드하지 않았고 실패 기록도 보존했습니다.

새 GPU 실행 전에는 snapshot·tokenizer·template·실제 KV dtype36층·pool·토큰 용량·입력 ID와 원생성 관측을 확인해야 합니다. BF16 weights, prefix cache OFF, 공통 eager·scheduler, EOS, parser, 문법·이름 제한 OFF와 모든 예산을 유지합니다. 본인이 시작한8152 loopback 서버만 관리합니다.

원 모델·전체 KV·가상환경·큰 profile·개인 경로는 검토 ZIP에 없습니다. P2/P3·EVAL 모델 실행·GitHub/Pages 게시도 이번 재실행 범위에 포함하지 않습니다.

최종 표시본은 case의 새 사본에서 `python3 -B scripts/render_review.py --case <새 사본>`으로 생성합니다. 동결한 최소 renderer를 수정하지 않고 DIOVA 색·설명·그림·필터를 더하는 표시 전용 wrapper입니다. `verify_sources.py`는 11개 과거 동결 집합을 별도 view로 복원해 260개 파일 해시를 대조합니다. P1 실행 파일 35개는 현재 경로에서도 원래 바이트를 유지합니다.
