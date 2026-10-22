# Case018 — 재고 복구와 완료 보고를 따로 검사했습니다 — CPU 기록 재생

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Python 3.12+ 표준 라이브러리만 사용합니다. 존재하지 않는 새 외부 output 경로를 지정하세요. 복원용 disk 1 GiB와 RAM 약 1 GiB 여유를 권장합니다. 모델·tokenizer 다운로드나 Torch/vLLM/xgrammar 설치가 필요하지 않습니다.

1. manifest·XZ 전체 경로·regular file·중복·bytes/hash·alias coverage를 검사하고 복원합니다.
2. 각 phase의 frozen source로 120개 모델 endpoint와 별도 P0 합성 trace 26개를 재생합니다.
3. 상태·관측·후속 절차·종료·예산·SSE/IDs/usage·비용·paired bootstrap을 다시 계산합니다.
4. 기존 120행 사후 진단·선택 원문·SVG를 재생성해 공개 요약과 대조합니다.
5. `--tests`는 publication 경계 검사 17개를 실행합니다. 과거 37/48/59개 이력은 원 receipt로 보존합니다.

로컬 공개 사본 검사와 GitHub-hosted Ubuntu의 CPU 기록 재생이 통과했습니다. 아래 링크에서 현재 main의 run과 receipt를 확인할 수 있습니다. 이 명령의 대상은 저장된 실행 기록이며, 과거 실제 tokenizer 재인코딩·KV/grammar 기술 검증은 보존된 receipt에 연결합니다.

[CPU 기록 재생 · main](https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain)

[Original mapping](publication/provenance/original_mapping.json) · [Public manifest](publication/PUBLIC_MANIFEST.json) · [Archive](publication/data/original-evidence.tar.xz) · [Redactions](publication/provenance/redactions.json)

2139개 논리 원파일을1553개 고유 object로 보존했습니다. 전체120 model episode와 모든 src Python bytes는 원형입니다. 개인 absolute home prefix가 있는82개 부수 문서/설정/receipt의 공개 사본만 symbolic root로 바꿨고 old/new bytes/hash·규칙·횟수를 기록했습니다. 원 연구·원 freeze는 수정하지 않았습니다. 해당 공개 사본은 원 파일과 byte-identical하다고 하지 않습니다. 경로 재바인딩은 publication wrapper에서만 하며 원 algorithm을 수정하지 않습니다. 원 whole project/environment/GPU 재구축은 이 archive의 범위가 아닙니다.

P0/P1/P2 원 entry 문서는 publication/originals/pre-publication에 있습니다. 원 base83e1cef에는 미커밋 Case018이 없었습니다. P1 실행 bundle e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b, P2 44287b4091ac8e9e892a263b5770200a140f9d90f97ccef19dca203a1d5241a2, seal d7afbef13bd8a46d46587348f17d25f26bf95b51ed55107a8b6fc64a46199ff5는 publication commit/build/deploy identity와 다릅니다.

[외부 증거 생략·중복 목록](publication/provenance/omissions.json): archive의2139개 논리 원문 외에 P0의 model-status 기록1개를 별도 원형 파일에서 복원합니다. 원 Case018의1739개 파일은 모두 대응하며, 외부 evidence의 중복·준비/임시·전체 profile은 endpoint 재생 필요 여부와 함께 생략 사유를 기록했습니다.
