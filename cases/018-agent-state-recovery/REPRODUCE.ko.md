# Case018 — 상태 복구와 최종 보고는 함께 맞을까? — CPU replay

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Python3.12+ 표준 라이브러리만 사용합니다. 새 외부 output을 요구하며 기존 output·연구 경로에는 쓰지 않습니다. 대략 RAM1GiB, 복원용 disk1GiB 여유를 권장합니다. 모델·tokenizer·Torch/vLLM/xgrammar 설치/다운로드가 필요 없습니다. GPU 경로는 이번 명령에 없습니다.

1. 공개 manifest·XZ signature/전체 member 경로·중복·regular file·bytes/hash·alias coverage를 검사하고 새 폴더에 복원합니다.
2. 각 phase의 실제 frozen source로 P0 26/P1 24/P2 96 endpoint·사전 관측·후속·종료·예산·raw SSE/IDs/usage를 독립 재생합니다.
3. 실제 token 합계·취소 결측·worker 비용·등록 paired bootstrap을 대조합니다.
4. 전체120행 새 사후 진단·표·선택 원문 excerpt·SVG를 재생성하고 공개 summary와 정확히 대조합니다.
5. --tests는 새 publication 경계17개를 검사합니다. 과거37/48/59개 검사 이력은 원 receipt 재사용이며 이번 전체 재실행이라고 쓰지 않습니다.

[Original mapping](publication/provenance/original_mapping.json) · [Public manifest](publication/PUBLIC_MANIFEST.json) · [Archive](publication/data/original-evidence.tar.xz) · [Redactions](publication/provenance/redactions.json)

2139개 논리 원파일을1553개 고유 object로 보존했습니다. 전체120 model episode와 모든 src Python bytes는 원형입니다. 개인 absolute home prefix가 있는82개 부수 문서/설정/receipt의 공개 사본만 symbolic root로 바꿨고 old/new bytes/hash·규칙·횟수를 기록했습니다. 원 연구·원 freeze는 수정하지 않았습니다. 해당 공개 사본은 원 파일과 byte-identical하다고 하지 않습니다. 경로 재바인딩은 publication wrapper에서만 하며 원 algorithm을 수정하지 않습니다. 원 whole project/environment/GPU 재구축은 이 archive의 범위가 아닙니다.

P0/P1/P2 원 entry 문서는 publication/originals/pre-publication에 있습니다. 원 base83e1cef에는 미커밋 Case018이 없었습니다. P1 실행 bundle e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b, P2 44287b4091ac8e9e892a263b5770200a140f9d90f97ccef19dca203a1d5241a2, seal d7afbef13bd8a46d46587348f17d25f26bf95b51ed55107a8b6fc64a46199ff5는 publication commit/build/deploy identity와 다릅니다.

[외부 증거 생략·중복 목록](publication/provenance/omissions.json): archive의2139개 논리 원문 외에 P0의 model-status 기록1개를 별도 원형 파일에서 복원합니다. 원 Case018의1739개 파일은 모두 대응하며, 외부 evidence의 중복·준비/임시·전체 profile은 endpoint 재생 필요 여부와 함께 생략 사유를 기록했습니다.
