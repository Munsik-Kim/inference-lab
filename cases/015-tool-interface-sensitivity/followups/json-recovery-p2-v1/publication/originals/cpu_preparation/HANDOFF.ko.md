# 다음 세션 인계

이 폴더는 Case015 P2의 **CPU 준비 완료본**입니다. 새 모델 추론과 복구 결과는 없고, `run`은 exit 3으로 거절됩니다. 기존 A1 자유 생성 실패는 그대로 실패입니다.

1. [protocol](specs/protocol.json), [source freeze](provenance/code_freeze.json), [사전 등록](provenance/preregistration.json)을 먼저 읽습니다.
2. [REPORT.ko.md](REPORT.ko.md)와 [gates.json](results/cpu_preparation/gates.json)의 확인 범위를 유지합니다. tokenizer/hash 검사 PASS를 새 C0 생성 재현이나 GPU 문법 통합 PASS로 바꾸지 않습니다.
3. 현재 실행할 수 있는 다음 명령은 [REPRODUCE.ko.md](REPRODUCE.ko.md)의 `validate`, `audit`, `baseline --tests`, `dry-run`입니다. 각 명령은 새 외부 출력 폴더를 사용합니다.
4. 향후 모델 실행에는 별도 승인·누적 worker wall 예산과 검토된 GPU adapter가 필요합니다. 기존 A1 실행기의 가드를 우회하거나 과거 raw를 새 응답으로 넣지 않습니다. 기존 로컬 snapshot/RTN/D16 파일은 다시 만들지 않습니다.
5. N/E는 같은 **새 C0** malformed-only 응답에서 분기합니다. 관측과 파일, prefix 2/2 사용량, 실제 C0 비용을 공유합니다. G는 최초 응답에만 적용합니다.
6. 새 C0 출력 ID가 과거 24개 대응과 다르면 원인 해석을 보류하고 환경·실행 차이를 먼저 분리합니다. 원 모델·artifact·환경을 맞춘다는 이유로 설치/업그레이드/재양자화하지 않습니다.

실제 작업공간과 private 로컬 경로는 사용자에게 전달한 외부 `LOCAL_HANDOFF.json`에 있습니다. 공유 가능한 문서·결과에는 의미 있는 label만 사용했습니다. 원본 가중치·INT4 payload·환경·캐시는 임시 생성물이 아니므로 정리하지 않습니다.

Git base는 `51385286063eb55e791f13a0b97f1e341ce6a879`, branch는 `research/case015-json-recovery-p2-cpu-20261019`입니다. source freeze hash는 `03829dc8549ad45a0949c7f372568ff2fef76e02f4bbb1d5e22131f61e4f4c80`입니다. 새 commit/stage/push/PR/merge/Pages 게시를 하지 않았습니다.
