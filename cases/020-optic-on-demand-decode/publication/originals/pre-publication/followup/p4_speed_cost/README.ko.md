# Case020 P4 — 메모리는 작게, 속도와 준비 비용은 따로 판단하기

**필요한 임베딩만 복원하면 메모리는 얼마나 줄고, 언제 느려지는가?**

기존 LARGE/Z32 결과에서 B의 상주 tensor는 548,928 bytes, A의 전체 재구성 table은 268,435,456 bytes였다. 이번 후속 실험의 같은 process 비교에서는 C의 중복 제거가 B보다 빠른 block이 0/30이었다. 폭을 Z644까지 키워도 반복 입력의 C/B 주 지표는 1.142배였다. 실제 세션에서는 N128의 A/B 순서가 block마다 섞였고 N1024·4096에서는 A가 세 block 모두 빨랐다.

PLE는 token이 각 layer에서 사용하는 추가 임베딩이다. **codebook**은 작은 벡터 사전, **decoder**는 코드를 임베딩으로 되돌리는 affine 계산, **resident**는 계속 남겨두는 tensor, **workspace**는 연산 라이브러리가 사용하는 작업 공간이다.

[통합 보고서](INTEGRATED_REPORT.ko.md)에서 선택 기준을, [P4 상세 보고서](REPORT.ko.md)에서 모든 block 통계와 근거를 읽을 수 있다. [CPU 재현](REPRODUCE.ko.md)은 모델·GPU 없이 raw→표→그림을 다시 만든다. English: [README](README.md).

- A: 같은 압축 표현을 전체 복원한 BF16 table 조회.
- B: 수정하지 않은 공식 decoder로 요청 행 전체 복원.
- C: 매 호출 unique→U행 복원→inverse gather. 결과는 M행이다.
- S: 중복 없는 한 조건에서만 정렬 후 M행 복원·원 순서 복귀. 중복 제거 방법으로 세지 않는다.

P4는 **COMPLETE**다. 30/30 timing worker, 96/96 method-block, 27/27 lifecycle worker, 추가 preflight3·smoke1·memory3을 완료했다. 실패·재시도·누락은 없다. 원본 747개와 외부 원본 18개는 그대로 보존했다. 원격 게시와 CI는 실행하지 않았다.

신규 파일의 범위는 이 디렉터리이며 [`PUBLIC_MANIFEST_P4.json`](PUBLIC_MANIFEST_P4.json)이 식별한다. 원 [`REPORT`](../../REPORT.ko.md)와 원 manifest는 당시 기록이다.
