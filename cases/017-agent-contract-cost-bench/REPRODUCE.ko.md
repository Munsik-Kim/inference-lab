# Case017 — 저장 결과 재생과 모델 재실행의 구분

일반 clone의 저장소 root에서 Python3.10+ 표준 라이브러리만 사용합니다. 의존성 설치·모델·tokenizer·GPU·개인 경로가 필요하지 않습니다. 아래 출력 경로는 존재하지 않는 새 경로여야 합니다. 압축 원자료 약2.83MB는 복원하면 약250MB이며 여유 공간1GB를 권장합니다.

```bash
python3 -B cases/017-agent-contract-cost-bench/publication/run_checks.py --output /tmp/case017-public-check-new --tests
```

실제 실행 경로는 공개 manifest·원 바이트를 검사하고, `original-view/cases/017-agent-contract-cost-bench/`에 529개 보존 파일과5개 pinned Case016 import를 복원합니다. 새 fixture에서 P1 64개/P2 192개 원문을 재파싱·실행하고 별도 endpoint·원 bootstrap10000회·표시 JSON·보완 ledger를 대조합니다. 원84개 단위 검사·8개 postprocess 검사와 새 공개 층 검사는 구분해 기록합니다. `verification.json`과각 로그가 결과입니다. 이는 저장 기록 재생이며 GPU 재현이 아닙니다.

표시 CSV만 새로 만들려면 다음 명령을 씁니다. 원본 성공 bool만 합산하지 않고 기록된 사건·최종 상태에서 독립 endpoint를 계산합니다. 전체 fixture 재생은 위 명령입니다.

```bash
python3 -B cases/017-agent-contract-cost-bench/publication/derive.py --output /tmp/case017-display-new
```

GPU 재실행은 별도 권한과 pinned snapshot/environment/private runtime이 필요합니다. 원 실측 명령과 필드는 [원 재현 문서](publication/originals/pre-publication/REPRODUCE.md#gpu-reproduction-boundary)에 보존했습니다. 완료된 result root는 덮어쓰기를 거절합니다. 다른 PC의 경로로 원 freeze를 고쳐 통과시키지 마세요. 공개 CPU 명령은 새 추론·서버·재토큰화·설치·P3를 실행하지 않습니다. 원안의6/8 gate를 누락한 과거 진행을 반복하거나 새로 승인하는 명령도 아닙니다.

공개본은 root 문서6개의 원 바이트를 originals에 보존했습니다. 과거 generated demo/browser 산출물·재생으로 재생성 가능한 fixture 상태는 생략했습니다. 전체256개 endpoint 재생에 필요한 원 cells/input/schema/source/receipt는 보존합니다. [전체 대응표와 생략 이유](publication/provenance/original_mapping.json)를 확인하세요.
