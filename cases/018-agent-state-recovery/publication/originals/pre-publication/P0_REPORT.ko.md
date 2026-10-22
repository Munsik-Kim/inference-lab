# Case018 P0 — 복구한 AI는 다음 업무도 제대로 수행할 수 있을까?

**판정: P0 PASS / P1_CPU_READY.** 이는 모델 실험을 위한 CPU 계약 준비 완료입니다. 실제 모델·tokenizer/문법 backend·KV 기술 검증 완료가 아닙니다. 새 GPU/model 요청0, 서버0, 설치0, 원격 쓰기0이며 P1/P2/P3는 미실행입니다.

## 구현한 것과 질문

재고 수량을 맞춰도 활성 처리 원장이 남으면 다음 이동이 막힐 수 있습니다. 하나의 재고 환경에 잘못된 대상 이동(A), 정상1회 뒤 중복(B), 부분 차감(C)을 주입하고, 공개 보상 도구로 복구한 뒤 고정 후속 업무 두 개를 비교하는 실행기와 평가기를 구현했습니다. 원문 근거와 차이는 [SOURCE_MAP](SOURCE_MAP.md), 세부 계약은 [P0_PROTOCOL](P0_PROTOCOL.ko.md)에 있습니다.

Q는 종료 상태·후속 업무 적합(B∧P∧C1∧C2), W는 여기에 관측된 최신 검증·정확한 final·정상 final 종료를 더한 workflow 완주입니다. 공개 validation은 불변식만 검사하므로 checkpoint와 다를 때도 true일 수 있습니다. 가드 역시 정답값을 읽지 않습니다.

## 작업 identity와 보존

- Worktree: `<LOCAL_HOME>/Codex_main/inference-lab-case018-state-recovery-20261022-093529`
- Branch: `research/case018-agent-state-recovery-20261022-093529`
- 실제 fetch base와 시작/종료 HEAD: `83e1cef6dd60384931a38e016ea5e8d069d40674`.
- 조상 `/`, `/home`, `/home/mskim`, `<LOCAL_HOME>/Codex_main` 및 저장소에서 적용되는 AGENTS.md 없음. 열린 PR/로컬 경로·브랜치에 다른 Case018 없음 확인 후 생성했습니다.
- 기존 Case017 공개 참조41개 + 원 연구40개 = source/spec/핵심 receipt/보고서 **81개 before/after hash 불일치0**. 원 연구와 참조 worktree의 HEAD/branch/status/staged/tracked diff가 시작값과 같습니다. [보존 receipt](provenance/preservation_after.json).
- 새 worktree에서 Case018 밖 tracked diff0, staged0, commit/push0. 원 model weight 전체 재해시·과거 전체 검사·GPU 재실험은 하지 않았습니다. 원 ZIP 전체 hash 검증까지 했다는 뜻도 아닙니다.

## CPU control과 독립 감사

아래는 **모델 성공률이 아닌 SYNTHETIC_CONTRACT_TEST**입니다. DEV6에서 세 control을 각각 한 번 저장한 최종 batch입니다.

| control | CPU trace | Q=1 | W=1 |
|---|---:|---:|---:|
| clean 정상 checkpoint | 6 | 6 | 6 |
| no_recovery 주입 상태 유지 | 6 | 0 | 0 |
| scripted_correct 공개 도구 보상+해제 | 6 | 6 | 6 |

18 trace의 terminal continuation은36 branch이며 checkpoint reference도36입니다. 반례8 trace의 terminal/reference는 각각16 branch를 별도로 가집니다. 합계26 trace를 raw에서 재파싱·독립 전이·관측·상태·후속 절차·예산/종료·endpoint로 재계산해 모두 일치했습니다. 두 branch를 독립 모델 표본으로 더하지 않습니다.

| 진단 script | B | F | C1/C2 | Q | W |
|---|---|---|---|---|---|
| DEV-C00 business_only/R0 | 1 | 1 | 0/0 | 0 | 0 |
| DEV-A00 stale_validation/R0 | 1 | 0 | 1/1 | 1 | 0 |
| DEV-A00 wrong_business/R1 | 0 | 1 | 0/0 | 0 | 0 |
| DEV-A00 no_final/R0 | 1 | 0 | 1/1 | 1 | 0 |
| DEV-A00 guard_retry/R0 | 1 | 0 | 1/1 | 1 | 0 |
| DEV-A00 guard_retry/R1 | 1 | 1 | 1/1 | 1 | 1 |
| DEV-A00 guard_twice/R1 | 1 | 1 | 1/1 | 1 | 1 |
| DEV-A00 repeated_final/R1 | 0 | 0 | 0/0 | 0 | 0 |

수량만 복구한 C00은 B1/F1이지만 활성 원장이 남아 C1/C2/Q0입니다. 올바른 복구 후 stale validation만 본 A00은 Q1/F0/W0입니다. 현재 validation이 true여도 잘못된 수량은 B0/Q0으로 남습니다. final 없는 복구도 Q1/W0으로 계산합니다. recovered=true인 미복구 control도 D1/Q0이며, 선언은 정답이 아닙니다.

가드 재검증 script의 R0/R1은 같은 최초 입력에서 시작합니다. R0의 부적격 final은 받아 사후 W0, R1은 거절→validation→final로 W1입니다. 이는 합성 분기 검사이며 실제 AI의 가드 효과나 성공 증가 측정이 아닙니다. R1의 세 번째 부적격 final은 GUARD_LIMIT이고 feedback은2회입니다.

## 검사·예산·실행 비용

최종 고유 unittest **37 PASS / failure0 / error0 / skip0**입니다. parser(중복 key·NaN·bool/int·숫자 문자열·다중 JSON·여분 prose), 보호 피해 후 원복/감사 prefix 삭제/새 음수, 합법적 revision 증가, unseen event, payload 누출, 정상 도구 오류·예산 경계, 16번째 final/17번째 금지, HOLDOUT gate, source/trace/관측 변조·manifest 누락/중복, symlink 및 CPU import/help 무SDK·무네트워크 경로를 검사했습니다.

첫 준비 run은 고유36개 PASS였고 장부 검사를 보강한 최종 run이37개 PASS입니다. 합산해73개 고유 검사라고 하지 않습니다. 첫 run receipt와 소스 사본을 보존했고 [CPU amendment](provenance/p0_amendment01.json)에 변경을 기록했습니다. endpoint나 모델 조건은 바꾸지 않았고 모델 관측도 없었습니다. 그 이전의 개발용18조합 in-memory probe도 모델 측정이 아닙니다.

정답 복구 script 최대7 requests/parsed attempts, 가드 거절2회 뒤 최신 검증과 final까지 최대9회로 공통16회 안에 듭니다. 후자의 남은 호출 여유는7회입니다. 실제 토큰 길이는 미검증이며 P1 gate입니다. 16번째 non-final 실행 후 TOOL_CALL_LIMIT, 16번째 final 수용, token/context/request/episode 한도 분리를 CPU simulated units로 검사했습니다.

최종 저장 trace의 fake 요청은 control78+진단51=129회, simulated output units는2910+1778=4688입니다. 실제 model requests/usage가 아닙니다. 단위는 UTF8 bytes/4 올림이며 fake 시간0.01초/응답도 simulated입니다. 단위 검사 내부의 추가 mock 호출은 이 canonical trace 분모에 넣지 않습니다.

P0 명령 전체(37 검사·control·진단·내부 audit·I/O)의 실제 CPU wall은 **2.047653초**, process CPU **2.023022초**, 프로세스 peak RSS **27476KiB**였습니다. 구현·문서 작성 전체 경과와 다릅니다. 별도 감사 명령의 exec wall은0.856531초, 별도 CPU/RSS는 NOT_MEASURED입니다. GPU worker time/실제 model tokens는 미측정(null)이며 비용0으로 추정하지 않습니다.

## 실제 실행한 명령과 결과

Case 디렉터리에서 Python `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]` / `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39`로 실행했습니다. 설치 없이 표준 라이브러리만 사용했습니다.

```bash
PYTHONPATH=src python3 -B -m agent_state_recovery p0 --output results/p0_r2
PYTHONPATH=src python3 -B -m agent_state_recovery audit --run results/p0_r2 --output <LOCAL_HOME>/Codex_main/case018-p0-evidence-20261022-093529/final_cpu_audit.json
```

[verification](results/p0_r2/verification.json), [tests](results/p0_r2/tests.log), [raw manifest](results/p0_r2/run_manifest.json), [내부 독립 audit](results/p0_r2/independent_audit.json), [control/반례](results/p0_r2/control_and_counterexamples.json), [예산 장부](results/p0_r2/budget_feasibility.json). 출력 경로는 새 경로만 받으며 기존 run/receipt를 덮어쓰지 않습니다. 다른 위치에서 새로 검사하는 명령은 [재현 안내](REPRODUCE.ko.md)에 있습니다.

## 동결과 다음 단계의 미검증 항목

최종 [working source freeze](provenance/code_freeze.json)는 src/tests/specs/DEV 원문 **23개**를 hash합니다. commit을 만들지 않았으므로 base commit이나 과거 실행 commit과 혼동하지 않습니다. 문서·결과는 별도 산출물입니다.

- Working source: `0ddcc3df452bb29998b5df2704eb3c48fbd3ca2eac5405c78782490247f77c86`
- protocol.json: `996e1ed65d3f9be59e5e7d211ce4a12dc8cb202ca1c9380a16e2c812eaa769e2`
- tools.json: `19d5a3a4c89f7382ac377b1e91dd734950599c6bf1c79a43248a63af41c14234`
- grammar.json: `33492b1ed8e71bc752bd5b33fb1861e418650de473a3291186c5d6236e590daf`
- DEV manifest: `1d68a9e0bb414b5cabfd03785ef4d8262a993a95854142b9651fa70240a7541a`

HOLDOUT generator/분리 seed·24개·동결 field 목록과 실패 시 P2 생성 거절을 구현했습니다. 실제 HOLDOUT input/gold는0개입니다. 성공수 gate는 명시적으로 null이며, P1 완료·freeze·기술 gate가 필요합니다. 실제 모델 점수/통계 구간은 만들지 않았습니다.

현재 BLOCKED 항목은 없습니다. 실제 로컬 artifact identity 재확인, 서버 adapter 구현, pinned tokenizer/template, grammar의 실제 토큰 mask, native BF16/NVFP4 저장/읽기 및 raw usage/deadline/abort 통합·최초 token IDs 비교는 **NOT_RUN/P1에서 필요**입니다. 기록된 Qwen revision/환경만 읽었으며 현재 weights를 검증·load하지 않았습니다. P1/P2/HOLDOUT 생성·P3/UI·사이트 연결·Git 게시를 자동 진행하지 않고 P0에서 종료합니다.
