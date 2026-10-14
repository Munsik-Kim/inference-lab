# Case 013 후속: 같은 history의 도구 선택

기존 실험 결과를 `diova-compare 0.1.2`에 연결하고, 같은 입력에서 도구 이름 선택과 오류 후 행동을 GPU로 비교했습니다. D16의 잘못된 이름 선택이 재현됐으며, 단일 projection 복원은 두 실패를 회복하지 못했습니다. 같은 비용의 대조 복원은 새 손실 두 개를 만들었습니다.

[English](README.md) · [결과 보고서](records/report_ko.md) · [CPU/GPU 재실행](README_REPRODUCE.md) · [판정](records/decision.json)

## 무엇을 사용할 수 있나요?

- [Case013 exporter](../../packages/diova-compare/src/diova_compare/case013.py): 원 기록의 source hash·입력·정답·실제 도구 실행 조건을 확인해 비교 형식으로 변환합니다. CPU/stdlib만 필요합니다.
- [동일-history runner](worker.py): 원 r2 artifact를 읽기만 하며 자연 생성, 강제 prefix, 오류 피드백을 구분합니다.
- [전체 projection 복원](intervention.py): weight만 바꾸지 않고 beta transport를 포함한 adapter를 원 nn.Linear로 교체합니다.
- [독립 CPU 검산](audit.py): 새 records의 성공 조건·pairing·margin·선택 규칙을 재계산합니다. 제외된 전체 logits/weights는 복구하지 않습니다.

## 결과

| 조건 | 기존8개 FREE DEV 성공 | 새 정답 | 새 손실 |
|---|---:|---:|---:|
| 새 D16 기준선 I0 | 6/8 | — | — |
| layer2 down_proj BF16 복원 I1 | 6/8 | 0 | 0 |
| layer35 동일 비용 대조 복원 I2 | 4/8 | 0 | 2 |

D16은 patch 이름의 첫 token에서 잘못된 CamelCase를 선호했습니다. 공통 성공 patch history의 validate 이름은 정확한 쪽을 선호해, 모든 도구에 같은 손상이 생긴 것은 아니었습니다. 가장 큰 마지막 입력-position 출력 오차를 고르는 규칙의 회복 효과는 확인하지 못했습니다.

이 기록은 이미 관측한 DEV의 후속 진단입니다. 고객 업무의 안정성·비열등성·packed serving 가속을 판정하지 않습니다. 기존 [Case 013](../../cases/013-dualquant-reference-quality/README.ko.md)의 자료와 판정은 유지했습니다.

## 한 가지 재사용 흐름

**양자화 전후 업무 샘플 기록 → 동일 입력·정답 계약 검사 → 정답 손실·새 정답·도구 계약 실패 보고서**

저장소 루트에서 설치한 CLI로 실행합니다. 출력은 모두 새 경로를 사용합니다.

```bash
diova-compare export-case013 --case cases/013-dualquant-reference-quality --output /tmp/case013-export-new
diova-compare validate --input /tmp/case013-export-new/agent/base/FREE/RTN.jsonl
diova-compare compare --baseline /tmp/case013-export-new/agent/base/FREE/RTN.jsonl --candidate /tmp/case013-export-new/agent/base/FREE/D16.jsonl --output /tmp/case013-compare-new
```

모드·길이·입력·정답 정의가 다르면 기본 pairing에서 거절합니다. 최종 JSON이 같아도 실제 수정·validation이 없으면 오답입니다. PPL과 저장 KL을 가짜 분류 정답률로 바꾸지 않습니다. [상세 사용법](../../packages/diova-compare/README.ko.md#case013-export).

## 검증과 공개 범위

현재 기능 검사50개(package46+진단4), 문서45개·사이트116개와10단계 build/검사가 통과했습니다. GPU worker9회,82generation 요청, 보수적 worker wall475.752초입니다. 기존 연구39개는 과거 receipt로 구분합니다. [검증](records/verification.json) · [원본 보존](records/preservation.json).

새 자료는 이 경로의 별도 manifest를 사용합니다. Case013의 닫힌 PUBLIC_MANIFEST는 수정하지 않았습니다. 공통 패키지0.1.1에 묶인 Case009/011의 옛 검사는 [외부 복원 wrapper](../../tools/run_case009_frozen_checks.py)와 [Case011 wrapper](../../tools/run_case011_frozen_checks.py)로 그대로 실행합니다. 현재0.1.2는 별도 CPU job으로 검사합니다.

공개 준비 records에는 개인 cache 경로를 식별자로 치환했습니다. 원본 로컬 명령·실패·hash는 외부 결과 폴더에 남아 있습니다. 모델·INT4 payload·전체 logits·activation vectors는 포함하지 않습니다. CPU 기록 검산은 가능하며 GPU 재실행은 원 checkpoint와 r2 artifact가 별도로 필요합니다. [export 규칙](export_public.py) · [별도 manifest](PUBLIC_MANIFEST.json).
