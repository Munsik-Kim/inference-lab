# Case 013 — 가중치 오차가 줄면 실제 답변도 좋아질까?

[English](README.md) | 한국어

공개 INT4 수식으로 가중치를 변환·저장·복원하는 도구와, 같은 입력에서 문장 점수·장문 답변·도구 호출을 비교하는 실행기를 만들었습니다. RTX 5080에서 Qwen 0.6B와 4B의 모든 Transformer projection에 적용하고, 실패한 출력까지 기록했습니다.

DualQuant의 공개 INT4 연산을 바탕으로 비공개 blockfmt 없이 실행하는 독립 reference variant입니다. 0.6B에서는 D16의 PPL이 RTN보다 낮았습니다. 4B에서는 weight MSE와 표본 위치 KL이 작아졌지만 PPL과 자유 도구 DEV는 개선되지 않았습니다. SM120 packed W4A16 실행은 확인했으나 전체 모델 수치 기준은 미통과여서 serving 속도·최대 문맥 이득은 아직 확립되지 않았습니다.

**행·열 스케일 보존 · 실제 4-bit packing · 동일 입력 품질 비교**

[무엇을 만들었나](#implementation) · [핵심 결과](#results) · [직접 확인](#reproduce) · [설계·출처](#sources)

<a id="implementation"></a>
## 무엇을 만들었나

| 기능 | 구현·근거 |
|---|---|
| 가중치만으로 행·열 스케일을 계산하는 INT4 변환 | [solver](source/paper_int4.py), [수식·저장 형식](source/reference_int4.py) |
| code와 모든 필요한 scale의 저장·재로딩 | [nibble codec](source/reference_int4.py), [원래 artifact 장부](records/artifact_manifest.json) |
| 열 스케일을 입력에도 적용하는 전체 모델 adapter | [ReferenceLinear](source/paper_int4.py), [순차 실행기](source/paper_model_runner.py) |
| 장문 정답·근거와 실제 도구 실행 조건을 따로 채점 | [실행·채점 코드](source/paper_model_runner.py), [독립 CPU 검산](analysis/audit.py) |
| 실제 packed CUDA 실행과 전체 모델 수치 대응을 별도로 검사 | [최종 probe source](provenance/source_snapshot/probe_public_marlin.py), [실패를 포함한 기술 기록](PACKED_PROBE.md) |

PyTorch · Transformers · NumPy · SDPA · vLLM/Marlin probe를 사용했습니다. DualQuant는 원저자의 방법입니다. 이 프로젝트는 공개 INT4 수식을 옮긴 **reference variant**와 저장·평가·검산 도구를 구현했습니다. 논문 전체 실행이나 저자의 비공개 packed 라이브러리를 재현한 결과가 아닙니다.

<a id="results"></a>
## 핵심 결과

가중치의 복원 오차는 두 모델에서 작아졌지만, 답변 품질은 같은 방향으로 움직이지 않았습니다.

| 모델 | 원본 BF16 PPL | 단순 INT4 RTN | 행·열 보정 D16 | D16 + 활성값 INT4 D4 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 21.275 | 27.576 | 24.445 | 34.734 |
| Qwen3-4B-Instruct-2507 | 10.133 | 10.810 | 10.894 | 12.339 |

PPL은 다음 기호에 준 확률 점수에서 계산하며 **작을수록 좋습니다**. 모델별 WikiText-2 일부 32구간 × 2,048기호, 유효 점수 65,504개입니다. 서로 다른 모델의 전체 능력 순위나 공식 benchmark 전체 점수가 아닙니다. [반올림 전 수치](records/quality_4b.csv) · [공통 표시 데이터](summary/project.json)

4B의 자유 도구 호출에서는 BF16·RTN **8/8**, D16 **6/8**, D4 **0/8** 작업을 완료했습니다. 같은 작업 중 네 개를 16K 입력으로 늘린 후속 DEV에서는 **4/4 · 4/4 · 3/4 · 0/4**였습니다. 이미 주어진 도구 결과를 읽는 replay는 모두 정답이어서, 그 점수와 직접 도구를 선택하는 성공률을 구분합니다.

실제 packed Marlin GEMM과 한 block은 수치 대응 기준을 통과했습니다. 전체 모델은 상대 오차 **2.61%**로 미리 정한 **2%** 한도를 넘었습니다. **실제 serving 가속·최대 문맥 이득은 아직 확인되지 않았습니다.** 본 품질 비교는 복원된 BF16 가중치 연산입니다.

<a id="reproduce"></a>
## 직접 확인

저장 기록의 검산은 GPU·모델·Torch 없이 NumPy만 사용합니다. 저장된 출력은 모두 [records](records)에 있습니다.

```bash
python3 -B cases/013-dualquant-reference-quality/analysis/audit.py --output /tmp/case013-audit.json
python3 -B -m unittest discover -s cases/013-dualquant-reference-quality/tests -v
python3 -B cases/013-dualquant-reference-quality/publication/verify.py --output /tmp/case013-public.json
```

출력 파일은 새 경로를 사용하세요. 필요한 CPU 의존성, 선택적 Torch 합성 검사, 실제 모델 재실행 요건은 [재현 안내](REPRODUCTION.ko.md)에 나눠 적었습니다. 가중치와 전체 logits는 공개 묶음에 없습니다.

<a id="sources"></a>
## 설계·출처

[정식 보고서](REPORT.ko.md) · [수식·실행 계약](records/reference_contract.json) · [전체 입력·평가 계획](records/experiment_manifest.json) · [보존·공개 대응표](provenance/export_map.json) · [기여·라이선스](NOTICE.md)

[논문·코드·실행 계약표](METHOD_CONTRACT.md) · [평가 프로토콜](EVALUATION_PROTOCOL.md) · [실패와 성공 사례](FAILURE_ANALYSIS.md) · [세 결과 그림](figures/README.md) · [후속 계획: 미실행](ROADMAP.md)

질문을 확인하는 순서는 복원 오차 → 문장 점수 → 장문 exact 답변 → 자유 도구 호출입니다. 작은 DEV 관측에서 이들 지표가 갈리는 조건을 볼 수 있으며, formal TEST나 GPTQ와의 serving 비교는 수행하지 않았습니다.
