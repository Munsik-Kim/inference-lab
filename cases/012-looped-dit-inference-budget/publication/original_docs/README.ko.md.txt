# Case 012 — 같은 생성 시간, 어디에 계산을 더 쓸까?

같은 Looped-DiT 저장본에서 내부 loop 깊이와 생성 단계 수에 시간을 나눠 쓰는 runner를 구현했습니다. 동일한 초기 잡음의 세 결과를 저장하고, 개수·색·좌우 요구를 조건을 가린 화면에서 평가할 수 있습니다.

[이미지와 결과](demo/viewer.ko.html) · [개별 이미지 평가](demo/annotation.ko.html) · [실행 코드](source/adapter.py) · [정식 보고서](REPORT.ko.md)

## 구현과 현재 관측

- **짝지은 생성·측정:** 공식 Euler와 동일한 결과, 초기 잡음 해시, 요청 시간·GPU sampling 시간·메모리를 별도 기록.
- **시간에 따른 설정 선택:** DEV 시간만으로 L1/S89 · L2/S66 · L4/S50을 고정. MAIN 16개 prompt × 4개 seed × 3설정 = **192장**, 별도 SMOKE **24장** 생성 완료.
- **평가와 탐색:** 한·영 blind annotation, JSON 내보내기/가져오기, 모든 입력의 세 이미지 비교, 모델 없는 CPU 검산.

DEV의 기준은 L4/S50 median **4.536초**입니다. A/B는 각각 기준보다 **7.09%/7.25%** 길어 5% 목표에 들지 못했습니다. 이 설정은 가까운 시간 예산의 비교이며, 정확히 같은 시간이라고 부르지 않습니다. 실제 MAIN 시간은 보고서의 64요청별 표에서 확인할 수 있습니다.

**품질 평가: 주석 대기 (`ANNOTATION_PENDING`).** 이미지를 생성했다는 사실과 조건을 만족했다는 판정은 구분합니다. 조건 충족 비율·paired 개선/손실·품질 기반 추천 preset은 사람의 주석 후 계산합니다.

## 직접 확인

```bash
python -m pip install -r requirements-cpu.txt
python analysis/analyze.py
python analysis/audit.py --output audit-receipt.json
```

[모든 원본 이미지](demo/gallery.ko.html) · [CPU/GPU 재현 안내](REPRODUCTION.md) · [고정 설정](configs/main_settings.json) · [출처와 기여](NOTICE.md)

범위: RTX 5080, B/32 EMA 하나, 512px, Euler, CFG=6, BF16 denoiser / FP32 T5, batch 1. B/32의 32는 patch 크기입니다. L=1도 4-loop 학습 저장본을 사용합니다.
