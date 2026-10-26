# Case020 결과 분석 보고서

- `ANALYSIS.ko.html`: 그림을 내장한 한국어 보고서. 단일 파일로 열 수 있다.
- `ANALYSIS.ko.md`: 수정 가능한 같은 내용의 원문. 그림은 `figures/`를 참조한다.
- `analysis_metrics.json`, `*_analysis.csv`, `memory_accounting.csv`, `setup_scenario.csv`: 원자료에서 다시 계산한 파생 값과 source hash.
- `analyze.py`: GPU를 사용하지 않는 분석·그림·보고서 생성 코드.

기존 Case020 공개 후보는 읽기만 했으며 source·raw·summary·PUBLIC_MANIFEST·이전 인계 TXT를 변경하지 않았다. 추가 GPU 실험, 모델/데이터 다운로드, Git 작업, 게시를 수행하지 않았다.

재생성 예시:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/optic-analysis-mpl \
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/home/mskim/miniforge3/envs/ai_lab/bin/python -B \
/home/mskim/Codex_main/case020-results-analysis-20261026-1418/analyze.py \
--case /home/mskim/Codex_main/inference-lab-case020-optic-decode-20261026-1215/cases/020-optic-on-demand-decode \
--out /tmp/optic-analysis-rebuild
```

관측한 CPU runtime은 Python3.10.19, NumPy2.2.6, Matplotlib3.10.8이다. 기존 환경을 읽기 전용 재사용했으며 clean environment나 다른 머신 재현을 주장하지 않는다. setup 교차점 그림은 측정값을 상수로 넣은 계산 시나리오이며 실제 수명주기 benchmark가 아니다.
