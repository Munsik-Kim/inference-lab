# Case019 CPU 기록 검산

Quick는 공개 Git 후보만, full은 같은 소스와 별도 evidence를 사용합니다. 어느 명령도 새 full ODE sweep이나 모델 추론을 실행하지 않습니다. --tests는 원 과학 검사의 작은 수치 fixture를 별도로 실행합니다.

## 격리 환경

Python 3.11.15, NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.10.9를 사용합니다. CPU 의존성 전체는 `publication/requirements-cpu.txt`에 고정했습니다. system/user site package를 사용하지 않는 새 venv를 만들며 기존 모델·서빙 환경은 변경하지 않습니다.

```sh
python3.11 -m venv /tmp/case019-cpu-env-new
/tmp/case019-cpu-env-new/bin/python -I -B -m pip install -r cases/019-silent-alignment-dynamics/publication/requirements-cpu.txt
/tmp/case019-cpu-env-new/bin/python -I -B -m pip check
```

## Quick — 작은 공개 사본

```sh
/tmp/case019-cpu-env-new/bin/python -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-quick-new --mode quick --tests
```

정확한 source/manifest, 명시한 4 개 raw slice, 작은 전체 요약의 J/G/A 산술을 검사합니다. frozen P0/P1/P2 과학 검사는 27/20/18 개로 각각 다른 source root에서 로드합니다. Quick를 17/81/128 개 전체 기록 replay로 표시하지 않습니다.

## Full — 공개 소스+배포 evidence

전체 기록은 전용 Release `case019-evidence-v1.0.0`의 [evidence asset](https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz)으로 배포합니다. 아래 다운로드는 사용자가 명시적으로 실행하는 별도 단계입니다. import/quick/사이트 빌드는 다운로드하지 않으며, full에는 bytes·SHA256을 확인한 파일을 `--evidence`로 지정합니다. URL 선언과 실제 업로드·공개 다운로드 검증은 별도 기록입니다.

Asset: `case019-recorded-evidence-v1.tar.xz`
Bytes: 608195012
SHA256: `d4fe6ba161de97a9f27cd409eb7966e8a4bab9ded7a04e9db6a261bf93084736`
정확한 논리 복원 범위: 1,382 개 파일 / 659,370,532 bytes.

```sh
curl --fail --location --output case019-recorded-evidence-v1.tar.xz https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz
sha256sum case019-recorded-evidence-v1.tar.xz
```

```sh
/tmp/case019-cpu-env-new/bin/python -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-full-new --mode full --evidence /path/to/case019-recorded-evidence-v1.tar.xz
```

출력 경로는 존재하지 않아야 하며 원 run/receipt를 덮어쓰지 않습니다. regular content-addressed object만 허용하며 경로 탈출·link·특수 파일·중복 member·alias chain·크기/hash/coverage 불일치와 XZ CRC 손상을 거절합니다. NPZ는 allow_pickle=False로 읽습니다. 원 기록의 역사적 절대 경로는 inert text로 보존하며 새 import/입력 경로로 쓰지 않습니다.

Full coverage: P0 17×1002 grids/136 physical checkpoints; P1 81×1702=137862/891; P2 128×1677=214656/1920. Original event/paired/mode tables, J/G/A,81 graph paths/144 persistence series and stored reference evidence are recalculated. The16 new P2 reference files have220 saved physical checkpoints. Existing all-grid GAUGE full-Z comparisons survive as error series and receipts; unsaved full factor paths cannot be recreated by recorded replay.

source/raw bytes와 discrete keys/status/null/order는 exact입니다. 과학 metric/reference tolerance는 각 frozen protocol을 유지합니다. 공개 derived CSV/JSON float 비교만 사전에 atol1e-10/rtol1e-9를 고정했습니다. 그림의 환경 간 byte 동일성은 약속하지 않으며 실제 byte 결과와 데이터/축/단위/범례·시각 확인을 분리합니다.

실제 imports와 sys.path는 receipt에 기록합니다. 검증된 명시적 phase source만 import하며 PYTHONPATH를 제거하고 -I/-B를 사용합니다. --deny-path를 여러 번 주면 Python open hook으로 지정 원 경로를 차단하는 추가 검산을 할 수 있습니다. 이는 OS sandbox가 아니며 각 child에 명시 적용합니다. 새 ODE 전 구간 재실행은 이 기록 검산과 다른 계산이며 이번 P3에서 실행하지 않습니다.

이미 완전히 복원한 디렉터리도 `--evidence`로 지정할 수 있습니다. 대응표의 모든 regular file과 정확한 파일 범위를 다시 확인하며 대형 사본을 추가로 만들지 않습니다. 최초 archive CRC/hash 검사와 복원 디렉터리 재사용은 별도 receipt로 남깁니다.
