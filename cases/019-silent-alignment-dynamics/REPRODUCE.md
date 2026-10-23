# Case019 CPU recorded replay

Quick uses the small Git candidate; full uses the same source plus separate evidence. Neither command runs a new full ODE sweep or model inference. --tests separately executes small numerical fixtures in the original scientific tests.

## Isolated environment

Python3.11.15, NumPy1.26.4, SciPy1.17.1, Matplotlib3.10.9; all CPU requirements are pinned in `publication/requirements-cpu.txt`. Use a fresh venv without system/user site packages. Existing model/serving environments remain unchanged.

```sh
python3.11 -m venv /tmp/case019-cpu-env-new
/tmp/case019-cpu-env-new/bin/python -I -B -m pip install -r cases/019-silent-alignment-dynamics/publication/requirements-cpu.txt
/tmp/case019-cpu-env-new/bin/python -I -B -m pip check
```

## Quick — small public copy

```sh
/tmp/case019-cpu-env-new/bin/python -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-quick-new --mode quick --tests
```

Checks exact source/manifest, four declared raw slices and J/G/A arithmetic in the small complete summary. Frozen P0/P1/P2 tests27/20/18 load from distinct source roots. Quick is not reported as complete17/81/128-path recorded replay.

## Full — public source plus distributed evidence

Full records use the [evidence asset](https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz) in the dedicated `case019-evidence-v1.0.0` Release. Download is an explicit user step; imports, quick checks and site builds never fetch it. Supply the file with verified bytes/SHA256 using `--evidence`. Location declaration and actual upload/public-download verification are separate records.

Asset: `case019-recorded-evidence-v1.tar.xz`
Bytes: 608195012
SHA256: `d4fe6ba161de97a9f27cd409eb7966e8a4bab9ded7a04e9db6a261bf93084736`
Exact logical restoration: 1382 files/659370532bytes.

```sh
curl --fail --location --output case019-recorded-evidence-v1.tar.xz https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz
sha256sum case019-recorded-evidence-v1.tar.xz
```

```sh
/tmp/case019-cpu-env-new/bin/python -I -B cases/019-silent-alignment-dynamics/publication/run_checks.py --output /tmp/case019-full-new --mode full --evidence /path/to/case019-recorded-evidence-v1.tar.xz
```

Output must not exist; original runs/receipts are never overwritten. Only regular content-addressed objects are accepted; path escapes, links, special files, duplicate members, alias chains, size/hash/coverage mismatches and XZ CRC corruption are rejected. NPZ uses allow_pickle=False. Historical absolute paths remain inert original text, never active imports or input paths.

Full coverage: P0 17×1002 grids/136 physical checkpoints; P1 81×1702=137862/891; P2 128×1677=214656/1920. Original event/paired/mode tables, J/G/A,81 graph paths/144 persistence series and stored reference evidence are recalculated. The16 new P2 reference files have220 saved physical checkpoints. Existing all-grid GAUGE full-Z comparisons survive as error series and receipts; unsaved full factor paths cannot be recreated by recorded replay.

Source/raw bytes and discrete keys/status/null/order are exact. Scientific metric/reference tolerances retain each frozen protocol. Derived CSV/JSON float comparison was fixed in advance at atol1e-10/rtol1e-9. Cross-environment figure bytes are not promised; actual byte identity and data/axes/units/legends/visual checks are reported separately.

Receipts record actual imports and sys.path. Only explicitly verified phase source is imported, with PYTHONPATH removed and -I/-B. Repeated --deny-path options add Python open-hook checks against original roots; this is not an OS sandbox and is explicitly installed in each child. A full ODE rerun is a different computation and is not performed in P3.

A complete previously restored directory can also be passed to `--evidence`. Every mapped regular file and the exact logical coverage are rechecked; no second large restoration copy is made. The initial archive CRC/hash receipt remains separate from this directory reuse.
