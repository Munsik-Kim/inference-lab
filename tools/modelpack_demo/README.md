# Tiny CPU artifact round trip

A **randomly initialized test fixture** demonstrates the preserved Case008 serializer and layer-aware loader. It is not a pretrained Qwen model or a quality/speed benchmark. [Korean instructions](../../docs/ko/GETTING_STARTED.md#tiny-model-demo) · [English instructions](../../docs/en/GETTING_STARTED.md#tiny-model-demo).

Use an existing compatible Python 3.12 environment, or prepare a separate temporary CPU environment. Run from the repository root:

```bash
DEMO_ENV=$(mktemp -d)
python3.12 -m venv "$DEMO_ENV/env"
"$DEMO_ENV/env/bin/python" -m pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$DEMO_ENV/env/bin/python" -m pip install -r tools/modelpack_demo/requirements-cpu.txt
"$DEMO_ENV/env/bin/python" -B tools/modelpack_demo/roundtrip.py --output "$DEMO_ENV/roundtrip"
```

Dependency setup downloads Python packages, not model weights. Running the fixture requires no runtime network request; Hugging Face offline flags are set, and pretrained loading is prohibited in the reload worker. CPU outputs remain in the new external directory, not the repository. The CLI also resolves its source paths correctly from another working directory.

The builder creates a BF16 two-layer Qwen fixture (hidden 32, intermediate 64, vocabulary 64), slices layer 1 to 48 intermediate channels, saves it, and exits. The parent copies the artifact and verifies its file hashes. A new process loads only the copy; an audit hook rejects reads under the builder directory. Checks cover full block/norm/logit finiteness, `[64,48]` widths, gate/up/down shapes, tied embedding/head identity, and exact output hashes. Small forwards are CPU fixture operations, never research measurements.

`summary.json` records the two process IDs, fixture kind, observed checks and scope. The result depends on the pinned arithmetic environment; a mismatch blocks success. Four wrapper tests include absent validity, invalid dimensions and overwrite rejection. Historical artifact tests cover missing tensors and corrupted structure. No broad support claim is made for other architectures or devices.
