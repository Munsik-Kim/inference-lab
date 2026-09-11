# Reproduction notes

The measured environment used Python 3.12.14, PyTorch 2.13.0+cu130, Transformers 5.17.0, NumPy 2.3.5, safetensors 0.8.0, huggingface_hub 1.30.0 and matplotlib 3.10.8. Use an isolated environment compatible with your GPU driver. This publication records the existing local environment; a new-machine installation has not been independently tested.

Example package pins for a fresh Python 3.12 environment:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers==5.17.0 numpy==2.3.5 safetensors==0.8.0 huggingface_hub==1.30.0 matplotlib==3.10.8
```

Check the resulting CUDA build and device support before continuing. `provenance/package_versions.txt` contains the actual visible package versions, including packages inherited read-only from an existing environment. Its presence is not a guarantee of dependency resolution on another machine. Neither vLLM nor FlashInfer is required for this audit.

The reproduction wrapper creates a new output directory, downloads only the pinned small checkpoint if missing, and runs each stage in a separate process. It refuses to overwrite the supplied results or reuse a populated trace directory. Substitute your own paths:

```bash
python scripts/reproduce.py --mode full --model-cache /path/to/model-cache --work-dir /path/to/new-private-work --output-dir /path/to/new-case003-output
```

`full` regenerates documents and token IDs, collects the small validation fixture, records the numerical tests, extracts development traces, searches the same 49 candidates, measures development/stress inputs, creates a new frozen specification, and only then extracts and evaluates held-out documents. A new freeze has its own environment/timing provenance; compare it with the supplied specification instead of pretending the hashes must match.

`--mode fixed-evaluation` preserves the supplied frozen inputs, code and calibrated setting and reruns only evaluation in a new output copy. It copies the **original** development summaries as references for transfer analysis. This mode is not a fresh calibration or an independent replication of the development stage. The script verifies the supplied frozen hashes before extracting evaluation traces.

To regenerate figures and summary tables from the already saved numerical evidence, without a GPU or model download:

```bash
python scripts/analyze.py
```

The four figures use `results/eval_units.jsonl` and `results/eval_rows.csv.gz`; calibration-transfer and synthetic tables also use their corresponding saved files. Large Q/K/V tensors and model weights are intentionally not distributed. The self-authored input text, exact tokenizer IDs, fixed revision, extraction code, settings and a small development fixture provide the regeneration path.
