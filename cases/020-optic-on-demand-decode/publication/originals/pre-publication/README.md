# How much memory does decoding only the requested embeddings save, and when is it slower?

Per-layer embeddings (PLE) are additional embeddings each token uses at each layer. This operator case compares a dense table reconstructed from the **same OPTIC artifact** (A), the unchanged official request decoder (B), and unique→official decode→inverse gather (C).

Resident module tensor storage fell from **16MiB to0.143MiB** (SMALL) and **256MiB to0.523MiB** (LARGE). All **30 cells /90 cell-blocks** completed and passed every-layer numerical gates. Timed allocated peaks include output, temporary tensors and additional retained framework allocations; payload ratios are not whole-device VRAM ratios.

| Shape | A resident bytes | B/C resident bytes | A / B payload ratio | NPZ bytes / arrays bytes |
|---|---:|---:|---:|---:|
| SMALL | 16,777,216 | 149,568 | 112.17× | 151634 / 149504 |
| LARGE | 268,435,456 | 548,928 | 489.02× | 550998 / 548864 |

| LARGE M2048/U512 | A | B | C |
|---|---:|---:|---:|
| Timed allocated peak MiB | 304.02–304.02 | 80.79–80.79 | 52.57–52.57 |
| Timed reserved peak MiB | 338.00–338.00 | 82.00–82.00 | 102.00–102.00 |

Payload ratio and allocated/reserved/device memory are separate measurements.


Across the primary grid, C was faster than B in **5/30 paired blocks**. **0 conditions** had C<B in all three blocks. The decision is `OPERATOR_VALIDATED, MEMORY_TRADEOFF_OBSERVED, DEDUP_UNRESOLVED_BLOCK_VARIATION`. Large host/event block variation prevents a general speed ranking; retain all block points and paired values.

| Block | A host median µs | B host median µs | C host median µs | C/B |
|---:|---:|---:|---:|---:|
| 0 | 29.55 | 123.14 | 869.78 | 7.064 |
| 1 | 60.16 | 126.41 | 247.07 | 1.955 |
| 2 | 25.35 | 178.65 | 860.85 | 4.819 |


![Memory accounting](figures/01_memory.png)

A codebook maps code indices to small vectors; the layer decoder applies an affine map. Resident storage persists between calls; workspace is temporary storage during a call. B pays for lookup, latent construction, GEMM, bias and BF16 conversion. C additionally pays for unique/sort/inverse reconstruction and returns all M rows.

![Synchronized operator latency](figures/02_latency.png)

Labels: `OPERATOR_VALIDATED, MEMORY_TRADEOFF_OBSERVED, DEDUP_UNRESOLVED_BLOCK_VARIATION`. The descriptive three-block consistency rule was selected after observing the diagnostic run, before corrected primary timing; it is not a preregistered statistical threshold. Faster individual blocks remain visible.

[Reproduce](REPRODUCE.md) • [Full results and methods](REPORT.md) • [Korean](README.ko.md)

```bash
# From this case directory. These commands need no model, Torch, or CUDA.
python src/audit_v2.py verify
python src/audit_v2.py summarize --out /tmp/optic-summary
MPLCONFIGDIR=/tmp/optic-mpl python src/audit_v2.py figures --out /tmp/optic-figures
python tests/test_audit_v2.py -v
```

The experiment uses synthetic artifacts, raw unscaled BF16 [M,L,H] output and warm-cache eager operators. It does not measure trained OPTIC quality, language capability, full-model serving, TTFT or TPOT. No model/dataset download, vLLM installation, site integration, Git stage/commit/push/PR/Release was performed.

[Paper §2/§6](https://arxiv.org/html/2610.31309v1) • [Pinned implementation](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/src/ple_eval/compact_adapter.py) • [Source map](SOURCE_MAP.md) • [Reuse](REUSE_MAP.md) • [Notices](THIRD_PARTY_NOTICES.md)

Case020 • run `20261026-1235` • local candidate, uncommitted.

Base: `3e6f2c6b36fb2a3853cc3677bee84904d28567cf`  
Branch: `research/case020-optic-on-demand-decode-20261026-1215`  
Upstream: `344c712aeb6620e39eb78930fb922447f0e123f3`  
Main source/input/config bundle: `c43ea3a0a27cd28be7b9e8e26ce5cd65a42af28c070a49cd1f1b06cd77bc489b`  
Protocol SHA256: `c930534682204b99990bc58540c170925bd32ada3045a0027367e50130f34987`  
GPU worker SHA256: `fd0716a262f3444b4bbec49131ed06ecd04532337f5a962119b3367a5fdf5b77`
