# Case 011 attribution and source boundaries

DIOVA Case 011 is an implementation and evaluation study by Munsik Kim, developed with Codex assistance. The contribution is the integration of a fixed recurrent-state execution path with a precisely delimited final-linear intervention, controlled fitting, paired lifetime evaluation, patch save/reload and CPU verification. The method's performance is established only by the recorded results in [the report](REPORT.md).

## Reused model, task and state implementation

- **OpenEuroLLM / ComplexKDA** provides the original group word-problem task, model architecture and learned readout. This study retains commit `ef9d108d1692387cae37f5b2d539a71826a127c1`. It uses the same three previously trained local S3 checkpoints; it does not retrain their recurrent backbone. The upstream source tree and model weights are not included in this review package.
- **DIOVA Case 010 v2** provides the retained physical-coordinate recurrence, coefficient table, actual packed INT8/native state, terminal-state representation and byte-only restart implementation. These files remain unchanged under [Case 010 versions](../010-ckda-finite-precision-memory-horizon/VERSION_MAP.md). [The original v1 notice](../010-ckda-finite-precision-memory-horizon/versions/v1/NOTICE.md) and [the original v2 README](../010-ckda-finite-precision-memory-horizon/versions/v2/README.md) retain their source and execution context. Original source/calibration hashes used here are listed in [the reference identity record](provenance/adapter_reference_identity.json).
- **PyTorch and NumPy** provide model operations, CPU linear algebra, L-BFGS optimization, arrays and deterministic random generators. Their upstream authors retain credit for these libraries. No custom CUDA kernel or new quantizer is claimed.

## Probing reference

John Hewitt and Percy Liang (2019), *Designing and Interpreting Probes with Control Tasks*, Proceedings of EMNLP-IJCNLP, pages 2733–2743. [Official ACL Anthology record](https://aclanthology.org/D19-1275/), DOI: `10.18653/v1/D19-1275`.

The paper motivates caution about interpreting probe accuracy without controls. Case 011 uses a restricted current-token/position predictor and a within-token/position feature permutation as its own diagnostics. It does not reproduce that paper's specific control-task experiment or claim a new theory of probing. The project reviewed the official record/abstract for this connection; it does not attribute additional unverified theorems to the paper.

## Licenses and distribution

The repository [LICENSE](../../LICENSE) applies to project-authored material according to its terms. Original notices and upstream license requirements remain applicable to reused material; repository integration does not replace them. Original private checkpoints, full fitting features, environments and credentials are outside this review package. The included code is a research path with explicit model/source requirements, not a universal recurrent-model library or a hosted inference service.

The two fitted tensors, when available locally, require the identified original checkpoint. Their storage format and metadata do not turn them into a complete standalone model. The public distribution contains code, compact measurement records and patch manifests. Original checkpoint and fitted head tensor values remain excluded; a public patch manifest is not a model download.
