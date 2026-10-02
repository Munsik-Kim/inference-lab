# Case 012 public GitHub edition

[한국어 소개](../README.ko.md) · [English overview](../README.md)

This edition adds a beginner-facing bilingual introduction, all 64 MAIN image pairs in Markdown, interpretation of saved measurements, and a supplemental AI assessment of all 192 MAIN images. It adds no generation, fitting or human annotation. The AI assessment is complete; the original human-pending summary is preserved as a historical record.

[AI results and method](assessment_v1/README.md) · [한국어 평가 결과](assessment_v1/README.ko.md) · [Every item judgment](assessment_v1/annotations.json). One AI rater viewed original PNGs with configuration labels masked. Prior exposure to 48 examples is disclosed; this is not independent fully blind human evaluation. Arithmetic is separately recalculated, which does not independently validate the AI's visual judgments.

The original 1,437 Case012 files are mapped in [original_inventory.json](original_inventory.json), with original sizes and hashes. Five edited documents retain their prior bytes in [original_docs](original_docs/); scientific code, configurations, records, figures, PNGs and historical receipts remain unchanged. [restore_original.py](restore_original.py) restores the original subtree into a new external directory before its CPU audit. It does not recreate the entire repository or the full review ZIP.

The original archive is `diova_case012_looped_dit_budget_review_20261002_v1r3.zip`, 132,922,838 bytes, SHA256 `668c9690d2afb7e96324626b6407f5623fb45af540fa45f8f25c451382b9340d`. It stays outside the Git repository, avoiding a second copy of the image data. No model weights or environments are published.

[verify.py](verify.py) checks original identities, the public inventory, saved request medians, historical missing labels and complete supplemental AI coverage. [PUBLICATION_SHA256SUMS](PUBLICATION_SHA256SUMS) hashes public files except itself. [example_identity.json](example_identity.json) identifies the earlier post-hoc non-blind illustration. The [source scalar analysis](posthoc/analysis.json) preserves request identities; [assessment_v1](assessment_v1/README.md) keeps subsequent scoring separate from the original experiment.

```bash
python publication/verify.py
python publication/assessment_v1/audit.py
python publication/restore_original.py --output /your/new-scratch/case012-original
```

Run from the Case012 directory. The output must not exist. GPU/model-dependent tests remain outside this CPU publication check. GitHub HTML links show source; the viewer and blind annotation run locally after cloning. Current publishing scope is a feature branch and review PR, with no merge or Pages deployment.
