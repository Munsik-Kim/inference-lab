# Case014 sources and boundaries

Project code follows the repository Apache-2.0 license. No third-party source was copied or relabeled. [Source audit](references/source_audit.json) records Tianhao Guo's paper and official code identity; the independent ridge/RTN implementation is not official full QEP reproduction. Original P0 source/docs/records are preserved byte-for-byte in the historical snapshot.

External model/data licenses remain applicable. No pretrained or INT4 model weights, source document text, activation arrays or logits are distributed. The tiny synthetic codec/matrix fixture is not model weights.

P1 executed30 core settings on96 original document groups, plus separate controls and one smoke document. It is dense BF16 quality reference execution with real INT4 serialization; no native serving speed/capacity claim is established. P2/P3 and remote publication were not run. Codex assisted implementation and reporting; no independent human review is claimed.
