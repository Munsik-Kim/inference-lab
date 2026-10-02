# Attribution and scope

Looped-DiT and MiniT2I are the authors' pretrained models and architecture, from OpenSenseNova/Looped-DiT at commit `92a9c1914258361f78e03426588c7c215944b504`. The copied `vendor/looped_dit` inference source is byte-identical; its MIT notice is retained in `vendor/LICENSE`. `source/adapter.py` adapts the pinned Euler implementation to explicit initial noise and measurement boundaries. The checkpoint is the authors' B/32 EMA model; no model training is performed here.

FLAN-T5-Large (`google/flan-t5-large`, pinned revision in provenance) supplies the frozen text encoder/tokenizer under Apache-2.0. Only unchanged encoder/shared tensors are used. PyTorch, Transformers, NumPy, Pillow and Hugging Face Hub provide the runtime, serialization, numerical and download facilities under their respective licenses.

DIOVA by Munsik Kim contributes the bounded paired runner, dedicated-noise identity, measured-time selection, atomic attempt ledger, blind annotation and export/import contracts, CPU auditing, and bilingual result inspection. Codex assisted implementation, documentation and verification. The code agent does not supply independent human image labels.

The authors already study loop depth versus denoising steps. This Case does not introduce that question or a new image-generation algorithm. It measures an explicit local request boundary on RTX 5080 and prepares paired constraint evaluation. A 16-prompt user subset is not the official full GenEval benchmark. The GenEval evaluator, training data, CLIP/Mask2Former and large judge models were not downloaded or run.

No private model weights, environment, credentials or training data are included in the review bundle. Quality scores remain unavailable until actual annotations are supplied.
