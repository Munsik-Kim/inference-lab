# Sources, rights and assistance

This study uses owned synthetic Case013 fixtures and the independent DIOVA INT4 reference implementation at repository base `e647b474bdc9fa558d851bd9c998b664b0e33638`. See [reuse identities](provenance/reuse.json). New source and these fixture copies follow the repository [Apache-2.0 license](LICENSE); this does not relicense external material.

Qwen provides Qwen3-4B-Instruct-2507 and its tokenizer under the model's Apache-2.0 notice, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. No model or derived INT4 weights are in this bundle. [Local model file identities](provenance/model_identity.json) are local SHA measurements, not a new official-digest comparison.

DualQuant's algorithm is by Mahsa Salmani, Sayeh Sharify and Hesham Mostafa. The independently implemented reference variant and D16 transport remain the Case013 contract. The unlicensed original vendor source is not copied. [DualQuant v1](https://arxiv.org/html/2610.13767v1).

Artin Tajdini, Manpreet Kaur and Peng Qi, *One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness*, arXiv2610.20576v1, motivates interface robustness. The author [README at0465909](https://github.com/ArtinTD/cross-interface-policy-regularization/tree/0465909ce219cdd1202affff75e41330697e4862) and actual paper HTML were retrieved. No CIPR code, training script, paper text or figure is redistributed. The probed root LICENSE URL returned404; no license grant is inferred. [Source audit](provenance/source_audit.json).

PyTorch and Transformers supply model execution; Python standard library supplies the CPU registry, fixture dispatch, bootstrap and explorer rendering. Additional plotting uses an already installed CPU plotting environment. Codex/ChatGPT assisted implementation, execution and separate scalar recomputation. This is not independent human peer review, author endorsement, or independent GPU replication.
