# NVFP4 KV 기술 후속: 원형 보존과 새 실측

[Case overview](../../README.ko.md) · [Report](../../REPORT.ko.md) · [CPU / GPU reproduction](../../REPRODUCE.ko.md)

[Frozen protocol](specs/protocol.json) · [Execution source hash](provenance/execution_freeze.json) · [Kernel-path annotation](provenance/kernel_path_annotation.json) · [Raw/CSV](results/analysis/)

원 agent P0/P1을 본 뒤 설계한 별도 기술 후속입니다. 가중치는 BF16이고 실제 NVFP4 KV만 비교합니다. 36개 고정 block/288 timed 요청, 별도 P0·warmup·sanity54요청을 완료했습니다. F4 prefill은 FA2, decode는 packed data/scale을 받는 XQA입니다. 기존 agent 판정은 바꾸지 않았습니다.
