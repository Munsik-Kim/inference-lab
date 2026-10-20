# Case016 stage map

| Stage | Evidence / result | Reproduction boundary |
|---|---|---|
| Original agent P0/P1 | B16 0/48,F4 6/48; baseline NOT_MET | Original source,amendments01–08 and96 records retained losslessly |
| Baseline CPU audit | 96 episodes,478 requests; score changes0 | Closed unchanged1.95MB audit bundle; CPU fixture replay |
| Native KV technical P1-T | Same4GiB pool,36 blocks,288 timed requests | New native vLLM/FlashInfer GPU requests; independent SSE/scalar audit |
| Agent P2/P3 | NOT_RUN | No new agent study |
| Pages | Builder/CI prepared | Manual publish=true was not dispatched |

[Original byte mapping](publication/provenance/original_mapping.json) · [Original P1 source freeze](results/p1_r1/code_freeze.json) · [Historical root report](publication/originals/agent-study/REPORT.ko.md) · [Technical protocol](followups/nvfp4-technical-r1/specs/protocol.json)

Historical root documents retain their exact bytes. Their original relative links work in the restored `agent-original` view created by the CPU wrapper. Large repeated JSON/HTML is stored as gzip; the wrapper checks decompressed original SHA256 before execution. Two private-path packaging helpers are not experiment sources and are omitted from public files. No original scientific file is replaced by the new report.
