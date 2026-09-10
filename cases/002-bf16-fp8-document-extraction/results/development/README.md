# Development evidence

These runs are not part of the 100-document primary evaluation or the latency summaries.

| Run prefix | Prompt | Purpose/outcome |
|---|---|---|
| dev-initial | v1 | Default V2 runner failed during WSL UVA initialization. |
| dev-v1 | v1 | V1 loaded the model but could not find the already-installed ninja executable on child PATH. |
| dev-v1-path | v1 | After the PATH fix, FlashInfer sampler JIT linking could not find CUDA libraries. |
| dev-common | v1 | Common V1 server with FlashInfer sampler disabled; 20 measured dev requests per model, 7/20 correct each. |
| dev-prompt2 | v2 | One shared prompt revision; same documents and gold, 20 requests per model, 7/20 correct each. This runner/configuration was frozen for evaluation. |

Each successful development server also ran one short and one long warmup. No unsuccessful startup reached the quality dataset. These failures were not counted as model quality results or silently retried as evaluation requests.

`runner_before_path_fix.py` preserves the runner source used by the first two startup attempts; its hash is recorded in their server.json files. Later attempts use the final scripts/run.py. The initial request layout is retained as `construction_messages` and `BASE_PROMPT` in scripts/data.py, while provenance/development_prompt_v1.json records its instruction. Version 2 is used by the final `messages` function. Documents/facts/gold remained unchanged; input-token metadata was recalculated. This makes the older request hashes and token counts interpretable without duplicating the dataset.

Full errors and shutdown warnings are retained with local path labels. The current experiment does not claim that the default V2 runner or FlashInfer sampler worked on this setup.
