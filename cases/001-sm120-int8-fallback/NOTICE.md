# Attribution and license

The included `pr54316-vllm-0.29.0.patch` is the exact runtime hunk from hclsys's vLLM PR #54316, head `4878fe1154e1bccf70a18d170f5fa093f1190734`. It was applied locally to the official vLLM 0.29.0 wheel. The original fix and existing device-free gate tests are the PR author's work. The tests themselves are not duplicated in this bundle.

The original changed file retains these notices; applying this hunk does not remove them:

```text
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

The unmodified vLLM Apache-2.0 license text is included as `LICENSE`, sourced from release commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. The local reproducer scripts are provided under Apache-2.0 as well. No model weights are included or relicensed; obtain the pinned model separately under its provider's Apache-2.0 terms.

The validation scripts and documentation were prepared with substantial OpenAI Codex assistance.

Public evidence copies replace personal paths with `<ENV_BEFORE>`, `<ENV_AFTER>`, `<MODEL_SNAPSHOT>`, `<PROJECT>`, `<REPLAY_WORKSPACE>`, `<IPC_TMP>`, `<HOME>` and, if present, `<WSL_HOST>` / `<WINDOWS_USER>`. These are redaction labels, not missing input files. The build-time `/workspace/csrc/...` location in the exception is preserved because it identifies the wheel's C++ error. Package versions, source hashes, run IDs, exit codes, backend observations and generated text are retained. Original local logs remain unchanged.
