# Maintaining the bilingual guides

The existing [case template](../../cases/TEMPLATE.md) remains the starting point for a separately authorized new case. The source revision now includes Cases001–007. These pages interpret existing evidence; they are not an alternative result database.

1. Read the relevant frozen case, its current publication record and underlying result. Record conflicts before changing a claim. Historical “not published” text in an original run record describes that snapshot, not today's repository state.
2. Update [claims.json](claims.json) with the source revision, hash, exact text/JSON locator, units, independent denominator and limits. Keep original and post-hoc evidence separate. Do not copy raw datasets into this map.
3. Update both language pages at the same depth. Keep arm names, native/shadow views, denominators, units and the strength of uncertainty statements aligned. Review meaning manually; matching numbers cannot prove equivalent translations.
4. Run the checks below from the repository root with an available Python 3.10+ interpreter. The checker requires only the standard library and uses an existing markdown-it-py parser when available. Choose a new report outside the checkout.

```bash
python -B -m unittest discover -s docs/_checks -p 'test_*.py' -v
python -B docs/_checks/check_docs.py --output /tmp/inference-lab-docs-check-new.json
```

When markdown-it-py is present, its extracted link destinations are cross-checked against the supported syntax. The checker handles this guide set's single-line inline links/images, ATX headings, fenced blocks, explicit ID anchors and claim comments. It rejects unsupported reference links and HTML in authored pages; it is not a complete Markdown parser. It checks source hashes/locators, selected literal displays, claim coverage and opposite-language/home navigation. It compares existing protected files against the recorded Git source revision, or an explicitly supplied private starting SHA256 inventory. It does not check external URL availability or prove scientific claims, translation quality or browser layout.

Follow [GitHub's README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) for repository-relative paths. Keep the root README as the entry point; do not add a `.github/README.md` that takes display precedence. The paired guides are [English](../en/CASEBOOK.md) and [Korean](../ko/CASEBOOK.md).

Changing these introductions does **not** authorize changing `cases/`, `notes/`, `downloads/`, licenses, frozen protocols or historical packages. Do not regenerate experiment ZIPs/checksums as a side effect. A new scientific result or publication revision needs its own review. Refresh both language download links only after checking actual current metadata, while preserving historical archives.

## Presentation tools

The editable bilingual guides and `presentation/` sit outside frozen case inventories.
`claims.json` keeps its original evidence revision; `protection_revision` separately
identifies the starting repository whose non-guide files must stay unchanged.
The source manifest in `presentation/` pins the exact public files read by the builder.
A new build records both that evidence revision and hashes of the presentation code.
Run the builder into a new external directory, then run `tools/showcase/check.py`.
The deployment artifact is an explicit allowlist, not a repository copy.

The CPU workflow and manual Pages workflow are prepared source files. Local results
are not remote CI or Pages publication. Update live URLs and badges only after an
approved remote run and readback. Preserve the case ZIPs and historical checksums
when editing these guides. Translation meaning and prose still require editorial review.
