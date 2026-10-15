# Case014 publication checks and responsive previews

Local pre-commit review of the unified P0–P2 publication, based on `4b6c838e98392743d5109d581dfb43dbc11c6e6e`.
The browser receipt identifies the built presentation fingerprint. Publication/merge/deployment revisions are separate from historical experiment source hashes.

[Case014 overview](../../../cases/014-quantization-compensation/README.md) · [CPU reproduction](../../../cases/014-quantization-compensation/REPRODUCE.md) · [ZIP identity](../../../downloads/case014_compensation_unified_publication_v1.json)

- [CPU scope](scope.json): rerun 15 P0 + 10 P1b + 9 P2 tests, 6 new publication tests; 45 docs and 134 site tests. Repeated runs are not additional unique tests.
- [Raw/CI recalculation](cpu-recalculation.json): P1 2,880 rows / 120 intervals; P1b 3,360 rows / 12 choices / 462 intervals; P2 320 rows / 8 aggregates / 10 comparisons / 20 interval fields / 392 CAL decompositions / 588 artifact metadata records.
- [Site regression](site-checks.json), [site links and source hashes](site.json), [documentation](docs.json).
- [Actual Edge checks](browser.json): 390 / 768 / 1280px, both languages, static JS-off content, keyboard details, anchors, language/back, figures and Case008–013/practical-guide regressions. This is a responsive viewport check, not an actual iPad test. Edge used an isolated temporary profile and disabled GPU rendering.
- [Fresh ZIP extraction](zip-verification.json): safety, CRC and all 614 selected-file hashes, followed by the CPU path in a new extracted folder.
- [Originals](original-preservation.json): 579 original Case014 files and 5,743 prior-case files unchanged. The publication copy retains 578 byte-identical research files and labels one sanitized model-path metadata copy; original private evidence remains unchanged.

| View | Korean | English |
| --- | --- | --- |
| Home 390px | [PNG](ko-index-390.png) | [PNG](en-index-390.png) |
| Home 768px | [PNG](ko-index-768.png) | [PNG](en-index-768.png) |
| Home 1280px | [PNG](ko-index-1280.png) | [PNG](en-index-1280.png) |
| Case card 390px | [PNG](ko-case014-home-card-390.png) | [PNG](en-case014-home-card-390.png) |
| Case card 1280px | [PNG](ko-case014-home-card-1280.png) | [PNG](en-case014-home-card-1280.png) |
| Detail 390px | [PNG](ko-case014-390.png) | [PNG](en-case014-390-wrapped.png) |
| Detail 768px | [PNG](ko-case014-768.png) | [PNG](en-case014-768.png) |
| Detail 1280px | [PNG](ko-case014-1280.png) | [PNG](en-case014-1280.png) |
| Results 390px | [PNG](ko-case014-results-390.png) | [PNG](en-case014-results-390.png) |
| Results 1280px | [PNG](ko-case014-results-1280.png) | [PNG](en-case014-results-1280.png) |

The site keeps archives out of Pages deploy members; the ZIP download is linked to a Git blob whose bytes match its published SHA256. No new model inference, K fitting, re-quantization or Monte Carlo run was performed during publication.

## HTTPS load-completion follow-up

The first deployed version (`d058f7b…`) matched all 360 artifact file hashes and 44 selected live HTTP files. A first HTTPS browser attempt reached the Korean figures before one lazy-loaded SVG finished: [original attempt](https-attempt-01.json). The cost SVG became complete with a valid natural width after another 108ms: [Korean image states](ko-https-image-load.json).

The browser utility now polls bounded image completion, retaining the same all-images-loaded predicate. [Source-change identity](https-load-wait-change.json) and [the HTTPS browser receipt](https-browser.json) identify the separate code and tested deployment. Both languages passed all **93 HTTPS checks**; the **94 file-mode checks** include one extra no-network-dependency assertion. The English images were already complete at first inspection ([states](en-https-image-load.json)). This corrects the verification utility’s wait, without revising any experiment or score.

## Narrow-screen text wrapping

An additional text-range check on the HTTPS page found a **6px** English overflow at 390px: a slash-joined runtime-settings token extended to x380.906 while the content viewport ended at x375 ([before](narrow-screen-before.json)). The original width assertion used scrollbar-inclusive `innerWidth`, which missed this small overflow.

Case014 now uses spaced setting separators and scoped `overflow-wrap:anywhere`. Its browser assertion compares against the actual `clientWidth`. All **94 file-mode browser checks** passed with the stronger assertion ([receipt](narrow-screen-browser.json)); [updated 390px view](en-case014-390-wrapped.png). The old [pre-fix view](en-case014-390.png) remains retained. This is a presentation-only change; the scientific source, measurements, Case014 manifest and public ZIP are unchanged.
