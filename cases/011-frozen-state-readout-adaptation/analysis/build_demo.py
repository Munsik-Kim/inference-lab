#!/usr/bin/env python3
"""Build a small, offline EN/KO table from completed Case 011 aggregates.

This is a recorded-result reader. It does not run a model or modify the DIOVA
homepage, Pages configuration, or historical Case 010 files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

CASE = Path(__file__).resolve().parents[1]
HEADS = ("ORIGINAL", "SHORT_REFIT", "MIXED_REFIT")
STORAGES = ("NATIVE_FP32", "UNIFORM_8")

COPY = {
    "en": {
        "title": "Case 011 · Frozen states, adapted readouts",
        "intro": "Compare three final linear readouts on each shared recurrent rollout. The original checkpoint, state dynamics, and storage codec stay fixed.",
        "scope": "Recorded TEST: 1,024 paired sequences × 2,048 group tokens; three existing checkpoints; Native FP32 and INT8 storage. Refit positions stop at 256.",
        "report": "Methods and full results", "reproduction": "CPU recomputation", "home": "Project README",
        "primary": "Primary comparison: INT8 MIXED_REFIT − ORIGINAL",
        "interval": "Each interval uses 5,000 paired base-sequence resamples at 98.333…% confidence: Bonferroni family 95% across the three checkpoints, with approximate bootstrap coverage.",
        "table": "All 18 logical arms", "seed": "Checkpoint", "storage": "Storage", "readout": "Readout",
        "all": "All", "rmst": "RMST0 (tokens)", "horizon": "Empirical T₀.₀₅", "accuracy": "All-token accuracy",
        "ce": "Gold CE (nats)", "valid": "Valid / total score tokens", "delta": "Δ RMST0 (tokens)",
        "bounds": "Bootstrap interval (tokens)", "legend": "RMST0 counts consecutive correct group tokens before the first error. Empirical T₀.₀₅ is a descriptive horizon, not a confidence-supported lower bound. Later correct answers count toward token accuracy but do not restore first-error survival.",
        "heads": "ORIGINAL uses the frozen head. SHORT_REFIT fits native features at positions 1–32; MIXED_REFIT uses a fixed sample of positions 1–256. Both fit 16,384 supervised feature rows and transfer unchanged to INT8.",
        "empty": "0 matching recorded arms", "count": "matching recorded arms", "download": "Read aggregate JSON",
        "curves": "First-error survival and primary paired differences", "curve_alt": "Three checkpoint panels comparing INT8 first-error survival for the original, short-refit and mixed-refit readouts.",
        "delta_alt": "Checkpoint-specific paired RMST0 differences between INT8 mixed-refit and original readouts, with bootstrap intervals.",
        "source": "Source identity", "controls": "Input-only and state-shuffle controls, position-range scores, and storage contrasts are included in the linked report and aggregate JSON.",
        "footer": "Offline recorded-result viewer · No model execution · No new measurements are made by this page",
    },
    "ko": {
        "title": "Case 011 · 상태는 고정하고 판독층을 보정하기",
        "intro": "같은 recurrent rollout을 세 가지 마지막 선형층으로 읽습니다. 원래 checkpoint, 상태 전이, 저장 codec은 그대로 유지합니다.",
        "scope": "기록된 TEST: 같은 입력 1,024개 × 군 token 2,048개, 기존 checkpoint 3개, Native FP32·INT8 저장. 보정에 사용한 최대 위치는 256입니다.",
        "report": "방법과 전체 결과", "reproduction": "CPU 재계산", "home": "프로젝트 README",
        "primary": "주 비교: INT8 MIXED_REFIT − ORIGINAL",
        "interval": "각 구간은 같은 입력을 묶어 5,000회 재표집한 98.333…% bootstrap 구간입니다. 세 checkpoint의 Bonferroni family 95% 수준이며, bootstrap의 근사 구간입니다.",
        "table": "18개 판독 설정", "seed": "Checkpoint", "storage": "저장 방식", "readout": "판독층",
        "all": "전체", "rmst": "RMST0 (token)", "horizon": "경험적 T₀.₀₅", "accuracy": "전체 token 정답률",
        "ce": "정답 CE (nats)", "valid": "유효 / 전체 점수 token", "delta": "RMST0 차이 (token)",
        "bounds": "Bootstrap 구간 (token)", "legend": "RMST0는 첫 오류 직전까지 연속해서 맞힌 군 token 수입니다. 경험적 T₀.₀₅는 기술통계이며 신뢰지원 하한이 아닙니다. 오류 이후 다시 맞힌 답은 token 정답률에 포함하지만 최초 실패 생존율에는 복귀시키지 않습니다.",
        "heads": "ORIGINAL은 원래 판독층입니다. SHORT_REFIT은 native 특징의 위치 1–32, MIXED_REFIT은 위치 1–256의 고정 표본을 사용합니다. 두 보정 모두 지도 특징 16,384행을 적합하고, 같은 판독층을 INT8에 그대로 적용합니다.",
        "empty": "조건에 맞는 기록 0개", "count": "개 설정 표시", "download": "집계 JSON 보기",
        "curves": "최초 실패 생존곡선과 주 비교", "curve_alt": "세 checkpoint에서 원래 판독층·짧은 보정·혼합 보정의 INT8 최초 실패 생존곡선을 비교한 그림.",
        "delta_alt": "INT8 혼합 보정과 원래 판독층의 checkpoint별 RMST0 차이와 bootstrap 구간.",
        "source": "자료 식별자", "controls": "입력만 사용하는 대조·상태 특징 순열 대조, 위치 구간별 점수, 저장 방식 비교는 연결된 보고서와 집계 JSON에서 확인할 수 있습니다.",
        "footer": "저장된 결과를 읽는 로컬 화면 · 모델 실행 없음 · 이 화면에서 새 측정은 수행하지 않음",
    },
}

STYLE = """
:root{color-scheme:light;--ink:#22333d;--muted:#4d6069;--copper:#955e36;--line:#d9e0df;--paper:#f7f7f2}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.6}
main{max-width:1120px;margin:auto;padding:32px 22px}header{margin-bottom:30px}h1{font-size:clamp(1.7rem,4vw,2.8rem);line-height:1.2;margin:.4em 0}h2{font-size:1.35rem;margin:1.8em 0 .7em}a{color:#75411e;text-underline-offset:3px}a:focus-visible,select:focus-visible{outline:3px solid #267a7c;outline-offset:4px}.brand{font-weight:700;letter-spacing:.1em}.language{float:right}.intro{font-size:1.12rem;max-width:850px}.scope,.note{color:var(--muted)}.scope{border-left:3px solid var(--copper);padding-left:15px}.links{display:flex;flex-wrap:wrap;gap:10px 24px}.filters{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:12px}label{display:flex;align-items:center;gap:8px}select{font:inherit;background:white;color:var(--ink);border:1px solid #72848c;border-radius:4px;padding:6px 10px}.table-wrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.92rem}th,td{text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#eef1ed;font-weight:650}tbody tr:last-child td{border-bottom:0}td.num{text-align:right}tr[hidden]{display:none}.status{min-height:1.6em;color:var(--muted)}img{display:block;width:100%;height:auto;margin-top:18px;background:white;border:1px solid var(--line);border-radius:6px}code{overflow-wrap:anywhere;white-space:normal;font-size:.84rem}footer{border-top:1px solid var(--line);margin-top:35px;padding-top:18px;color:var(--muted);font-size:.88rem}@media(max-width:480px){main{padding:23px 15px}.filters{display:grid;grid-template-columns:1fr;gap:9px}label{justify-content:space-between}select{min-width:170px}th,td{padding:9px 10px}.links{gap:9px 15px}}
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_url(target, output):
    return quote(os.path.relpath(Path(target), Path(output)).replace(os.sep, "/"), safe="/.-_")


def fmt(value, precision=3):
    return "—" if value is None else f"{value:,.{precision}f}"


def prepare_rows(result):
    arms = result["arms"]
    if len(arms) != 18 or len(result["primary"]) != 3:
        raise ValueError("viewer requires the complete frozen 18-arm / three-primary aggregate")
    seen = set()
    rows = []
    for arm in arms:
        key = (arm["checkpoint_seed"], arm["storage"], arm["head"])
        if key in seen or key[0] not in (0,1,2) or key[1] not in STORAGES or key[2] not in HEADS:
            raise ValueError("duplicate or unknown arm identity")
        seen.add(key)
        if arm["n_sequences"] != 1024 or arm["max_group_tokens"] != 2048:
            raise ValueError("viewer must label the actual complete predeclared TEST")
        scores = [s for s in result["scores"] if (s["checkpoint_seed"], s["storage"], s["head"]) == key
                  and s["position_scope"] == "position_range"]
        if [(s["first_group_token"], s["last_group_token"]) for s in scores] != [(1,32),(33,128),(129,256),(257,512),(513,1024),(1025,2048)]:
            raise ValueError("score intervals incomplete or out of order")
        valid = sum(s["valid_score_observations"] for s in scores)
        total = sum(s["total_token_observations"] for s in scores)
        if total != 1024*2048:
            raise ValueError("score token denominator differs from TEST identity")
        ce = sum(s["gold_ce_sum_nats"] for s in scores)/valid if valid else None
        rows.append({**arm, "gold_ce_nats": ce, "valid_score_tokens": valid, "total_score_tokens": total})
    return rows


def render(result, language, links, source_hash):
    text = COPY[language]
    rows = prepare_rows(result)
    esc = html.escape
    title = esc(text["title"])
    primary_rows = []
    for item in sorted(result["primary"], key=lambda row: row["checkpoint_seed"]):
        lo, hi = item["interval_tokens"]
        primary_rows.append(f'<tr><th scope="row">{item["checkpoint_seed"]}</th><td class="num">{item["mean_delta_tokens"]:+,.3f}</td><td class="num">[{lo:+,.3f}, {hi:+,.3f}]</td></tr>')
    table_rows = []
    for row in rows:
        seed, storage, head = row["checkpoint_seed"], row["storage"], row["head"]
        table_rows.append(f'<tr data-seed="{seed}" data-storage="{esc(storage)}" data-head="{esc(head)}"><th scope="row">{seed}</th><td>{esc(storage)}</td><td>{esc(head)}</td><td class="num">{fmt(row["rmst0_tokens"])}</td><td class="num">{row["empirical_t005_tokens"]}</td><td class="num">{100*row["token_accuracy"]:.2f}%</td><td class="num">{fmt(row["gold_ce_nats"])}</td><td class="num">{row["valid_score_tokens"]:,} / {row["total_score_tokens"]:,}</td></tr>')
    filters = []
    for name, label, values in (("seed", "seed", (0,1,2)), ("storage", "storage", STORAGES), ("head", "readout", HEADS)):
        options = '<option value="">' + esc(text["all"]) + '</option>'
        options += ''.join(f'<option value="{esc(str(value))}">{esc(str(value))}</option>' for value in values)
        filters.append(f'<label for="{name}-filter">{esc(text[label])}<select id="{name}-filter" data-filter="{name}">{options}</select></label>')
    # Escape '<' in JSON to prevent a closing script tag in any source string.
    embedded = json.dumps({"rows": rows, "primary": result["primary"], "source_sha256": source_hash}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    js_text = json.dumps({"empty": text["empty"], "count": text["count"]}, ensure_ascii=False).replace("<", "\\u003c")
    headings = ("seed", "storage", "readout", "rmst", "horizon", "accuracy", "ce", "valid")
    return f'''<!doctype html>
<html lang="{language}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="referrer" content="no-referrer"><title>{title}</title><style>{STYLE}</style></head>
<body><main><header><a class="language" id="language-link" href="{esc(links["language"])}" lang="{'en' if language=='ko' else 'ko'}">{'English' if language=='ko' else '한국어'}</a><span class="brand">DIOVA</span><h1>{title}</h1><p class="intro">{esc(text["intro"])}</p><p class="scope">{esc(text["scope"])}</p><nav class="links"><a href="{esc(links["report"])}">{esc(text["report"])}</a><a href="{esc(links["reproduction"])}">{esc(text["reproduction"])}</a><a href="{esc(links["readme"])}">{esc(text["home"])}</a><a href="{esc(links["aggregate"])}">{esc(text["download"])}</a></nav></header>
<section aria-labelledby="primary"><h2 id="primary">{esc(text["primary"])}</h2><div class="table-wrap"><table><thead><tr><th scope="col">{esc(text["seed"])}</th><th scope="col">{esc(text["delta"])}</th><th scope="col">{esc(text["bounds"])}</th></tr></thead><tbody>{''.join(primary_rows)}</tbody></table></div><p class="note">{esc(text["interval"])}</p></section>
<section aria-labelledby="arms"><h2 id="arms">{esc(text["table"])}</h2><p>{esc(text["heads"])}</p><div class="filters">{''.join(filters)}</div><p class="status" id="filter-status" role="status" aria-live="polite">18 {esc(text["count"])}</p><div class="table-wrap" tabindex="0" aria-label="{esc(text["table"])}"><table id="arm-table"><thead><tr>{''.join('<th scope="col">'+esc(text[key])+'</th>' for key in headings)}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><p class="note">{esc(text["legend"])}</p><p>{esc(text["controls"])}</p></section>
<section aria-labelledby="curves"><h2 id="curves">{esc(text["curves"])}</h2><img src="{esc(links["survival"])}" alt="{esc(text["curve_alt"])}" loading="lazy"><img src="{esc(links["difference"])}" alt="{esc(text["delta_alt"])}" loading="lazy"></section><footer>{esc(text["footer"])}<br>{esc(text["source"])}: <code>aggregate.json SHA256 {esc(source_hash)}</code></footer>
<script type="application/json" id="recorded-data">{embedded}</script><script>
'use strict';
const messages={js_text};
const selectors=[...document.querySelectorAll('[data-filter]')];
const rows=[...document.querySelectorAll('#arm-table tbody tr')];
const languageLink=document.getElementById('language-link');
const languageBase=languageLink.getAttribute('href');
function updateFilters(save){{
 let visible=0;const params=new URLSearchParams();
 for(const selector of selectors)if(selector.value)params.set(selector.dataset.filter,selector.value);
 for(const row of rows){{row.hidden=selectors.some(s=>s.value&&row.dataset[s.dataset.filter]!==s.value);if(!row.hidden)visible++;}}
 document.getElementById('filter-status').textContent=visible?String(visible)+' '+messages.count:messages.empty;
 languageLink.setAttribute('href',languageBase+(params.size?'#'+params.toString():''));
 if(save)history.replaceState(null,'',location.pathname+location.search+(params.size?'#'+params.toString():''));
}}
function restoreFilters(){{const params=new URLSearchParams(location.hash.slice(1));for(const s of selectors){{const value=params.get(s.dataset.filter)||'';s.value=[...s.options].some(o=>o.value===value)?value:'';}}updateFilters(false);}}
for(const s of selectors)s.addEventListener('change',()=>updateFilters(true));
window.addEventListener('hashchange',restoreFilters);restoreFilters();
</script></main></body></html>'''


def build(aggregate_path, output, case=CASE):
    aggregate_path, output, case = Path(aggregate_path).resolve(), Path(output).resolve(), Path(case).resolve()
    if output.exists():
        raise ValueError("demo output exists; choose a new directory")
    def bad(value):
        raise ValueError(f"nonfinite JSON value {value}")
    result = json.loads(aggregate_path.read_text(), parse_constant=bad)
    if result.get("status") != "COMPLETE_SAVED_SCALAR_AGGREGATION" or result.get("input_pairing", {}).get("status") != "PASS":
        raise ValueError("demo needs completed, independently paired scalar results")
    prepare_rows(result)
    required = [aggregate_path.parent/"figures/int8_survival.png", aggregate_path.parent/"figures/primary_rmst0.png",
                case/"README.md", case/"README.ko.md", case/"REPORT.md", case/"REPORT.ko.md", case/"REPRODUCTION.md"]
    for path in required:
        if not path.is_file():
            raise ValueError(f"missing demo link target: {path.name}")
    source_hash = digest(aggregate_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+".partial-", dir=output.parent))
    for language in ("en", "ko"):
        report = case/("REPORT.ko.md" if language == "ko" else "REPORT.md")
        readme = case/("README.ko.md" if language == "ko" else "README.md")
        links = {"report": relative_url(report, output), "readme": relative_url(readme, output),
                 "reproduction": relative_url(case/"REPRODUCTION.md", output),
                 "aggregate": relative_url(aggregate_path, output),
                 "survival": relative_url(aggregate_path.parent/"figures/int8_survival.png", output),
                 "difference": relative_url(aggregate_path.parent/"figures/primary_rmst0.png", output),
                 "language": "en.html" if language == "ko" else "ko.html"}
        (stage/f"{language}.html").write_text(render(result, language, links, source_hash))
    (stage/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case 011 · DIOVA</title><main><h1>Case 011 · Frozen-State Readout Adaptation</h1><p><a href="ko.html" lang="ko">한국어 결과 보기</a> · <a href="en.html" lang="en">Read results in English</a></p></main></html>')
    identity = {"scope": "local recorded-results viewer; no site deployment or model execution",
                "aggregate_sha256": source_hash,
                "files": {p.name: digest(p) for p in sorted(stage.glob("*.html"))},
                "external_requests": [], "row_count": 18,
                "links_scope": "relative files in this Case 011 tree; use with the case review package"}
    (stage/"manifest.json").write_text(json.dumps(identity, indent=2, allow_nan=False)+"\n")
    stage.rename(output)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=CASE)
    args = parser.parse_args()
    result = build(args.aggregate, args.output, args.case)
    print(json.dumps({"status": "BUILT", "row_count": result["row_count"], "files": list(result["files"])}))


if __name__ == "__main__":
    main()
