"""Post-hoc analysis of saved records; no model execution or annotation writes."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values):
    return statistics.median(list(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scalars-only", action="store_true", help="No figures or plotting dependencies")
    a = ap.parse_args()
    root, out = a.case, a.output
    out.mkdir(parents=True, exist_ok=True)
    index_path = root / "analysis/record_index.json"
    index = json.loads(index_path.read_text())["rows"]
    rows, hashes = [], {}
    for original in index:
        path = root / "results/attempts" / original["job_id"] / ("attempt-" + str(original["attempt"])) / "record.json"
        raw = json.loads(path.read_text())
        for field in raw:
            assert raw[field] == original[field], (path, field)
        rows.append(raw)
        hashes[str(path.relative_to(root))] = sha(path)
    settings = ["A_time", "B_time", "C_time"]
    main_rows = [r for r in rows if r["job"]["phase"] == "main"]
    assert len(main_rows) == 192 and all(r["status"] == "SUCCESS" for r in main_rows)
    lookup = {(r["job"]["prompt_id"], r["job"]["seed_label"], r["job"]["setting"]["id"]): r for r in main_rows}
    assert len(lookup) == 192
    keys = sorted({(p, seed) for p, seed, setting in lookup})
    for p, seed in keys:
        rr = [lookup[p, seed, setting] for setting in settings]
        assert len({r["initial_noise_sha256"] for r in rr}) == 1
        assert len({r["input_ids_sha256"] for r in rr}) == 1
        assert all(sha(root / next(x["relative_image_path"] for x in index if x["job_id"] == r["job_id"])) == r["image_sha256"] for r in rr)
    result = {
        "status": "POST_HOC_SAME_RECORDED_SCALARS",
        "model_forward_count": 0,
        "quality_status": "ANNOTATION_PENDING",
        "quality_score": None,
        "source_hashes": {"analysis/record_index.json": sha(index_path), **hashes},
        "input_pair_count": len(keys),
        "independent_prompt_clusters": len({p for p, seed in keys}),
        "time_uncertainty": "Descriptive only: MAIN shares one process; repeats have only two pairs per process. No CI calculated.",
        "settings": {},
        "paired_main_time": {},
    }
    for setting in settings:
        rr = [r for r in main_rows if r["job"]["setting"]["id"] == setting]
        config = rr[0]["job"]["setting"]
        residual = [r["complete_seconds"] * 1000 - r["sampler_gpu_ms"] for r in rr]
        result["settings"][setting] = {
            "loops": config["loops"], "steps": config["steps"], "requests": len(rr),
            "complete_median_s": median(r["complete_seconds"] for r in rr),
            "complete_mean_s": statistics.mean(r["complete_seconds"] for r in rr),
            "complete_minmax_s": [min(r["complete_seconds"] for r in rr), max(r["complete_seconds"] for r in rr)],
            "cuda_event_median_ms": median(r["sampler_gpu_ms"] for r in rr),
            "wall_less_than_cuda_event_count": sum(v < 0 for v in residual),
            "wall_minus_cuda_event_ms_minmedianmax": [min(residual), median(residual), max(residual)],
            "torch_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rr),
            "torch_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rr),
            "derived_model_calls": 2 * config["steps"],
            "derived_joint_block_visits": 2 * config["steps"] * (12 + 5 * config["loops"]),
            "derived_core_block_visits": 2 * config["steps"] * 5 * config["loops"],
            "repeat_processes": [],
        }
        for n in range(3):
            repeat = [r for r in rows if r["job"]["phase"] == "repeat" and r["job"]["round"] == n and r["job"]["setting"]["id"] == setting]
            assert len(repeat) == 2
            result["settings"][setting]["repeat_processes"].append({"round": n, "requests": 2, "complete_median_s": median(r["complete_seconds"] for r in repeat)})
    for candidate, reference in [("A_time", "C_time"), ("B_time", "C_time"), ("B_time", "A_time")]:
        ratios = [lookup[p, seed, candidate]["complete_seconds"] / lookup[p, seed, reference]["complete_seconds"] for p, seed in keys]
        diffs = [lookup[p, seed, candidate]["complete_seconds"] - lookup[p, seed, reference]["complete_seconds"] for p, seed in keys]
        result["paired_main_time"][candidate + "_vs_" + reference] = {
            "direction": "candidate/reference time; smaller means faster",
            "median_per_pair_time_ratio": median(ratios),
            "median_per_pair_seconds_delta": median(diffs),
            "candidate_faster_pairs": sum(v < 1 for v in ratios),
            "pairs": len(keys),
            "ratio_of_setting_medians": result["settings"][candidate]["complete_median_s"] / result["settings"][reference]["complete_median_s"],
        }
    result["measurement_caution"] = "45/192 MAIN CUDA event times exceed surrounding CPU wall time. Do not subtract clocks to estimate T5/transfer/PIL overhead. Root cause not diagnosed; preserve both raw measurements."
    assert sum(v["wall_less_than_cuda_event_count"] for v in result["settings"].values()) == 45
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    if a.scalars_only:
        print(json.dumps({"status": result["status"], "main_pairs": result["input_pair_count"]}))
        return

    # Fixed first seed and every MAIN prompt, chosen before looking at images.
    # Contact sheets aid qualitative inspection; they are neither blind nor scored.
    from PIL import Image, ImageDraw, ImageFont
    prompts = json.loads((root / "configs/prompts.json").read_text())["prompts"]
    prompts = [p for p in prompts if p["split"] == "MAIN"]
    protocol = json.loads((root / "configs/protocol.json").read_text())
    seed = min(protocol["data"]["main_seeds"])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    contacts = []
    for category in ["count", "color_binding", "left_right", "compound"]:
        pp = [p for p in prompts if p["category"] == category]
        sheet = Image.new("RGB", (804, 1240), "#f4f0e8")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), category + " | first frozen seed " + str(seed) + " | qualitative only", fill="#251e17", font=font)
        members = []
        for i, p in enumerate(pp):
            y = 46 + i * 297
            draw.text((12, y), p["prompt_id"], fill="#251e17", font=font)
            for j, setting in enumerate(settings):
                r = lookup[p["prompt_id"], seed, setting]
                path = root / next(x["relative_image_path"] for x in index if x["job_id"] == r["job_id"])
                with Image.open(path) as image:
                    sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (12 + 264 * j, y + 28))
                cfg = r["job"]["setting"]
                draw.text((12 + 264 * j, y + 12), setting + " L" + str(cfg["loops"]) + "/S" + str(cfg["steps"]), fill="#66412b", font=small)
                members.append({"prompt_id": p["prompt_id"], "seed": seed, "setting": setting, "image_sha256": r["image_sha256"]})
        path = out / ("visual-" + category + ".png")
        sheet.save(path)
        contacts.append({"file": path.name, "sha256": sha(path), "members": members})
    (out / "visual_manifest.json").write_text(json.dumps({"status": "POST_HOC_QUALITATIVE_NONBLIND_AI_INSPECTION", "selection": "first frozen MAIN seed, all 16 MAIN prompts, three conditions", "quality_annotations_written": 0, "contacts": contacts}, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for setting, marker, color in [("A_time", "o", "#39485a"), ("B_time", "s", "#9d623a"), ("C_time", "^", "#467563")]:
        ss = result["settings"][setting]
        values = [ss["complete_median_s"]] + [r["complete_median_s"] for r in ss["repeat_processes"]]
        ax.plot(range(4), values, marker=marker, color=color, label=f'{setting}: L{ss["loops"]}/S{ss["steps"]}')
    ax.set_xticks(range(4), ["MAIN\n64 requests/arm", "Round 0\n2 requests/arm", "Round 1\n2 requests/arm", "Round 2\n2 requests/arm"])
    ax.set_ylabel("Warmed complete request median (seconds)")
    ax.set_title("Recorded runtime ranking varies across fresh processes")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "runtime-rounds.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ["settings", "paired_main_time", "measurement_caution"]}, indent=2))


if __name__ == "__main__":
    main()
