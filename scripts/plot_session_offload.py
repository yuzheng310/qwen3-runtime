#!/usr/bin/env python3
"""Regenerate offload figures from published observations; no model execution.

Bars are medians; points are individual runs, not confidence intervals.
"""
import argparse
import hashlib
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GIB = 1024 ** 3
COLORS = {"B11": "#64748B", "apc-only": "#6366F1", "O-sync": "#0D9488"}

def arm_label(settings):
    if not settings["session_kv"]:
        return "APC only"
    label = "Session KV"
    if settings["apc"]:
        label += " + APC"
    if settings["cpu_offload"] == "sync":
        label += "\n+ CPU offload"
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.source or args.root / "bench/results/session-cpu-offload/observations.json"
    data = json.loads(source.read_text())
    cases = {c["name"]: c for c in data["cases"]}
    out = args.out or args.root / "docs/assets/performance"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#172B4D", "axes.labelcolor": "#475569",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#CBD5E1",
                         "svg.hashsalt": "qwen3-offload-evidence-v1"})

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
        fig.savefig(out / f"{name}.svg", metadata={"Date": None}, facecolor="white")
        svg = out / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    fig.subplots_adjust(left=.08, right=.98, top=.67, bottom=.27, wspace=.56)
    fig.suptitle("CPU KV offload: measure the gain and its boundary", x=.035, y=.96,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(.035, .89, "CodeScout-4B BF16 · one RTX 4090 reporting 49,140 MiB · 24 tasks / 102 turns / 9,871 output tokens", fontsize=11)
    chart_data = {}
    specs = [
        ("default4-finished", ["B11", "O-sync"], "4 clients · 26.40 GiB KV", "No CPU copies; no credible gain"),
        ("fixed24", ["B11", "apc-only", "O-sync"], "24 clients · 26.40 GiB KV", "11.75% less time vs GPU APC"),
        ("automatic24", ["B11", "apc-only", "O-sync"], "24 clients · 33.47 GiB KV", "No CPU copies; no credible gain"),
    ]
    for ax, (name, arms, title, verdict) in zip(axes, specs):
        case = cases[name]
        chart_data[name] = {}
        for i, arm in enumerate(arms):
            rows = [r for r in case["runs"] if r["arm"] == arm]
            warm = [r["elapsed_s"] for r in rows if r["repeat"] >= 0]
            cold = [r["elapsed_s"] for r in rows if r["repeat"] == -1]
            assert len(warm) == 3 and len(cold) == 1
            median = stats.median(warm)
            assert abs(median - case["summary_by_arm"][arm]["median_s"]) < 1e-9
            chart_data[name][arm] = {"warm_s": warm, "first_use_s": cold[0], "median_s": median}
            ax.barh(i, median, color=COLORS[arm], height=.48, alpha=.85)
            ax.scatter(warm, [i-.12, i, i+.12], s=21, c="#172B4D", zorder=3)
            ax.scatter(cold, [i], marker="D", s=46, facecolors="white", edgecolors="#172B4D", zorder=4)
            ax.text(max(warm + cold) + 1.8, i, f"{median:.2f} s", va="center", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(arms)), [arm_label(case["arm_configuration"][a]) for a in arms], fontsize=10)
        ax.set_ylim(len(arms)-.4, -.7)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.set_xlabel("Complete replay time (s) ↓")
        ax.set_title(title, loc="left", pad=16, fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
        ax.text(0, -.33, verdict, transform=ax.transAxes, fontsize=10,
                color="#0F766E" if name == "fixed24" else "#475569", fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color="#64748B", lw=8, label="Bar: median of 3 warm runs"),
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="#172B4D", label="Each warm run"),
                        Line2D([0], [0], marker="D", color="none", markeredgecolor="#172B4D", markerfacecolor="white", label="First-use pass")],
               loc="upper left", bbox_to_anchor=(.035, .845), ncol=3, frameon=False, fontsize=10)
    fig.text(.035, .10, "Fixed tokens; synthetic 1 s tool wait. All completed sessions released. Transfers are included in elapsed time.", fontsize=10)
    fig.text(.035, .055, "4-client case: APC off, legacy parking quotas. 24-client cases: APC on, equal parking quotas, 8 active max. Not full GRPO.", fontsize=9, color="#64748B")
    save(fig, "offload-boundary")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.1))
    fig.subplots_adjust(left=.075, right=.98, top=.69, bottom=.28, wspace=.38)
    fig.suptitle("Engineering gains beyond moving KV to the CPU", x=.035, y=.96,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(.035, .89, "Loading, capacity and session lifecycle · measured separately · gains must not be multiplied", fontsize=11)
    loader = data["loader_fresh_process_runs"]
    measurements = [
        ([[r["after_load"]["peak_allocated"]/GIB for r in loader[a]] for a in ["before", "after"]],
         ["Before", "Target dtype"], "Loading peak ↓", "GPU allocated (GiB)", ".2f", "Fresh processes; n=3 per path"),
        ([[r["factory_blocks"] for r in loader[a]] for a in ["before", "after"]],
         ["Before", "Target dtype"], "Available KV capacity ↑", "16-token GPU blocks", ",.0f", "+26.80% capacity for GPU controls too"),
        ([[r["offload_after_cleanup"]["unused_d2h_bytes"]/GIB for r in cases[c]["runs"] if r["arm"] == "O-sync" and r["repeat"] >= 0]
          for c in ["default4-retained", "default4-finished"]],
         ["Retain finished", "Explicit finish"], "Unused CPU saves ↓", "GiB written per replay", ".2f", "4 clients; zero restores in both cases"),
    ]
    engineering = []
    for ax, (runs, labels, title, unit, fmt, note) in zip(axes, measurements):
        vals = [stats.median(v) for v in runs]
        engineering.append({"metric": title, "labels": labels, "runs": runs, "unit": unit})
        ax.bar([0, 1], vals, color=["#94A3B8", "#0D9488"], width=.48)
        for i, v in enumerate(vals):
            ax.scatter([i-.07, i, i+.07], runs[i], s=20, c="#172B4D", zorder=3)
            ax.text(i, max(runs[i])+max(vals)*.06, format(v, fmt), ha="center", fontweight="bold", fontsize=13)
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(0, max(vals)*1.3)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", pad=13, fontsize=13, fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
        ax.text(0, -.30, note, transform=ax.transAxes, fontsize=9, color="#475569")
    fig.text(.035, .105, "Bars: medians; dots: individual runs. Loader equivalence: 8,044,936,192 weight bytes + 64 MiB RoPE buffers, exact.", fontsize=10)
    fig.text(.035, .055, "Lower loading peak is not faster loading. Avoided writes are lifecycle cleanup, not CPU-cache hits. Same 49,140 MiB GPU.", fontsize=9, color="#64748B")
    save(fig, "offload-engineering")
    (out / "offload-figure-data.json").write_text(json.dumps({
        "source": "bench/results/session-cpu-offload/observations.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boundary": chart_data, "engineering": engineering,
        "arm_configuration": {name: cases[name]["arm_configuration"] for name in chart_data},
        "note": "Bar medians and individual samples; no confidence interval or cross-engine speed claim."
    }, indent=2) + "\n")
    print("Generated two offload figures and their numeric provenance")


if __name__ == "__main__":
    main()
