import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt

# ---------- config ----------
RESULT_GLOBS = [
    "results/*svamp*_gpt-4_1-mini.json",
    "results/*drop*_gpt-4_1-mini.json",
    "results/*gsm8k*_gpt-4_1-mini.json",
    "results/*truthfulqa*_gpt-4_1-mini.json",
    "results/*humaneval*_gpt-4_1-mini.json",
]
OUTDIR = Path("results/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["baseline", "checklist", "conditional_critique", "verifier"]
DATASET_ORDER = ["gsm8k", "truthfulqa", "svamp", "drop", "humaneval"]


# ---------- helpers ----------
def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_rows():
    files = []
    for pat in RESULT_GLOBS:
        files.extend(glob.glob(pat))

    # de-dupe
    files = sorted(set(files))
    if not files:
        raise SystemExit(
            "No result files found. Update RESULT_GLOBS patterns in scripts/make_figures.py "
            "or place your JSONs under results/."
        )

    rows = []
    for fp in files:
        data = json.loads(Path(fp).read_text())
        # some of your files might be "single-task" jsons (task_name is inferable from content)
        results = data.get("results", {})
        for task_name, methods in results.items():
            for method_name, payload in methods.items():
                summary = payload.get("summary", {})
                # unify metric name:
                if task_name == "humaneval":
                    acc = summary.get("pass@1", None)
                else:
                    acc = summary.get("accuracy", None)

                avg_tokens = summary.get("avg_total_tokens", None)

                # If avg_total_tokens missing (e.g. humaneval summary only), compute from rows:
                if avg_tokens is None:
                    rr = payload.get("rows", [])
                    toks = [safe_get(r, "usage", "total_tokens") for r in rr]
                    toks = [t for t in toks if isinstance(t, (int, float))]
                    if toks:
                        avg_tokens = sum(toks) / len(toks)

                rows.append(
                    {
                        "dataset": task_name,
                        "method": method_name,
                        "accuracy": float(acc) if acc is not None else None,
                        "avg_tokens": (
                            float(avg_tokens)
                            if avg_tokens is not None
                            else None
                        ),
                        "source_file": fp,
                    }
                )
    return rows


def pivot(rows):
    # dict[dataset][method] -> (acc, tokens)
    out = {ds: {} for ds in DATASET_ORDER}
    for r in rows:
        ds = r["dataset"]
        m = r["method"]
        if ds not in out:
            out[ds] = {}
        out[ds][m] = (r["accuracy"], r["avg_tokens"])
    return out


# ---------- figure 1: grouped bar accuracy ----------
def fig_accuracy(piv):
    datasets = [d for d in DATASET_ORDER if d in piv and piv[d]]
    methods = METHOD_ORDER

    # build y matrix
    y = []
    for m in methods:
        y.append([piv[d].get(m, (None, None))[0] for d in datasets])

    x = list(range(len(datasets)))
    width = 0.18
    offsets = [(-1.5 + i) * width for i in range(len(methods))]

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(methods):
        vals = [v if v is not None else 0.0 for v in y[i]]
        plt.bar([xi + offsets[i] for xi in x], vals, width=width, label=m)

    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy / pass@1")
    plt.title("Figure 1 — Performance by dataset and method (n=50)")
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    out = OUTDIR / "fig1_accuracy_by_dataset.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


# ---------- figure 2: accuracy vs tokens scatter ----------
def fig_tradeoff(piv):
    import math

    plt.figure(figsize=(9, 5.5))

    # Deterministic encodings
    method_colors = {
        "baseline": "#1f77b4",  # blue
        "checklist": "#ff7f0e",  # orange
        "conditional_critique": "#2ca02c",  # green
        "verifier": "#d62728",  # red
    }
    dataset_markers = {
        "gsm8k": "o",
        "truthfulqa": "s",
        "svamp": "^",
        "drop": "D",
        "humaneval": "P",
    }

    # Collect plotted points for optional labeling
    points = []  # (toks, acc, ds, method)

    # Plot
    for ds in DATASET_ORDER:
        for m in METHOD_ORDER:
            acc, toks = piv.get(ds, {}).get(m, (None, None))
            if acc is None or toks is None:
                continue

            plt.scatter(
                toks,
                acc,
                s=90,
                marker=dataset_markers.get(ds, "o"),
                c=method_colors[m],
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
            )
            points.append((toks, acc, ds, m))

    plt.ylim(0, 1.05)
    plt.xlabel("Avg total tokens")
    plt.ylabel("Accuracy / pass@1")
    plt.title("Figure 2 — Accuracy vs token cost (effort tradeoff)")

    # Method legend (colors)
    method_handles = []
    method_labels = []
    for m in METHOD_ORDER:
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=method_colors[m],
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
        )
        method_handles.append(h)
        method_labels.append(m)
    leg1 = plt.legend(
        method_handles,
        method_labels,
        title="Method (color)",
        loc="lower right",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    plt.gca().add_artist(leg1)

    # Dataset legend (markers)
    dataset_handles = []
    dataset_labels = []
    for ds in DATASET_ORDER:
        h = plt.Line2D(
            [0],
            [0],
            marker=dataset_markers.get(ds, "o"),
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
        )
        dataset_handles.append(h)
        dataset_labels.append(ds)
    plt.legend(
        dataset_handles,
        dataset_labels,
        title="Dataset (marker)",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    out = OUTDIR / "fig2_accuracy_vs_tokens.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    rows = load_rows()
    piv = pivot(rows)
    fig_accuracy(piv)
    fig_tradeoff(piv)


if __name__ == "__main__":
    main()
