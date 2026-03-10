from pathlib import Path

import matplotlib.pyplot as plt

# ---------- config ----------
OUTDIR = Path("results/plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Treat s1_budget_forcing as a normal method
METHOD_ORDER = [
    "baseline",
    "checklist",
    "verifier_reprompt_rm",
    "s1_budget_forcing",
]

DATASET_ORDER = [
    "svamp",
    "drop",
    "humaneval",
    "math500",
]

# Accept hardcoded results now
HARDCODED_RESULTS = {
    "svamp": {
        "baseline": {"accuracy": 0.86, "avg_tokens": 239.18},
        "checklist": {"accuracy": 0.90, "avg_tokens": 347.56},
        "verifier_reprompt_rm": {"accuracy": 0.94, "avg_tokens": 669.66},
        "s1_budget_forcing": {"accuracy": 0.88, "avg_tokens": 686.74},
    },
    "drop": {
        "baseline": {"accuracy": 0.72, "avg_tokens": 391.54},
        "checklist": {"accuracy": 0.44, "avg_tokens": 418.94},
        "verifier_reprompt_rm": {"accuracy": 0.88, "avg_tokens": 535.02},
        "s1_budget_forcing": {"accuracy": 0.56, "avg_tokens": 993.94},
    },
    "humaneval": {
        "baseline": {"accuracy": 0.64, "avg_tokens": 232.06},
        "checklist": {"accuracy": 0.66, "avg_tokens": 274.82},
        "verifier_reprompt_rm": {"accuracy": None, "avg_tokens": None},
        "s1_budget_forcing": {"accuracy": 0.40, "avg_tokens": 1161.20},
    },
    "math500": {
        "baseline": {"accuracy": 0.40, "avg_tokens": 695.66},
        "checklist": {"accuracy": 0.36, "avg_tokens": 836.74},
        "verifier_reprompt_rm": {"accuracy": 0.52, "avg_tokens": 1510.58},
        "s1_budget_forcing": {"accuracy": 0.44, "avg_tokens": 2019.46},
    },
}


# ---------- loaders ----------
def load_rows():
    rows = []
    for dataset, methods in HARDCODED_RESULTS.items():
        for method, vals in methods.items():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "accuracy": vals.get("accuracy"),
                    "avg_tokens": vals.get("avg_tokens"),
                    "source_file": "hardcoded",
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
    plt.figure(figsize=(9, 5.5))

    method_colors = {
        "baseline": "#1f77b4",
        "checklist": "#ff7f0e",
        "verifier_reprompt_rm": "#d62728",
        "s1_budget_forcing": "#9467bd",
    }

    dataset_markers = {
        "svamp": "^",
        "drop": "D",
        "humaneval": "P",
        "math500": "o",
    }

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

    plt.ylim(0, 1.05)
    plt.xlabel("Avg total tokens")
    plt.ylabel("Accuracy / pass@1")
    plt.title("Figure 2 — Accuracy vs token cost (effort tradeoff)")

    # Method legend
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

    # Dataset legend
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
        ncol=len(DATASET_ORDER),
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
