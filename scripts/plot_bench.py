"""Plots for item 27. usage: uv run python scripts/plot_bench.py [directory with the CSVs]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
STYLE = {"ours": dict(marker="o", color="C0"), "tenpy": dict(marker="s", color="C1")}


def load(name):
    p = d / name
    return (
        np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None)
        if p.exists()
        else None
    )


def plain_x(ax):
    """Log-2 x axis with plain tick labels (chi and N are not decades)."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())


def guide(ax, x, y, power, label):
    """Dashed line y ~ x^power through the last point."""
    x = np.asarray(x, dtype=float)
    ax.loglog(x, y[-1] * (x / x[-1]) ** power, "k--", alpha=0.4, label=label)


def runtime_plot(csv, xcol, power, xlabel, title, out):
    t = load(csv)
    if t is None:
        print(f"skip {csv}")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for b in ("ours", "tenpy"):
        m = t["backend"] == b
        ax.loglog(t[xcol][m], t["seconds"][m], label=b, **STYLE[b])
    m = t["backend"] == "ours"
    guide(ax, t[xcol][m], t["seconds"][m], power, f"~ {xcol}^{power}")
    plain_x(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("wall time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


runtime_plot(
    "bench_runtime_n.csv",
    "N",
    1,
    "N (sites)",
    "Runtime vs N (fixed chi)",
    "bench_runtime_n.png",
)
runtime_plot(
    "bench_runtime_chi.csv",
    "chi",
    3,
    "chi_max",
    "Runtime vs chi (fixed N)",
    "bench_runtime_chi.png",
)

e = load("bench_error_chi.csv")
if e is not None:
    floor = 1e-16
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [
        (axes[0], "infid", "vs exact (includes Trotter floor)"),
        (axes[1], "infid_trunc", "truncation only (vs same code, no cap)"),
    ]
    for ax, col, title in panels:
        for b in ("ours", "tenpy"):
            m = e["backend"] == b
            ax.loglog(
                e["chi"][m],
                np.maximum(e[col][m], floor),
                label=f"{b}: true",
                **STYLE[b],
            )
            ax.loglog(
                e["chi"][m],
                np.maximum(e["est_infid"][m], floor),
                ls="--",
                alpha=0.6,
                label=f"{b}: estimate",
                color=STYLE[b]["color"],
            )
        plain_x(ax)
        ax.set_xlabel("chi_max")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("infidelity")
    axes[0].legend(fontsize=8)
    fig.savefig("bench_error_chi.png", dpi=150, bbox_inches="tight")
    print("wrote bench_error_chi.png")
