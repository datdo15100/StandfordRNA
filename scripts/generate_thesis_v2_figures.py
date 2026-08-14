"""Generate the vector figures used by the bilingual thesis.

The script contains only frozen aggregate values reported by the repository's
confirmatory analyses. It does not read native structures or rerun prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "thesis_v2" / "figures"

NAVY = "#17324d"
BLUE = "#2d6cdf"
TEAL = "#159a9c"
ORANGE = "#e8871e"
RED = "#c84630"
GREEN = "#3a8f5c"
GRAY = "#687783"
LIGHT = "#eef3f7"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def box(ax, xy, width, height, text, color, subtitle=None) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.59, text, ha="center", va="center", color=color, weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.29, subtitle, ha="center", va="center", color=GRAY, fontsize=7.4)


def arrow(ax, start, end, color=GRAY) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color))


def pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    box(ax, (0.15, 1.9), 1.45, 1.15, "RNA sequence", NAVY, "target and temporal cutoff")
    box(ax, (2.15, 3.15), 2.05, 1.15, "Time-safe TBM", BLUE, "MMseqs2 + composite search")
    box(ax, (2.15, 0.7), 2.05, 1.15, "DRfold2", TEAL, "independent deep models")
    box(ax, (4.9, 1.9), 1.75, 1.15, "Candidate bank", NAVY, "3 TBM + 2 DRfold2")
    box(ax, (7.25, 1.9), 1.75, 1.15, "Geometry\nrefinement", ORANGE, "candidate-wise, 300 steps")
    box(ax, (9.55, 1.9), 1.25, 1.15, "Five\nstructures", GREEN, "C1' coordinates")
    arrow(ax, (1.6, 2.48), (2.15, 3.55), BLUE)
    arrow(ax, (1.6, 2.48), (2.15, 1.25), TEAL)
    arrow(ax, (4.2, 3.72), (4.9, 2.63), BLUE)
    arrow(ax, (4.2, 1.28), (4.9, 2.33), TEAL)
    arrow(ax, (6.65, 2.48), (7.25, 2.48), ORANGE)
    arrow(ax, (9.0, 2.48), (9.55, 2.48), GREEN)
    ax.text(3.18, 4.63, "Template release date < target cutoff", ha="center", color=RED, fontsize=8.2)
    ax.text(5.78, 1.42, "source diversity", ha="center", color=GRAY, fontsize=8)
    ax.text(8.13, 1.42, "local correction\nglobal restraint", ha="center", color=GRAY, fontsize=8)
    save(fig, "pipeline_overview")


def evaluation_protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.plot([0.8, 9.7], [2.15, 2.15], color=NAVY, linewidth=2)
    specs = [
        (1.45, "Train", "60 RNA", "2024-01-10 to\n2024-12-04", BLUE),
        (4.05, "Calibration", "20 RNA", "2024-12-11 to\n2025-01-08", ORANGE),
        (6.65, "Validation", "20 RNA", "2025-01-15 to\n2025-03-26", GREEN),
        (9.15, "External", "Kaggle", "hidden public and\nprivate labels", TEAL),
    ]
    for x, title, n, date, color in specs:
        ax.scatter([x], [2.15], s=170, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(x, 3.15, title, ha="center", color=color, weight="bold", fontsize=10)
        ax.text(x, 2.77, n, ha="center", color=NAVY, weight="bold")
        ax.text(x, 1.42, date, ha="center", va="top", color=GRAY, fontsize=7.8)
    ax.text(1.45, 0.54, "estimate priors", ha="center", color=BLUE)
    ax.text(4.05, 0.54, "select and freeze", ha="center", color=ORANGE)
    ax.text(6.65, 0.54, "open once", ha="center", color=GREEN)
    ax.text(9.15, 0.54, "pipeline-level check", ha="center", color=TEAL)
    ax.text(4.05, 3.78, "No sequence group or family group crosses a local split", ha="center", color=RED, fontsize=8.5)
    save(fig, "evaluation_protocol")


def tbm_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    box(ax, (0.2, 2.0), 1.35, 1.0, "Query RNA", NAVY, "sequence + cutoff")
    box(ax, (2.0, 3.45), 1.8, 1.0, "MMseqs2", BLUE, "fast homolog retrieval")
    box(ax, (2.0, 0.55), 1.8, 1.0, "Composite", TEAL, "G, L, F, K3 signals")
    box(ax, (4.35, 2.0), 1.75, 1.0, "Safety gate", RED, "date + self-PDB filter")
    box(ax, (6.55, 2.0), 1.6, 1.0, "Realign\nand rank", ORANGE, "identity × coverage")
    box(ax, (8.6, 2.0), 1.65, 1.0, "Transfer C1'", GREEN, "gap fill + confidence")
    arrow(ax, (1.55, 2.5), (2.0, 3.95), BLUE)
    arrow(ax, (1.55, 2.5), (2.0, 1.05), TEAL)
    arrow(ax, (3.8, 3.95), (4.35, 2.72), RED)
    arrow(ax, (3.8, 1.05), (4.35, 2.28), RED)
    arrow(ax, (6.1, 2.5), (6.55, 2.5), ORANGE)
    arrow(ax, (8.15, 2.5), (8.6, 2.5), GREEN)
    ax.text(2.9, 4.88, "high precision when a close homolog is found", ha="center", color=BLUE, fontsize=8)
    ax.text(2.9, 0.13, "broad recall when fast retrieval returns no hit", ha="center", color=TEAL, fontsize=8)
    save(fig, "tbm_flow")


def refinement_mechanism() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    t = np.linspace(0, 4.5 * np.pi, 64)
    base_x = 0.28 * t * np.cos(0.42 * t)
    base_y = 0.28 * t * np.sin(0.42 * t)
    perturb = np.zeros_like(base_x)
    perturb[24:34] = np.array([0.0, 0.2, 0.7, 1.25, 1.65, 1.9, 1.45, 0.8, 0.25, 0.0])
    raw_x = base_x + perturb
    raw_y = base_y - 0.25 * perturb
    refined_x = base_x + 0.22 * perturb
    refined_y = base_y - 0.05 * perturb
    ax.plot(raw_x, raw_y, color=RED, linewidth=2.0, alpha=0.85, label="raw candidate")
    ax.plot(refined_x, refined_y, color=BLUE, linewidth=2.2, label="refined candidate")
    ax.scatter(raw_x[::6], raw_y[::6], color=RED, s=17)
    ax.scatter(refined_x[::6], refined_y[::6], color=BLUE, s=17)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Conservative correction of a local distortion", color=NAVY, weight="bold")

    ax = axes[1]
    ax.axis("off")
    terms = [
        ("Source restraint", "preserve the global fold", NAVY),
        ("Backbone prior", "regularize adjacent C1' distances", BLUE),
        ("Angle and torsion priors", "discourage unlikely local geometry", TEAL),
        ("Clash and kink terms", "remove steric and trace artifacts", ORANGE),
        ("Radius-of-gyration term", "limit global expansion or collapse", GREEN),
    ]
    y = 0.92
    for title, subtitle, color in terms:
        ax.add_patch(FancyBboxPatch((0.03, y - 0.12), 0.94, 0.13, boxstyle="round,pad=0.02", facecolor=LIGHT, edgecolor=color, linewidth=1.1))
        ax.text(0.07, y - 0.045, title, transform=ax.transAxes, color=color, weight="bold", va="center")
        ax.text(0.56, y - 0.045, subtitle, transform=ax.transAxes, color=GRAY, va="center", fontsize=7.4)
        y -= 0.18
    ax.set_title("Geometry refinement objective", color=NAVY, weight="bold")
    save(fig, "refinement_mechanism")


def complementarity_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    labels = ["3 TBM", "2 DRfold2", "Union\n3T + 2D"]
    values = [0.565183, 0.502627, 0.588304]
    colors = [BLUE, TEAL, GREEN]
    axes[0].bar(labels, values, color=colors, width=0.62)
    axes[0].set_ylim(0.45, 0.62)
    axes[0].set_ylabel("Mean best-available TM-score")
    axes[0].set_title("Candidate-bank complementarity", color=NAVY, weight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.006, f"{value:.3f}", ha="center", weight="bold", color=colors[i])
    axes[0].annotate("+0.023\n95% CI [0.006, 0.048]", xy=(2, values[2]), xytext=(1.28, 0.612), arrowprops={"arrowstyle": "->", "color": GREEN}, color=GREEN, ha="center", fontsize=8)

    axes[1].bar(["TBM wins", "DRfold2 wins"], [12, 8], color=[BLUE, TEAL], width=0.62)
    axes[1].set_ylim(0, 14)
    axes[1].set_ylabel("Held-out RNA targets")
    axes[1].set_title("Best source by target", color=NAVY, weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    for i, value in enumerate([12, 8]):
        axes[1].text(i, value + 0.4, str(value), ha="center", weight="bold")
    fig.tight_layout()
    save(fig, "complementarity_results")


def geometry_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    metrics = ["TM-score", "C1'-lDDT"]
    delta = [-0.001107, 0.006660]
    low = [-0.002809, 0.000933]
    high = [0.000418, 0.013093]
    y = np.arange(2)
    err = np.array([np.array(delta) - np.array(low), np.array(high) - np.array(delta)])
    axes[0].errorbar(delta, y, xerr=err, fmt="o", color=NAVY, ecolor=BLUE, capsize=4, markersize=7)
    axes[0].axvline(0, color=GRAY, linewidth=1)
    axes[0].set_yticks(y, metrics)
    axes[0].set_xlabel("Paired mean change after refinement")
    axes[0].set_title("Score change with 95% CI", color=NAVY, weight="bold")
    axes[0].grid(axis="x", alpha=0.2)

    metrics = ["SW-RMSD9", "SW-RMSD15"]
    gain = [0.040386, 0.017257]
    low = [0.029354, 0.000617]
    high = [0.052090, 0.031336]
    y = np.arange(2)
    err = np.array([np.array(gain) - np.array(low), np.array(high) - np.array(gain)])
    axes[1].errorbar(gain, y, xerr=err, fmt="o", color=ORANGE, ecolor=ORANGE, capsize=4, markersize=7)
    axes[1].axvline(0, color=GRAY, linewidth=1)
    axes[1].set_yticks(y, metrics)
    axes[1].set_xlabel("RMSD reduction in angstrom")
    axes[1].set_title("Local-window improvement with 95% CI", color=NAVY, weight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, "geometry_results")


def leaderboard_results() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    methods = ["Reference\nTBM-only", "Our time-safe\nTBM", "Our hybrid +\ngeometry refinement"]
    public = [np.nan, 0.60084, 0.62809]
    private = [0.59298, 0.60175, 0.61390]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, np.nan_to_num(public), width, color=TEAL, label="Public")
    ax.bar(x + width / 2, private, width, color=NAVY, label="Private")
    ax.set_ylim(0.56, 0.64)
    ax.set_ylabel("Leaderboard TM-score")
    ax.set_xticks(x, methods)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("External hidden-set evaluation", color=NAVY, weight="bold")
    for i, value in enumerate(private):
        ax.text(i + width / 2, value + 0.002, f"{value:.5f}", ha="center", fontsize=8, color=NAVY, weight="bold")
    for i, value in enumerate(public):
        if np.isfinite(value):
            ax.text(i - width / 2, value + 0.002, f"{value:.5f}", ha="center", fontsize=8, color=TEAL, weight="bold")
    ax.annotate("+0.01215 private", xy=(2.17, 0.6139), xytext=(1.42, 0.632), arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=8)
    save(fig, "leaderboard_results")


def leakage_diagnostic() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["Temporal-safe", "No temporal filter", "Oracle leak"]
    values = [0.1612, 0.6388, 0.9566]
    colors = [GREEN, ORANGE, RED]
    ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean best-of-five TM-score")
    ax.set_title("Illustration of evaluation inflation from template leakage", color=NAVY, weight="bold")
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(i, value + 0.025, f"{value:.4f}", ha="center", weight="bold", color=colors[i])
    ax.text(1.5, 0.08, "Unsafe variants are diagnostics, not valid methods", ha="center", color=RED, fontsize=8.5)
    save(fig, "leakage_diagnostic")


def placeholder_structure_overlay() -> None:
    fig, ax = plt.subplots(figsize=(9.3, 4.7))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.82, transform=ax.transAxes, boxstyle="round,pad=0.02", facecolor="#fbfcfd", edgecolor=GRAY, linestyle=":"))
    ax.text(0.5, 0.68, "STRUCTURE OVERLAY PLACEHOLDER", transform=ax.transAxes, ha="center", color=NAVY, weight="bold", fontsize=13)
    ax.text(
        0.5,
        0.46,
        "Insert a native, raw-prediction, and refined C1' trace after selecting\n"
        "one representative held-out RNA. Superpose all traces with the same\n"
        "Kabsch transform, mark the corrected local window, and report the\n"
        "target ID, candidate source, TM-score, C1'-lDDT, and SW-RMSD9.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        linespacing=1.5,
    )
    ax.text(0.5, 0.18, "Do not use a hidden Kaggle target because its native structure is unavailable.", transform=ax.transAxes, ha="center", color=RED, fontsize=8.5)
    save(fig, "structure_overlay_placeholder")


def main() -> None:
    setup()
    pipeline_overview()
    evaluation_protocol()
    tbm_flow()
    refinement_mechanism()
    complementarity_results()
    geometry_results()
    leaderboard_results()
    leakage_diagnostic()
    placeholder_structure_overlay()


if __name__ == "__main__":
    main()
