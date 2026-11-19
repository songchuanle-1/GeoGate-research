"""Draw ICML Figure 1 from the original 50-scene profiles, without inference."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
FROZEN = HERE / "profiles_method" / "frozen"
STEM = "fig1_three_phase"
MODELS = (("llava3d", "LLaVA-3D", "#215F89", "-", "o"),
          ("3dllava", "3D-LLaVA", "#B35B28", (0, (4.2, 1.8)), "s"))
PANELS = (("Spatial attention", "Attention to spatial tokens (%)", (0, 73), (0, 20, 40, 60)),
          ("Half-mass coverage", "Spatial tokens needed (%)", (0, 50), (0, 15, 30, 45)),
          ("Attention inequality", "Gini of received attention", (.19, .55), (.2, .3, .4, .5)))
INK, MUTED, GUIDE = "#172C3B", "#4D6070", "#73818A"
STAMP = datetime(2026, 11, 19, tzinfo=timezone.utc)


def main():
    source = FROZEN / "attention_profiles.source.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    recorded = json.loads((FROZEN / f"{STEM}.render.json").read_text())["evidence"]["series"]
    assert len(rows) == 96 and len(recorded) == 3
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.spines.left": False, "axes.edgecolor": "#ABB6BC", "axes.linewidth": .65,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "geogate-fig1-icml",
        "legend.frameon": False, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(6.75, 2.48), facecolor="white")
    layer = np.arange(32)
    for j, (letter, panel) in enumerate(zip("abc", PANELS)):
        title, unit, ylim, yticks = panel
        x0 = (.051, .392, .734)[j]
        ax = fig.add_axes([x0, .205, .253, .513])
        fig.text(x0, .957, f"({letter})  {title}", fontsize=10.6, fontweight="bold", va="top")
        fig.text(x0, .857, unit, fontsize=8.4, color=MUTED, va="top")
        ax.set(xlim=(-.45, 31.55), ylim=ylim, xticks=[0, 8, 14, 24, 31], yticks=yticks)
        ax.tick_params(axis="both", labelsize=8, pad=2.5)
        ax.tick_params(axis="x", length=2.3, width=.55)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", lw=.48, color="#DEE4E7", zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(14, lw=.8, color=GUIDE, linestyle=(0, (1.4, 2)), zorder=1.5)
        tick = ax.get_xticklabels()[2]
        tick.set_color(INK)
        tick.set_fontweight("bold")
        group = [r for r in rows if r["panel"] == letter]
        np.testing.assert_array_equal([int(r["layer"]) for r in group], layer)
        for key, _, color, style, marker in MODELS:
            y = np.array([float(row[key]) for row in group])
            np.testing.assert_array_equal(y, recorded[j][key])
            line, = ax.plot(layer, y, color=color, lw=1.55, linestyle=style, zorder=3,
                            solid_capstyle="round", dash_capstyle="butt")
            np.testing.assert_array_equal(line.get_ydata(), y)
            assert min(y) >= ylim[0] and max(y) <= ylim[1]
            # The two Gini values nearly coincide at L14. A larger open circle
            # lets the smaller square remain visible without moving either point.
            is_circle = marker == "o"
            ax.plot(14, y[14], marker=marker, ms=5.2 if is_circle else 3.1,
                    markeredgewidth=1 if is_circle else .45,
                    markerfacecolor="white" if is_circle else color,
                    markeredgecolor=color if is_circle else "white", zorder=4)
    fig.text(.051, .043, "Decoder layer", va="center", fontsize=8.7)
    handles = [Line2D([], [], color=color, lw=1.6, linestyle=style, label=label)
               for _, label, color, style, _ in MODELS]
    handles.append(Line2D([], [], linestyle="none", marker="|", color=GUIDE,
                          markersize=8, markeredgewidth=1, label="L14: pruning layer"))
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.993, .043),
               bbox_transform=fig.transFigure, borderaxespad=0, borderpad=0,
               handlelength=1.8, handletextpad=.55, columnspacing=1.45, ncol=3, fontsize=8.7)
    pdf = HERE.parents[1] / "02_manuscript" / "figures" / f"{STEM}.pdf"
    svg = HERE / "svg" / f"{STEM}.svg"
    preview = HERE / "fig1_preview.png"
    fig.savefig(pdf, metadata={"Title": "Spatial attention profiles", "Creator": "Matplotlib",
                               "CreationDate": STAMP, "ModDate": STAMP})
    fig.savefig(svg, metadata={"Date": STAMP.date().isoformat()})
    fig.savefig(preview, dpi=275)
    plt.close(fig)
    print(f"Verified 192 data values in original layer order. PDF SHA-256: {hashlib.sha256(pdf.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
