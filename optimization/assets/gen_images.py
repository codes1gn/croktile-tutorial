#!/usr/bin/env python3
"""Generate tutorial diagram PNGs for the Dense GEMM FP16 tutorial.

Each figure is rendered twice — once for dark mode and once for light mode —
so that MkDocs Material can swap them via  #only-dark / #only-light.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)

# ── Theme palettes ───────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":       "#1e1e2e",
        "box_bg":   "#2d2d44",
        "box_bg2":  "#1a3a1a",
        "box_bg3":  "#3d2d1a",
        "box_bg4":  "#2d1a1a",
        "text":     "#e4e4e7",
        "gray":     "#adb5bd",
        "green":    "#40c057",
        "red":      "#ff6b6b",
        "blue":     "#74c0fc",
        "yellow":   "#ffd43b",
        "purple":   "#b197fc",
        "orange":   "#ffa94d",
        "green2":   "#2d8a2d",
        "suffix":   "",
    },
    "light": {
        "bg":       "#ffffff",
        "box_bg":   "#e8eaf0",
        "box_bg2":  "#dff0df",
        "box_bg3":  "#f5ead6",
        "box_bg4":  "#f5d6d6",
        "text":     "#1a1a2e",
        "gray":     "#6c757d",
        "green":    "#2b8a3e",
        "red":      "#c92a2a",
        "blue":     "#1971c2",
        "yellow":   "#e67700",
        "purple":   "#7048e8",
        "orange":   "#d9480f",
        "green2":   "#1a6b1a",
        "suffix":   "_light",
    },
}


def _apply_theme(t):
    plt.rcParams.update({
        'figure.facecolor': t["bg"],
        'axes.facecolor':   t["bg"],
        'text.color':       t["text"],
        'axes.labelcolor':  t["text"],
        'xtick.color':      t["gray"],
        'ytick.color':      t["gray"],
        'font.family':      'DejaVu Sans',
        'font.size':        11,
    })


def _alpha_hex(hex_color, alpha_hex="20"):
    """Append alpha to a hex color for RGBA fill."""
    return hex_color + alpha_hex


def fig1_v0_memory_access(t):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("v0: Memory Access Pattern — 4 Threads",
                 fontsize=14, fontweight='bold', pad=15, color=t["text"])

    a_x, a_y, a_w, a_h = 0.3, 1.0, 3.5, 3.5
    ax.add_patch(mpatches.FancyBboxPatch(
        (a_x, a_y), a_w, a_h, boxstyle="round,pad=0.1",
        facecolor=t["box_bg"], edgecolor=t["blue"], linewidth=1.5))
    ax.text(a_x + a_w/2, a_y + a_h + 0.25, "A  [M × K]",
            ha='center', fontsize=12, fontweight='bold', color=t["blue"])
    for i, (label, color) in enumerate([
        ("row 0", t["green"]), ("row 1", t["orange"]),
        ("row 0", t["green"]), ("row 1", t["orange"]),
    ]):
        y = a_y + a_h - 0.5 - i * 0.8
        ax.annotate('', xy=(a_x + a_w - 0.1, y), xytext=(a_x + 0.2, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(a_x + 0.25, y + 0.15, label, fontsize=9, color=color)

    b_x, b_y, b_w, b_h = 5.8, 1.0, 3.5, 3.5
    ax.add_patch(mpatches.FancyBboxPatch(
        (b_x, b_y), b_w, b_h, boxstyle="round,pad=0.1",
        facecolor=t["box_bg"], edgecolor=t["purple"], linewidth=1.5))
    ax.text(b_x + b_w/2, b_y + b_h + 0.25,
            "B  [N × K]  (row = col of B^T)",
            ha='center', fontsize=12, fontweight='bold', color=t["purple"])
    labels_b = [
        ("col 0", t["green"]),  ("col 0", t["green"]),
        ("col 1", t["orange"]), ("col 1", t["orange"]),
    ]
    threads = ["thread(0,0)", "thread(1,0)",
               "thread(0,1)", "thread(1,1)"]
    for i, ((label, color), thr) in enumerate(zip(labels_b, threads)):
        y = b_y + b_h - 0.5 - i * 0.8
        ax.annotate('', xy=(b_x + b_w - 0.1, y), xytext=(b_x + 0.2, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(b_x + 0.25, y + 0.15, label, fontsize=9, color=color)
        ax.text(b_x + b_w + 0.15, y, thr,
                fontsize=8, color=t["gray"], va='center')

    for i in range(4):
        y = a_y + a_h - 0.5 - i * 0.8
        ax.annotate('', xy=(b_x, y), xytext=(a_x + a_w + 0.05, y),
                    arrowprops=dict(arrowstyle='->', color=t["red"],
                                    lw=1.5, linestyle='dotted'))

    ax.text(5, 0.55,
            "Each thread reads:  row_m of A + col_n of B"
            " = 2 × 8192 × 2B = 32 KB from HBM",
            ha='center', fontsize=10, color=t["yellow"])
    ax.text(5, 0.15,
            "Same rows/cols re-read redundantly."
            " Total traffic ≈ 2 TB  (minimum ≈ 400 MB)",
            ha='center', fontsize=9, color=t["gray"])

    fig.tight_layout()
    fig.savefig(OUT / f"v0_memory_access{t['suffix']}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig2_v1_tile_reuse(t):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("v1: Tile Reuse — Sliding Window Along K",
                 fontsize=14, fontweight='bold', pad=15, color=t["text"])

    stages = ["iv_k=0", "iv_k=1", "...", "iv_k=63"]
    colors = [t["blue"], t["purple"], t["gray"], t["green"]]
    for i, (label, color) in enumerate(zip(stages, colors)):
        x = 0.5 + i * 2.4
        if label == "...":
            ax.text(x + 0.7, 3.0, "⋯", fontsize=24,
                    ha='center', va='center', color=t["gray"])
            continue
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 3.5), 1.5, 1.0, boxstyle="round,pad=0.05",
            facecolor=t["box_bg"], edgecolor=color, linewidth=1.5))
        ax.text(x + 0.75, 4.05, "HBM tile",
                ha='center', fontsize=8, color=color)
        ax.text(x + 0.75, 3.7, label,
                ha='center', fontsize=9, fontweight='bold', color=t["text"])
        ax.annotate('', xy=(x + 0.75, 2.8), xytext=(x + 0.75, 3.45),
                    arrowprops=dict(arrowstyle='->', color=t["yellow"], lw=2))
        ax.text(x + 0.95, 3.1, "dma",
                fontsize=8, color=t["yellow"], style='italic')
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1.8), 1.5, 1.0, boxstyle="round,pad=0.05",
            facecolor=t["box_bg2"], edgecolor=t["green"], linewidth=1.5))
        ax.text(x + 0.75, 2.35, "SRAM tile",
                ha='center', fontsize=8, color=t["green"])
        ax.text(x + 0.75, 2.0, "[32×128]",
                ha='center', fontsize=9, fontweight='bold', color=t["text"])
        ax.annotate('', xy=(x + 0.75, 1.1), xytext=(x + 0.75, 1.75),
                    arrowprops=dict(arrowstyle='->', color=t["orange"], lw=2))

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, 0.3), 9.0, 0.7, boxstyle="round,pad=0.05",
        facecolor=t["box_bg3"], edgecolor=t["orange"], linewidth=1.5))
    ax.text(5.0, 0.65,
            "acc += lhs_s × rhs_s    (1024 threads, scalar FMA)",
            ha='center', fontsize=11, color=t["orange"])
    ax.text(5.0, 1.3,
            "Each 8 KB tile loaded once, read 32× by threads"
            " in the block  →  reuse factor = 32",
            ha='center', fontsize=9, color=t["gray"])

    fig.tight_layout()
    fig.savefig(OUT / f"v1_tile_reuse{t['suffix']}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig3_double_buffering(t):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("Double-Buffering (STAGES=2): TMA Latency Fully Hidden",
                 fontsize=14, fontweight='bold', pad=12, color=t["text"])

    ax.annotate('', xy=(10.5, 4.3), xytext=(0.5, 4.3),
                arrowprops=dict(arrowstyle='->', color=t["gray"], lw=1.5))
    ax.text(10.6, 4.3, "time", fontsize=10, color=t["gray"], va='center')

    ax.text(0.15, 3.5, "Producer\n(wg=0)",
            fontsize=9, color=t["blue"], fontweight='bold', va='center')
    prod_tiles = [
        (1.5, "tile[0]\nslot 0", t["blue"]),
        (3.5, "tile[1]\nslot 1", t["purple"]),
        (5.5, "tile[2]\nslot 0", t["blue"]),
        (7.5, "tile[3]\nslot 1", t["purple"]),
    ]
    for x, label, color in prod_tiles:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 3.1), 1.6, 0.8, boxstyle="round,pad=0.05",
            facecolor=_alpha_hex(color), edgecolor=color, linewidth=2))
        ax.text(x + 0.8, 3.5, label,
                ha='center', va='center', fontsize=8, color=t["text"])

    ax.text(0.15, 1.8, "Consumer\n(wg=1,2)",
            fontsize=9, color=t["green"], fontweight='bold', va='center')
    cons_tiles = [
        (2.5, "WGMMA\ntile[0]\nslot 0", t["green"]),
        (4.5, "WGMMA\ntile[1]\nslot 1", t["green2"]),
        (6.5, "WGMMA\ntile[2]\nslot 0", t["green"]),
        (8.5, "WGMMA\ntile[3]\nslot 1", t["green2"]),
    ]
    for x, label, color in cons_tiles:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1.3), 1.6, 1.0, boxstyle="round,pad=0.05",
            facecolor=_alpha_hex(color), edgecolor=color, linewidth=2))
        ax.text(x + 0.8, 1.8, label,
                ha='center', va='center', fontsize=7, color=t["text"])

    ax.annotate('', xy=(4.0, 2.6), xytext=(2.0, 2.6),
                arrowprops=dict(arrowstyle='<->', color=t["yellow"], lw=2))
    ax.text(3.0, 2.75, "overlap",
            fontsize=9, color=t["yellow"], ha='center', fontweight='bold')

    for px, cx in [(2.3, 2.5), (4.3, 4.5), (6.3, 6.5)]:
        ax.annotate('', xy=(cx + 0.3, 2.35), xytext=(px + 0.5, 3.05),
                    arrowprops=dict(arrowstyle='->', color=t["yellow"],
                                    lw=1, linestyle='--'))
    ax.text(1.5, 2.5, "full[0]↓", fontsize=7, color=t["yellow"])

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, 0.05), 10.0, 0.8, boxstyle="round,pad=0.05",
        facecolor=t["box_bg4"], edgecolor=t["red"], linewidth=1))
    ax.text(5.5, 0.45,
            "STAGES=1 (v3):  no overlap — producer must wait"
            " for consumer to finish → TMA latency exposed",
            ha='center', fontsize=9, color=t["red"])

    fig.tight_layout()
    fig.savefig(OUT / f"double_buffering_timeline{t['suffix']}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig4_warpn_sweep(t):
    fig, ax = plt.subplots(figsize=(9, 5))

    warp_ns = [128, 136, 144, 152, 160, 168, 176, 192, 208, 224]
    tflops  = [365, 386, 434, 402, 457, 457, 476, 466, None, None]
    colors_list = []
    for wn, tf in zip(warp_ns, tflops):
        if tf is None:
            colors_list.append(t["red"])
        elif wn == 192:
            colors_list.append(t["green"])
        else:
            colors_list.append(t["blue"])

    valid_ns = [wn for wn, tf in zip(warp_ns, tflops) if tf is not None]
    valid_tf = [tf for tf in tflops if tf is not None]
    valid_colors = [c for c, tf in zip(colors_list, tflops)
                    if tf is not None]

    bars = ax.bar([str(n) for n in valid_ns], valid_tf,
                  color=valid_colors, width=0.7, edgecolor='none')

    ax.axhline(y=288, color=t["orange"], linestyle='--', lw=1.5, alpha=0.7)
    ax.text(0.02, 291, "v3 baseline (288)", fontsize=8,
            color=t["orange"], transform=ax.get_yaxis_transform())

    ax.axhline(y=447.5, color=t["yellow"], linestyle='--', lw=1.5, alpha=0.7)
    ax.text(0.02, 450, "cuBLAS (447)", fontsize=8,
            color=t["yellow"], transform=ax.get_yaxis_transform())

    for bar, tf in zip(bars, valid_tf):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 5, str(tf),
                ha='center', fontsize=9, color=t["text"], fontweight='bold')

    winner_idx = valid_ns.index(192)
    ax.text(bars[winner_idx].get_x() + bars[winner_idx].get_width()/2,
            valid_tf[winner_idx] + 18, "← winner (v4)",
            fontsize=10, color=t["green"], fontweight='bold', ha='center')

    fail_x_positions = ["208", "224"]
    for fx in fail_x_positions:
        ax.text(len(valid_ns) - 0.5 + fail_x_positions.index(fx) * 0.6,
                350, f"WN={fx}\n✗ FAIL",
                fontsize=8, color=t["red"], ha='center', fontweight='bold')

    ax.set_xlabel("WARP_N", fontsize=12, fontweight='bold')
    ax.set_ylabel("TFLOPS", fontsize=12, fontweight='bold')
    ax.set_title("WARP_N Sweep (STAGES=2, CONSUMERS=2, TILE_K=64)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(300, 520)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(t["gray"])
    ax.spines['bottom'].set_color(t["gray"])

    fig.tight_layout()
    fig.savefig(OUT / f"warpn_sweep{t['suffix']}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig5_optimization_ladder(t):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = ["v0\nnaive", "v1\nSMEM", "v2\nTMA+WGMMA",
              "v3\nwarpspec", "v4\ntuned", "cuBLAS"]
    values = [0.38, 1.51, 284.4, 288.3, 489.9, 447.5]
    colors_l = [t["red"], t["orange"], t["blue"],
                t["purple"], t["green"], t["yellow"]]

    bars = ax.barh(range(len(labels)), values,
                   color=colors_l, height=0.65, edgecolor='none')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel("TFLOPS (H800 PCIe, 8192³, f16)",
                  fontsize=11, fontweight='bold')
    ax.set_title("The Optimization Ladder",
                 fontsize=14, fontweight='bold', pad=12)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 5 if val > 50 else bar.get_width() + 2
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.1f}",
                va='center', fontsize=10, color=t["text"], fontweight='bold')

    speedups = ["", "3.9×", "188×", "1.01×", "1.70×", ""]
    for i, s in enumerate(speedups):
        if s:
            ax.text(max(values[i], values[i-1]) + 40, i, s,
                    va='center', fontsize=9, color=t["yellow"],
                    fontweight='bold')

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(t["gray"])
    ax.spines['bottom'].set_color(t["gray"])
    ax.set_xlim(0, 550)

    fig.tight_layout()
    fig.savefig(OUT / f"optimization_ladder{t['suffix']}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


ALL_FIGS = [
    ("v0_memory_access",          fig1_v0_memory_access),
    ("v1_tile_reuse",             fig2_v1_tile_reuse),
    ("double_buffering_timeline", fig3_double_buffering),
    ("warpn_sweep",               fig4_warpn_sweep),
    ("optimization_ladder",       fig5_optimization_ladder),
]

if __name__ == "__main__":
    for theme_name, theme in THEMES.items():
        _apply_theme(theme)
        for base_name, fn in ALL_FIGS:
            fn(theme)
            suffix = theme["suffix"]
            print(f"  ✓ {base_name}{suffix}.png")
        print(f"[{theme_name}] done")

    print(f"\nAll images saved to {OUT}")
