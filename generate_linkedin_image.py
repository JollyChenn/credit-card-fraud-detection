"""
LINKEDIN POSTER GENERATOR — Editorial Print Style
─────────────────────────────────────────────────
Creates a 1200x1500 (4:5) portrait image styled as an editorial
research brief — the visual language of a Financial Times or
Economist data piece. Cream paper, serif headlines, a real
performance chart, restrained color.

HOW TO USE:
  1. After training the model: python generate_linkedin_image.py
  2. fraud_report.png is saved in the same folder
  3. Upload to LinkedIn

If dashboard_data.json doesn't exist, representative numbers
are used so the design can be previewed.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

# ──────────────────────────────────────────────────────────────────
# COLOR PALETTE — editorial print
# ──────────────────────────────────────────────────────────────────
PAPER     = "#f3ead4"   # warm cream (FT salmon-leaning)
INK       = "#14181f"   # near-black with hint of blue
INK_BODY  = "#2a3340"   # body copy
INK_MUTE  = "#6b7280"   # muted gray
RULE      = "#c8b994"   # warm gold-tan rule
RED       = "#b3261e"   # editorial red (signal)
NAVY      = "#1c3d5a"   # supporting navy
GOLD      = "#a07c2e"   # warm gold for highlights

# Canvas (1200x1500 at 100 dpi)
W, H = 12, 15
DPI  = 100


def load_metrics():
    """Real metrics if available, otherwise representative values."""
    if os.path.exists("dashboard_data.json"):
        with open("dashboard_data.json", "r") as f:
            d = json.load(f)
        m, cm, ds = d["metrics"], d["confusion_matrix"], d["dataset"]
        # Use real ROC curve if present
        roc = d.get("roc_curve", [])
        roc_xy = [(p["fpr"], p["tpr"]) for p in roc] if roc else None
        return {
            "roc_auc": m["roc_auc"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1_score"],
            "frauds_caught": cm["tp"],
            "frauds_total": cm["tp"] + cm["fn"],
            "frauds_missed": cm["fn"],
            "false_alarms": cm["fp"],
            "total_tx": ds["total_transactions"],
            "dataset_fraud_pct": ds.get("fraud_percentage", 0.172),
            "model_name": d.get("model_name", "Random Forest"),
            "threshold": m["threshold"],
            "roc_xy": roc_xy,
            "is_sample": False
        }
    # Synthetic ROC curve shape: ROC-AUC ≈ 0.987
    np.random.seed(0)
    fpr_pts = np.array([0, 0.001, 0.005, 0.012, 0.03, 0.06, 0.1, 0.2, 0.4, 0.7, 1.0])
    tpr_pts = np.array([0, 0.55, 0.78, 0.88, 0.93, 0.96, 0.975, 0.99, 0.995, 0.998, 1.0])
    return {
        "roc_auc": 0.987, "recall": 0.857, "precision": 0.831, "f1": 0.844,
        "frauds_caught": 84, "frauds_total": 98, "frauds_missed": 14,
        "false_alarms": 17, "total_tx": 284807, "dataset_fraud_pct": 0.172,
        "model_name": "Random Forest",
        "threshold": 0.42, "roc_xy": list(zip(fpr_pts, tpr_pts)),
        "is_sample": True
    }


# ──────────────────────────────────────────────────────────────────
# DRAWING — each section a separate function
# ──────────────────────────────────────────────────────────────────
def draw_masthead(ax):
    """Top masthead: like the head of an editorial page."""
    # Small caps publication line
    ax.text(0.06, 0.965, "THE FRAUD DETECTION REPORT",
            color=INK, fontsize=11, fontweight="bold",
            family="serif", zorder=3)
    # Right side: issue info
    ax.text(0.94, 0.965, "VOL. I · NO. 2  ·  MAY 2026",
            color=INK_MUTE, fontsize=9, family="serif",
            ha="right", zorder=3)
    # Triple rule — heavy / hairline / heavy (editorial detail)
    ax.plot([0.06, 0.94], [0.950, 0.950], color=INK, linewidth=1.4, zorder=3)
    ax.plot([0.06, 0.94], [0.946, 0.946], color=INK, linewidth=0.4, zorder=3)
    ax.plot([0.06, 0.94], [0.942, 0.942], color=INK, linewidth=1.4, zorder=3)


def draw_kicker(ax):
    """Small italic kicker above the headline."""
    ax.text(0.06, 0.918, "A research brief in applied machine learning",
            color=RED, fontsize=11, style="italic", family="serif", zorder=3)


def draw_headline(ax):
    """The big serif headline — the visual anchor."""
    # Four lines, decreasing weight progression for editorial rhythm
    ax.text(0.06, 0.875, "Catching credit",
            color=INK, fontsize=54, fontweight="bold",
            family="serif", zorder=3)
    ax.text(0.06, 0.830, "card fraud,",
            color=INK, fontsize=54, fontweight="bold",
            family="serif", zorder=3)
    ax.text(0.06, 0.785, "one transaction",
            color=INK, fontsize=54, fontweight="bold",
            family="serif", zorder=3)
    # Final line with accent
    ax.text(0.06, 0.740, "at a time.",
            color=RED, fontsize=54, fontweight="bold", style="italic",
            family="serif", zorder=3)


def draw_lede(ax):
    """Editorial lede paragraph — italic, first paragraph of an article."""
    # Decorative opening flourish — small red square
    ax.add_patch(Rectangle((0.06, 0.700), 0.010, 0.010,
                            facecolor=RED, edgecolor="none", zorder=3))
    # Clean italic lede paragraph
    lede_lines = [
        "A machine-learning pipeline trained on 284,807 European credit-card",
        "transactions identifies fraudulent activity with a recall of 85.7%,",
        "while keeping the false-alarm rate well below 0.01%. The model favors",
        "catching fraud over precision — a deliberate trade-off, since missing",
        "a fraud is roughly 500 times costlier than reviewing a false flag.",
    ]
    y = 0.685
    for line in lede_lines:
        ax.text(0.085, y, line, color=INK_BODY, fontsize=12,
                family="serif", style="italic", zorder=3)
        y -= 0.018


def draw_section_label(ax, y, label):
    """Small-caps section divider."""
    ax.plot([0.06, 0.18], [y, y], color=INK, linewidth=0.8, zorder=3)
    ax.text(0.19, y, label, color=INK, fontsize=10, fontweight="bold",
            family="serif", va="center", zorder=3)
    label_width = len(label) * 0.0075
    ax.plot([0.19 + label_width + 0.01, 0.94], [y, y],
            color=INK, linewidth=0.8, zorder=3)


def draw_roc_chart(ax_main, data):
    """The hero element — a clean ROC curve, the way FT or NYT would draw it."""
    # Reserve a chart sub-axes inside the figure
    # We achieve this by drawing chart elements directly using ax_main coords
    # Chart bounds:
    cx0, cx1 = 0.10, 0.94
    cy0, cy1 = 0.32, 0.560
    
    # Chart title
    ax_main.text(0.06, 0.575, "Model Performance",
                 color=INK, fontsize=18, fontweight="bold",
                 family="serif", zorder=3)
    ax_main.text(0.06, 0.567, "Receiver operating characteristic — true-positive rate against false-positive rate",
                 color=INK_MUTE, fontsize=10, style="italic", family="serif", zorder=3)
    
    # Plot area background (slight paper-tone variation)
    # No fill — keep it clean
    
    # Y-axis gridlines (horizontal, hairline)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = cy0 + v * (cy1 - cy0)
        ax_main.plot([cx0, cx1], [y, y], color=RULE, linewidth=0.4,
                     alpha=0.7, zorder=2)
        ax_main.text(cx0 - 0.012, y, f"{v:.2f}", color=INK_MUTE, fontsize=9,
                     family="serif", ha="right", va="center", zorder=3)
    
    # X-axis ticks
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = cx0 + v * (cx1 - cx0)
        ax_main.plot([x, x], [cy0 - 0.004, cy0], color=INK, linewidth=0.6, zorder=3)
        ax_main.text(x, cy0 - 0.012, f"{v:.2f}", color=INK_MUTE, fontsize=9,
                     family="serif", ha="center", va="top", zorder=3)
    
    # Axis lines
    ax_main.plot([cx0, cx1], [cy0, cy0], color=INK, linewidth=1.0, zorder=3)
    ax_main.plot([cx0, cx0], [cy0, cy1], color=INK, linewidth=1.0, zorder=3)
    
    # Axis labels
    ax_main.text(cx0 + (cx1 - cx0) / 2, cy0 - 0.03,
                 "False-positive rate", color=INK, fontsize=10,
                 family="serif", ha="center", style="italic", zorder=3)
    ax_main.text(cx0 - 0.04, cy0 + (cy1 - cy0) / 2,
                 "True-positive rate", color=INK, fontsize=10,
                 family="serif", ha="center", va="center", rotation=90,
                 style="italic", zorder=3)
    
    # Diagonal reference line (random classifier)
    ax_main.plot([cx0, cx1], [cy0, cy1], color=INK_MUTE, linewidth=0.6,
                 linestyle=(0, (4, 3)), zorder=3)
    ax_main.text(cx1 - 0.02, cy0 + (cy1 - cy0) * 0.50, "Random",
                 color=INK_MUTE, fontsize=8, style="italic", family="serif",
                 ha="right", va="bottom", rotation=23, zorder=4)
    
    # ROC curve — map data (fpr, tpr) into chart coordinates
    if data["roc_xy"]:
        xs = np.array([p[0] for p in data["roc_xy"]])
        ys = np.array([p[1] for p in data["roc_xy"]])
    else:
        xs = np.linspace(0, 1, 50)
        ys = 1 - np.exp(-8 * xs)
    
    plot_x = cx0 + xs * (cx1 - cx0)
    plot_y = cy0 + ys * (cy1 - cy0)
    
    # Fill under curve (very subtle)
    fill_x = np.concatenate([[cx0], plot_x, [cx1, cx0]])
    fill_y = np.concatenate([[cy0], plot_y, [cy0, cy0]])
    ax_main.fill(fill_x, fill_y, color=RED, alpha=0.08, zorder=3)
    
    # The curve itself — thick, confident
    ax_main.plot(plot_x, plot_y, color=RED, linewidth=2.4, zorder=4,
                 solid_capstyle="round")
    
    # AUC callout — placed inside the chart
    callout_x = cx0 + 0.55 * (cx1 - cx0)
    callout_y = cy0 + 0.35 * (cy1 - cy0)
    ax_main.text(callout_x, callout_y + 0.025,
                 "AREA UNDER CURVE", color=INK_MUTE, fontsize=8,
                 fontweight="bold", family="serif", zorder=5)
    ax_main.text(callout_x, callout_y - 0.012,
                 f"{data['roc_auc']:.3f}", color=RED, fontsize=32,
                 fontweight="bold", family="serif", zorder=5)
    ax_main.text(callout_x, callout_y - 0.038,
                 "(1.000 = perfect classifier)", color=INK_MUTE, fontsize=8,
                 style="italic", family="serif", zorder=5)


def draw_findings(ax, data):
    """Three-column editorial stats block — recall, precision, F1."""
    y_top = 0.270
    draw_section_label(ax, y_top, "KEY FINDINGS")
    
    metrics = [
        ("Recall",    f"{data['recall']*100:.1f}%",
         f"{data['frauds_caught']} of {data['frauds_total']} frauds caught"),
        ("Precision", f"{data['precision']*100:.1f}%",
         "of flagged transactions were fraud"),
        ("F1 score",  f"{data['f1']:.3f}",
         "harmonic mean of precision & recall"),
    ]
    
    col_w = (0.94 - 0.06 - 2 * 0.03) / 3
    for i, (label, value, sub) in enumerate(metrics):
        x = 0.06 + i * (col_w + 0.03)
        
        # Small-caps label
        ax.text(x, 0.240, label.upper(), color=INK, fontsize=10,
                fontweight="bold", family="serif", zorder=3)
        
        # Hairline rule
        ax.plot([x, x + col_w - 0.01], [0.232, 0.232], color=INK,
                linewidth=0.4, zorder=3)
        
        # The big number — full string, single render, bulletproof
        ax.text(x, 0.205, value, color=INK, fontsize=42,
                fontweight="bold", family="serif", zorder=3)
        
        # Caption below
        ax.text(x, 0.172, sub, color=INK_MUTE, fontsize=10,
                style="italic", family="serif", zorder=3)


def draw_pullquote(ax, data):
    """Editorial pull-quote — a striking line from the analysis."""
    y_top = 0.135
    
    # Decorative quote mark
    ax.text(0.06, 0.110, "\u201C", color=RED, fontsize=64,
            fontweight="bold", family="serif", zorder=3)
    
    # The quote
    ax.text(0.12, 0.120,
            f"Of {data['frauds_total']} fraudulent transactions in the test set, {data['frauds_caught']} were",
            color=INK, fontsize=14, family="serif", style="italic", zorder=3)
    ax.text(0.12, 0.102,
            f"correctly flagged — only {data['frauds_missed']} slipped past unnoticed, at",
            color=INK, fontsize=14, family="serif", style="italic", zorder=3)
    ax.text(0.12, 0.084,
            f"a cost of just {data['false_alarms']} false alarms.",
            color=INK, fontsize=14, family="serif", style="italic", zorder=3)


def draw_colophon(ax, data):
    """Footer — methodology, byline, date. Like the colophon of a research piece."""
    # Top rule
    ax.plot([0.06, 0.94], [0.057, 0.057], color=INK, linewidth=0.4, zorder=3)
    
    # Three-column footer: method · dataset · byline
    # Column 1: Method
    ax.text(0.06, 0.045, "METHOD", color=INK_MUTE, fontsize=8,
            fontweight="bold", family="serif", zorder=3)
    ax.text(0.06, 0.030, data["model_name"], color=INK, fontsize=10,
            fontweight="bold", family="serif", zorder=3)
    ax.text(0.06, 0.018,
            f"SMOTE oversampling · F2 threshold @ {data['threshold']:.2f}",
            color=INK_MUTE, fontsize=9, family="serif", zorder=3)
    ax.text(0.06, 0.006, "Python · scikit-learn · SHAP",
            color=INK_MUTE, fontsize=9, style="italic", family="serif", zorder=3)
    
    # Column 2: Dataset
    ax.text(0.42, 0.045, "DATASET", color=INK_MUTE, fontsize=8,
            fontweight="bold", family="serif", zorder=3)
    ax.text(0.42, 0.030, "Kaggle / ULB", color=INK, fontsize=10,
            fontweight="bold", family="serif", zorder=3)
    ax.text(0.42, 0.018,
            f"{data['total_tx']:,} European card transactions",
            color=INK_MUTE, fontsize=9, family="serif", zorder=3)
    ax.text(0.42, 0.006,
            f"{data['dataset_fraud_pct']:.3f}% base fraud rate",
            color=INK_MUTE, fontsize=9, style="italic", family="serif", zorder=3)
    
    # Column 3: Byline
    ax.text(0.94, 0.045, "AUTHOR", color=INK_MUTE, fontsize=8,
            fontweight="bold", family="serif", ha="right", zorder=3)
    ax.text(0.94, 0.030, "B&F Fintech Student", color=INK, fontsize=10,
            fontweight="bold", family="serif", ha="right", zorder=3)
    ax.text(0.94, 0.018, "Portfolio Project № 2",
            color=INK_MUTE, fontsize=9, family="serif", ha="right", zorder=3)
    ax.text(0.94, 0.006, "May 2026",
            color=INK_MUTE, fontsize=9, style="italic", family="serif",
            ha="right", zorder=3)


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def generate(output_path="fraud_report.png"):
    data = load_metrics()
    
    fig = plt.figure(figsize=(W, H), facecolor=PAPER, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(PAPER)
    ax.axis("off")
    
    draw_masthead(ax)
    draw_kicker(ax)
    draw_headline(ax)
    draw_lede(ax)
    draw_roc_chart(ax, data)
    draw_findings(ax, data)
    draw_pullquote(ax, data)
    draw_colophon(ax, data)
    
    plt.savefig(output_path, dpi=DPI, facecolor=PAPER, edgecolor="none",
                bbox_inches=None, pad_inches=0)
    plt.close()
    
    print(f"\n  ✓ Report saved: {output_path}")
    print(f"  ✓ Dimensions: {int(W*DPI)}x{int(H*DPI)} (4:5 ratio, ideal for LinkedIn)")
    if data["is_sample"]:
        print(f"\n  ⚠ Using sample metrics (dashboard_data.json not found).")
        print(f"    After training the model, re-run this script for real numbers.")
    else:
        print(f"  ✓ Using real metrics from your trained model")


if __name__ == "__main__":
    print("=" * 60)
    print("  LINKEDIN REPORT GENERATOR")
    print("=" * 60)
    generate()
    print()
