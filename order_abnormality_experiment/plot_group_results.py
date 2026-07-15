"""Regenerate group experiment plots with routine/non-routine distinguished by line style.

Reads CSVs from results_group/ and writes plots back to the same directory.
Routine panels use solid lines; non-routine panels use dashed lines.

Run from the order_abnormality_experiment directory:
    python plot_group_results.py
    python plot_group_results.py --input-dir results_group --output-dir results_group
"""

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


ROUTINE_PANELS = {"BMP", "CBC", "LFT", "Coagulation", "ABG"}
NONROUTINE_PANELS = {
    "Electrolytes", "Iron Studies", "Creatine Kinase", "Pancreatic Enzymes",
    "LDH", "Troponin", "HbA1c", "CRP", "TSH", "Lipid Panel", "Osmolality", "Uric Acid",
}

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(EXPERIMENT_DIR / "results_group"))
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "results_group"))
    return parser.parse_args()


def categorize(panel):
    if panel in ROUTINE_PANELS:
        return "routine"
    if panel in NONROUTINE_PANELS:
        return "nonroutine"
    return "other"


def _linestyle(category):
    return "-" if category == "routine" else "--"


def _add_category_legend(ax):
    """Add a small inset legend explaining the line-style encoding."""
    routine_handle = mlines.Line2D([], [], color="black", linestyle="-",
                                   linewidth=1.5, label="Routine")
    nonroutine_handle = mlines.Line2D([], [], color="black", linestyle="--",
                                      linewidth=1.5, label="Non-routine")
    return routine_handle, nonroutine_handle


def plot_metric_bar_split(metrics_df, metric, title, output_path):
    routine = metrics_df[metrics_df["category"] == "routine"].dropna(subset=[metric]).sort_values(metric)
    nonroutine = metrics_df[metrics_df["category"] == "nonroutine"].dropna(subset=[metric]).sort_values(metric)

    if routine.empty and nonroutine.empty:
        return

    has_routine = not routine.empty
    has_nonroutine = not nonroutine.empty
    n_cols = int(has_routine) + int(has_nonroutine)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, max(3, max(
        len(routine) if has_routine else 0,
        len(nonroutine) if has_nonroutine else 0,
    ) * 0.45 + 1)))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if has_routine:
        axes[col].barh(routine["panel"], routine[metric])
        axes[col].set_xlim(0, 1)
        axes[col].set_xlabel(metric)
        axes[col].set_title(f"Routine — {title}")
        col += 1
    if has_nonroutine:
        axes[col].barh(nonroutine["panel"], nonroutine[metric])
        axes[col].set_xlim(0, 1)
        axes[col].set_xlabel(metric)
        axes[col].set_title(f"Non-routine — {title}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_curves(curves_df, task, x_col, y_col, xlabel, ylabel, title, output_path, diagonal=True):
    if curves_df.empty:
        return
    task_df = curves_df[curves_df["task"] == task].copy()
    if task_df.empty:
        return

    task_df["category"] = task_df["label"].map(categorize)

    fig, ax = plt.subplots(figsize=(8, 6))
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1)

    for label, sub_df in task_df.groupby("label"):
        cat = sub_df["category"].iloc[0]
        ax.plot(sub_df[x_col], sub_df[y_col],
                linestyle=_linestyle(cat), linewidth=1.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Panel legend to the right; category style legend below it.
    panel_legend = ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1),
                             loc="upper left", borderaxespad=0, title="Panel")
    ax.add_artist(panel_legend)
    r_handle, nr_handle = _add_category_legend(ax)
    ax.legend(handles=[r_handle, nr_handle], fontsize=8,
              bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0,
              title="Type")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decile(deciles_df, output_path):
    if deciles_df.empty:
        return
    deciles_df = deciles_df.copy()
    deciles_df["category"] = deciles_df["panel"].map(categorize)

    fig, ax = plt.subplots(figsize=(8, 5))
    for panel, panel_deciles in deciles_df.groupby("panel"):
        cat = panel_deciles["category"].iloc[0]
        ax.plot(
            panel_deciles["mean_pred_order"],
            panel_deciles["observed_mean_abnormal_rate"],
            marker="o", linewidth=1.5,
            linestyle=_linestyle(cat), label=panel,
        )

    ax.set_xlabel("Mean predicted order probability (decile)")
    ax.set_ylabel("Observed abnormal lab rate among ordered panels")
    ax.set_title("Predicted Order Probability vs Observed Abnormal Rate")

    panel_legend = ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1),
                             loc="upper left", borderaxespad=0, title="Panel")
    ax.add_artist(panel_legend)
    r_handle, nr_handle = _add_category_legend(ax)
    ax.legend(handles=[r_handle, nr_handle], fontsize=8,
              bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0,
              title="Type")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(input_dir / "metrics.csv")
    roc_df = pd.read_csv(input_dir / "roc_curves.csv")
    cal_df = pd.read_csv(input_dir / "calibration_curves.csv")
    deciles_df = pd.read_csv(input_dir / "order_decile_abnormal_yield.csv")

    metrics_df["category"] = metrics_df["panel"].map(categorize)

    ordered_col = "ordered_any"

    metric_specs = [
        ("order_auroc", f"Panel Order Prediction AUROC ({ordered_col})"),
        ("order_brier", f"Panel Order Prediction Brier Score ({ordered_col})"),
        ("abnormal_auroc", "Panel Abnormality Prediction AUROC (Among Ordered)"),
        ("abnormal_auprc", "Panel Abnormality Prediction AUPRC (Among Ordered)"),
        ("joint_auroc", "Joint Ordered-And-Abnormal Prediction AUROC"),
        ("joint_auprc", "Joint Ordered-And-Abnormal Prediction AUPRC"),
        ("joint_brier", "Joint Ordered-And-Abnormal Prediction Brier Score"),
    ]
    for metric, title in metric_specs:
        if metric not in metrics_df.columns:
            continue
        plot_metric_bar_split(metrics_df, metric, title, output_dir / f"bar_{metric}.png")

    for task, roc_title, cal_title in [
        ("order", "ROC: Panel Order Prediction", "Calibration: Panel Order Prediction"),
        ("abnormal_if_ordered", "ROC: Abnormality Among Ordered", "Calibration: Abnormality Among Ordered"),
        ("joint_observed_abnormal", "ROC: Joint Ordered-And-Abnormal", "Calibration: Joint Ordered-And-Abnormal"),
    ]:
        plot_curves(roc_df, task, "fpr", "tpr",
                    "False positive rate", "True positive rate",
                    roc_title, output_dir / f"roc_{task}.png")
        plot_curves(cal_df, task, "mean_pred", "frac_positive",
                    "Mean predicted probability", "Observed frequency",
                    cal_title, output_dir / f"calibration_{task}.png")

    plot_decile(deciles_df, output_dir / "order_decile_abnormal_rate.png")

    print(f"Done. Plots written to {output_dir}")


if __name__ == "__main__":
    main()
