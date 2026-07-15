"""Regenerate individual-lab experiment plots split into routine vs non-routine panels.

Reads CSVs from full_results/ and writes split plots back to the same directory.

Run from the order_abnormality_experiment directory:
    python plot_individual_results.py
    python plot_individual_results.py --input-dir full_results --output-dir full_results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent

ROUTINE_LABS = {
    # BMP
    "Bicarbonate", "Calcium Total", "Chloride", "Creatinine",
    "Glucose", "Potassium", "Sodium", "Urea Nitrogen",
    # CBC
    "Hematocrit", "Hemoglobin", "MCH", "MCHC", "MCV", "Platelet Count",
    "RDW", "Red Blood Cells", "White Blood Cells", "RDW-SD",
    "Absolute Lymphocyte Count", "Bands", "Basophils", "Eosinophils",
    "Lymphocytes", "Metamyelocytes", "Monocytes", "Myelocytes", "Neutrophils",
    "Nucleated Red Cells", "Absolute Basophil Count", "Absolute Eosinophil Count",
    "Absolute Monocyte Count", "Absolute Neutrophil Count", "Immature Granulocytes",
    # LFT
    "ALT", "Albumin", "Alkaline Phosphatase", "AST",
    "Bilirubin Indirect", "Bilirubin Total",
    # Coagulation
    "Fibrinogen", "INR", "PT", "PTT",
    # ABG
    "Base Excess", "Calculated Total CO2", "Chloride Whole Blood", "Free Calcium",
    "Glucose Blood Gas", "Hematocrit Calculated", "Hemoglobin Blood Gas", "Lactate",
    "Oxygen Saturation", "pCO2", "pH", "pO2",
    "Potassium Whole Blood", "Sodium Whole Blood", "Temperature",
}

NONROUTINE_LABS = {
    # Electrolytes
    "Magnesium", "Phosphate",
    # Creatine Kinase
    "Creatine Kinase", "Creatine Kinase MB",
    # LDH
    "Lactate Dehydrogenase",
    # Troponin
    "Troponin T",
    # Iron Studies
    "Ferritin", "Iron", "Iron Binding Capacity", "Transferrin",
    # Pancreatic Enzymes
    "Amylase", "Lipase",
    # Targeted single-biomarker
    "Hemoglobin A1c", "C-Reactive Protein", "Thyroid Stimulating Hormone",
    "Triglycerides", "Osmolality", "Uric Acid",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(EXPERIMENT_DIR / "full_results"))
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "full_results"))
    return parser.parse_args()


def categorize(lab_name):
    if lab_name in ROUTINE_LABS:
        return "routine"
    if lab_name in NONROUTINE_LABS:
        return "nonroutine"
    return "other"


def plot_bar(df, metric, title, output_path, xlabel=None):
    plot_df = df.dropna(subset=[metric]).sort_values(metric)
    if plot_df.empty:
        return
    height = max(3, len(plot_df) * 0.32)
    fig, ax = plt.subplots(figsize=(6, height))
    ax.barh(plot_df["lab_name"], plot_df[metric])
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel or metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _side_legend(ax, n_labs):
    ax.legend(
        fontsize=7,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        ncol=max(1, n_labs // 30),
    )


def plot_curves(curves_df, task, x_col, y_col, xlabel, ylabel, title, output_path, diagonal=True):
    task_df = curves_df[curves_df["task"] == task]
    if task_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for lab_name, sub in task_df.groupby("lab_name"):
        ax.plot(sub[x_col], sub[y_col], linewidth=1.2, label=lab_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _side_legend(ax, task_df["lab_name"].nunique())
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decile(deciles_df, title, output_path):
    if deciles_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab_name, sub in deciles_df.groupby("lab_name"):
        ax.plot(sub["mean_pred_order"], sub["observed_abnormal_rate"],
                marker="o", linewidth=1.2, label=lab_name)
    ax.set_xlabel("Mean predicted order probability (decile)")
    ax.set_ylabel("Observed abnormal rate among ordered labs")
    ax.set_title(title)
    _side_legend(ax, deciles_df["lab_name"].nunique())
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(input_dir / "metrics.csv")
    roc_df = pd.read_csv(input_dir / "roc_curves.csv")
    cal_df = pd.read_csv(input_dir / "calibration_curves.csv")
    deciles = pd.read_csv(input_dir / "order_decile_abnormal_yield.csv")

    metrics["category"] = metrics["lab_name"].map(categorize)
    roc_df["category"] = roc_df["lab_name"].map(categorize)
    cal_df["category"] = cal_df["lab_name"].map(categorize)
    deciles["category"] = deciles["lab_name"].map(categorize)

    metric_specs = [
        ("order_auroc", "Order Prediction AUROC", "AUROC"),
        ("order_brier", "Order Prediction Brier Score", "Brier score"),
        ("abnormal_auroc", "Abnormality Prediction AUROC (Among Ordered)", "AUROC"),
        ("abnormal_auprc", "Abnormality Prediction AUPRC (Among Ordered)", "AUPRC"),
        ("joint_auroc", "Joint Ordered-And-Abnormal AUROC", "AUROC"),
        ("joint_auprc", "Joint Ordered-And-Abnormal AUPRC", "AUPRC"),
        ("joint_brier", "Joint Ordered-And-Abnormal Brier Score", "Brier score"),
    ]

    for cat, cat_label in [("routine", "Routine"), ("nonroutine", "Non-routine")]:
        cat_metrics = metrics[metrics["category"] == cat]
        cat_roc = roc_df[roc_df["category"] == cat]
        cat_cal = cal_df[cal_df["category"] == cat]
        cat_deciles = deciles[deciles["category"] == cat]

        # Bar plots.
        for metric, title, xlabel in metric_specs:
            if metric not in cat_metrics.columns:
                continue
            plot_bar(
                cat_metrics, metric,
                f"{cat_label} — {title}",
                output_dir / f"bar_{metric}_{cat}.png",
                xlabel=xlabel,
            )

        # ROC curves per task.
        for task, title in [
            ("order", "Lab Order Prediction"),
            ("abnormal_if_ordered", "Abnormality Among Ordered"),
            ("joint_observed_abnormal", "Joint Ordered-And-Abnormal"),
        ]:
            plot_curves(
                cat_roc, task, "fpr", "tpr",
                "False positive rate", "True positive rate",
                f"{cat_label} ROC — {title}",
                output_dir / f"roc_{task}_{cat}.png",
            )
            plot_curves(
                cat_cal, task, "mean_pred", "frac_positive",
                "Mean predicted probability", "Observed frequency",
                f"{cat_label} Calibration — {title}",
                output_dir / f"calibration_{task}_{cat}.png",
            )

        # Severity decile.
        plot_decile(
            cat_deciles,
            f"{cat_label} — Predicted Order Probability vs Observed Abnormal Rate",
            output_dir / f"order_decile_abnormal_rate_{cat}.png",
        )

    written = sorted(p.name for p in output_dir.glob("*_routine.png")) + \
              sorted(p.name for p in output_dir.glob("*_nonroutine.png"))
    for f in written:
        print(f"  {f}")
    print(f"\nDone. {len(written)} split plots written to {output_dir}")


if __name__ == "__main__":
    main()
