import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
PANELS = [
    "BMP", "CBC", "LFT", "Coagulation",
    "ABG", "Electrolytes", "Iron Studies", "Creatine Kinase", "Pancreatic Enzymes",
    "LDH", "Troponin", "HbA1c", "CRP", "TSH", "Lipid Panel", "Osmolality", "Uric Acid",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run panel-level group ordering/abnormality experiment."
    )
    parser.add_argument(
        "--input",
        default=str(EXPERIMENT_DIR / "data" / "group_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_DIR / "results_group"),
    )
    parser.add_argument(
        "--ordered-col",
        default="ordered_any",
        choices=["ordered_any", "ordered_majority"],
        help="Primary ordering target: ordered_any (>=1 lab) or ordered_majority (>50%% of panel).",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def fit_predict_logistic(X_train, y_train, X_test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    return model, model.predict_proba(X_test)[:, 1]


def add_calibration(calibration_frames, label, task, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return
    frac_pos, mean_pred = calibration_curve(
        y_true, y_pred, n_bins=10, strategy="quantile"
    )
    calibration_frames.append(
        pd.DataFrame({
            "task": task,
            "label": label,
            "mean_pred": mean_pred,
            "frac_positive": frac_pos,
        })
    )


def add_roc(roc_frames, label, task, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_frames.append(
        pd.DataFrame({
            "task": task,
            "label": label,
            "fpr": fpr,
            "tpr": tpr,
        })
    )


def plot_metric_bar(metrics_df, metric, title, output_path):
    plot_df = metrics_df.dropna(subset=[metric]).sort_values(metric)
    if plot_df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.barh(plot_df["panel"], plot_df[metric])
    plt.xlim(0, 1)
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_curves(curves_df, task, x_col, y_col, xlabel, ylabel, title, output_path):
    if curves_df.empty:
        return
    task_df = curves_df[curves_df["task"] == task]
    if task_df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for label, sub_df in task_df.groupby("label"):
        plt.plot(sub_df[x_col], sub_df[y_col], marker="o", linewidth=1.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def within_panel_analysis(df, z_cols, ordered_col, seed, test_size):
    """Tasks 1–3 for each panel independently.

    Task 1: predict panel ordered (ordered_col).
    Task 2: predict abnormal_any among panels that were ordered.
    Task 3: joint expected yield = pred_order × pred_abnormal vs observed ordered & abnormal.
    """
    metrics = []
    prediction_frames = []
    decile_frames = []
    calibration_frames = []
    roc_frames = []

    for panel in PANELS:
        panel_df = df[df["panel"] == panel].copy().dropna(subset=z_cols)

        if panel_df[ordered_col].nunique() < 2:
            print(f"Skipping {panel}: only one class in {ordered_col}.")
            continue

        train_idx, test_idx = train_test_split(
            panel_df.index,
            test_size=test_size,
            random_state=seed,
            stratify=panel_df[ordered_col],
        )
        train_df = panel_df.loc[train_idx]
        test_df = panel_df.loc[test_idx].copy()

        X_train = train_df[z_cols].to_numpy()
        y_train_order = train_df[ordered_col].astype(int).to_numpy()
        X_test = test_df[z_cols].to_numpy()
        y_test_order = test_df[ordered_col].astype(int).to_numpy()

        # Task 1: can the embedding predict whether the panel was ordered?
        _, pred_order = fit_predict_logistic(X_train, y_train_order, X_test)
        test_df["pred_order"] = pred_order

        order_auc = safe_auc(y_test_order, pred_order)
        order_brier = brier_score_loss(y_test_order, pred_order)

        add_calibration(calibration_frames, panel, "order", y_test_order, pred_order)
        add_roc(roc_frames, panel, "order", y_test_order, pred_order)

        # Task 2: among ordered panels, predict abnormal_any.
        ordered_train = train_df[train_df[ordered_col] == 1].dropna(subset=["abnormal_any"])
        ordered_test = test_df[test_df[ordered_col] == 1].dropna(subset=["abnormal_any"])

        abnormal_auc = np.nan
        abnormal_auprc = np.nan
        if (
            len(ordered_train) > 0
            and len(ordered_test) > 0
            and ordered_train["abnormal_any"].nunique() > 1
        ):
            abnormal_model, pred_abnormal_ordered = fit_predict_logistic(
                ordered_train[z_cols].to_numpy(),
                ordered_train["abnormal_any"].astype(int).to_numpy(),
                ordered_test[z_cols].to_numpy(),
            )
            # Score all test rows for Task 3.
            test_df["pred_abnormal"] = abnormal_model.predict_proba(X_test)[:, 1]
            y_test_abnormal = ordered_test["abnormal_any"].astype(int).to_numpy()
            abnormal_auc = safe_auc(y_test_abnormal, pred_abnormal_ordered)
            abnormal_auprc = average_precision_score(y_test_abnormal, pred_abnormal_ordered)
            add_calibration(
                calibration_frames, panel, "abnormal_if_ordered",
                y_test_abnormal, pred_abnormal_ordered,
            )
            add_roc(
                roc_frames, panel, "abnormal_if_ordered",
                y_test_abnormal, pred_abnormal_ordered,
            )
        else:
            test_df["pred_abnormal"] = np.nan
            print(f"Skipping abnormality model for {panel}: not enough cases or classes.")

        # Task 3: joint expected yield = P(ordered) × P(abnormal | ordered).
        test_df["expected_abnormal"] = test_df["pred_order"] * test_df["pred_abnormal"]
        test_df["observed_and_abnormal"] = (
            (test_df[ordered_col] == 1) & (test_df["abnormal_any"] == 1)
        ).astype(int)
        joint_eval = test_df.dropna(subset=["expected_abnormal"])

        joint_auc = np.nan
        joint_auprc = np.nan
        joint_brier = np.nan
        if len(joint_eval) > 0 and joint_eval["observed_and_abnormal"].nunique() > 1:
            y_joint = joint_eval["observed_and_abnormal"].astype(int).to_numpy()
            pred_joint = joint_eval["expected_abnormal"].to_numpy()
            joint_auc = safe_auc(y_joint, pred_joint)
            joint_auprc = average_precision_score(y_joint, pred_joint)
            joint_brier = brier_score_loss(y_joint, pred_joint)
            add_calibration(
                calibration_frames, panel, "joint_observed_abnormal", y_joint, pred_joint
            )
            add_roc(roc_frames, panel, "joint_observed_abnormal", y_joint, pred_joint)

        # Severity decile: bin ordered panels by pred_order, check abnormal_rate.
        test_df["order_decile"] = pd.qcut(
            test_df["pred_order"], q=10, labels=False, duplicates="drop"
        )
        deciles = (
            test_df[test_df[ordered_col] == 1]
            .groupby("order_decile", dropna=True)
            .agg(
                panel=("panel", "first"),
                n_rows=(ordered_col, "size"),
                mean_pred_order=("pred_order", "mean"),
                observed_abnormal_any_rate=("abnormal_any", "mean"),
                observed_mean_n_abnormal=("n_abnormal", "mean"),
                observed_mean_abnormal_rate=("abnormal_rate", "mean"),
            )
            .reset_index()
        )

        metrics.append({
            "panel": panel,
            "n_total": len(panel_df),
            "n_ordered": int(panel_df[ordered_col].sum()),
            "n_any_abnormal_ordered": int(panel_df["abnormal_any"].sum(skipna=True)),
            "order_auroc": order_auc,
            "order_brier": order_brier,
            "abnormal_auroc": abnormal_auc,
            "abnormal_auprc": abnormal_auprc,
            "joint_auroc": joint_auc,
            "joint_auprc": joint_auprc,
            "joint_brier": joint_brier,
        })
        prediction_frames.append(test_df)
        decile_frames.append(deciles)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    deciles_df = (
        pd.concat(decile_frames, ignore_index=True)
        if decile_frames
        else pd.DataFrame()
    )
    calibration_df = (
        pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames
        else pd.DataFrame(columns=["task", "label", "mean_pred", "frac_positive"])
    )
    roc_df = (
        pd.concat(roc_frames, ignore_index=True)
        if roc_frames
        else pd.DataFrame(columns=["task", "label", "fpr", "tpr"])
    )

    return metrics_df, predictions_df, deciles_df, calibration_df, roc_df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    z_cols = [col for col in df.columns if col.startswith("z_")]
    if not z_cols:
        raise RuntimeError("No z_ columns found. Run build_group_dataset.py first.")

    df = df[df["panel"].isin(PANELS)].copy()
    counts = df["panel"].value_counts().to_dict()
    print(f"Loaded {len(df)} rows: {counts}")

    print(f"\n=== Within-panel analysis (target: {args.ordered_col}) ===")
    within_metrics, predictions_df, deciles_df, calibration_df, roc_df = (
        within_panel_analysis(df, z_cols, args.ordered_col, args.seed, args.test_size)
    )
    print(within_metrics.to_string(index=False))

    # Write CSVs.
    within_metrics.to_csv(output_dir / "metrics.csv", index=False)
    if not predictions_df.empty:
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    if not deciles_df.empty:
        deciles_df.to_csv(output_dir / "order_decile_abnormal_yield.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_curves.csv", index=False)
    roc_df.to_csv(output_dir / "roc_curves.csv", index=False)

    # Within-panel bar plots.
    for metric, title in [
        ("order_auroc", f"Panel Order Prediction AUROC ({args.ordered_col})"),
        ("order_brier", f"Panel Order Prediction Brier Score ({args.ordered_col})"),
        ("abnormal_auroc", "Panel Abnormality Prediction AUROC (Among Ordered)"),
        ("abnormal_auprc", "Panel Abnormality Prediction AUPRC (Among Ordered)"),
        ("joint_auroc", "Joint Ordered-And-Abnormal Prediction AUROC"),
        ("joint_auprc", "Joint Ordered-And-Abnormal Prediction AUPRC"),
        ("joint_brier", "Joint Ordered-And-Abnormal Prediction Brier Score"),
    ]:
        plot_metric_bar(within_metrics, metric, title, output_dir / f"bar_{metric}.png")

    # ROC and calibration curves.
    for task, title_suffix in [
        ("order", "Panel Order Prediction"),
        ("abnormal_if_ordered", "Panel Abnormality Among Ordered"),
        ("joint_observed_abnormal", "Joint Ordered-And-Abnormal Prediction"),
    ]:
        plot_curves(
            roc_df, task, "fpr", "tpr",
            "False positive rate", "True positive rate",
            f"ROC: {title_suffix}",
            output_dir / f"roc_{task}.png",
        )
        plot_curves(
            calibration_df, task, "mean_pred", "frac_positive",
            "Mean predicted probability", "Observed frequency",
            f"Calibration: {title_suffix}",
            output_dir / f"calibration_{task}.png",
        )

    # Severity: decile of predicted order probability vs observed abnormal_rate.
    if not deciles_df.empty:
        plt.figure(figsize=(8, 5))
        for panel, panel_deciles in deciles_df.groupby("panel"):
            plt.plot(
                panel_deciles["mean_pred_order"],
                panel_deciles["observed_mean_abnormal_rate"],
                marker="o",
                linewidth=1.5,
                label=panel,
            )
        plt.xlabel("Mean predicted order probability (decile)")
        plt.ylabel("Observed abnormal lab rate among ordered panels")
        plt.title("Predicted Order Probability vs Observed Abnormal Rate")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "order_decile_abnormal_rate.png", dpi=200)
        plt.close()

    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
