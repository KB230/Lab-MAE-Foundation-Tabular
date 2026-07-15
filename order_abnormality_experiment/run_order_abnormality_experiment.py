import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ordering and abnormality tests on Lab-MAE embeddings."
    )
    parser.add_argument("--input", default=str(EXPERIMENT_DIR / "data" / "order_abnormality_dataset.csv"))
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "results"))
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


def add_calibration(calibration_frames, lab_name, task, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return

    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_pred,
        n_bins=10,
        strategy="quantile",
    )
    calibration_frames.append(
        pd.DataFrame(
            {
                "task": task,
                "lab_name": lab_name,
                "mean_pred": mean_pred,
                "frac_positive": frac_pos,
            }
        )
    )


def add_roc(roc_frames, lab_name, task, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_frames.append(
        pd.DataFrame(
            {
                "task": task,
                "lab_name": lab_name,
                "fpr": fpr,
                "tpr": tpr,
            }
        )
    )


def plot_metric_bar(metrics_df, metric, title, output_path):
    plot_df = metrics_df.dropna(subset=[metric]).sort_values(metric)
    if plot_df.empty:
        return

    height = max(5, len(plot_df) * 0.32)
    plt.figure(figsize=(9, height))
    plt.barh(plot_df["lab_name"], plot_df[metric])
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

    n_series = task_df["lab_name"].nunique()
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    for lab_name, lab_df in task_df.groupby("lab_name"):
        plt.plot(lab_df[x_col], lab_df[y_col], marker="o", linewidth=1.5,
                 label=lab_name if n_series <= 12 else None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n_series <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover z columns from the header without loading the full file.
    header = pd.read_csv(args.input, nrows=0)
    z_cols = [c for c in header.columns if c.startswith("z_")]
    if not z_cols:
        raise RuntimeError("No z_ columns found. Run build_order_abnormality_dataset.py first.")

    # Stream the CSV in chunks, accumulating per-lab arrays in float32.
    # Loading all 76 labs × 325k patients as float64 DataFrames would require ~15 GB;
    # this approach uses ~6.5 GB by reading only the needed columns in float32.
    print(f"Streaming {args.input} into per-lab arrays...")
    usecols = ["lab_name", "ordered", "abnormal"] + z_cols
    lab_data: dict[str, dict[str, list]] = {}
    for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=50_000):
        chunk = chunk.dropna(subset=z_cols)
        for lab_name, sub in chunk.groupby("lab_name", sort=False):
            if lab_name not in lab_data:
                lab_data[lab_name] = {"z": [], "ordered": [], "abnormal": []}
            lab_data[lab_name]["z"].append(sub[z_cols].to_numpy(dtype=np.float32))
            lab_data[lab_name]["ordered"].append(sub["ordered"].to_numpy(dtype=np.int8))
            lab_data[lab_name]["abnormal"].append(sub["abnormal"].to_numpy(dtype=np.float32))

    lab_arrays = {
        name: {
            "z":        np.concatenate(d["z"]),
            "ordered":  np.concatenate(d["ordered"]).astype(int),
            "abnormal": np.concatenate(d["abnormal"]),
        }
        for name, d in lab_data.items()
    }
    del lab_data
    gc.collect()
    print(f"Loaded {len(lab_arrays)} labs.")

    metrics = []
    prediction_frames = []
    decile_frames = []
    calibration_frames = []
    roc_frames = []

    for lab_name, arrays in lab_arrays.items():
        z       = arrays["z"]
        ordered = arrays["ordered"]
        abnormal = arrays["abnormal"]

        if len(np.unique(ordered)) < 2:
            print(f"Skipping {lab_name}: only one ordering class.")
            continue

        idx = np.arange(len(z))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=ordered,
        )

        X_train, X_test         = z[train_idx], z[test_idx]
        y_train_order           = ordered[train_idx]
        y_test_order            = ordered[test_idx]
        abnormal_train          = abnormal[train_idx]
        abnormal_test           = abnormal[test_idx]

        # Task 1: can the embedding predict whether this lab was measured next day?
        _, pred_order = fit_predict_logistic(X_train, y_train_order, X_test)

        order_auc   = safe_auc(y_test_order, pred_order)
        order_brier = brier_score_loss(y_test_order, pred_order)
        add_calibration(calibration_frames, lab_name, "order", y_test_order, pred_order)
        add_roc(roc_frames, lab_name, "order", y_test_order, pred_order)

        # Task 2: among measured next-day labs, predict abnormality.
        ordered_mask_train = (y_train_order == 1) & ~np.isnan(abnormal_train)
        ordered_mask_test  = (y_test_order  == 1) & ~np.isnan(abnormal_test)

        pred_abnormal  = np.full(len(test_idx), np.nan, dtype=np.float32)
        abnormal_auc   = np.nan
        abnormal_auprc = np.nan

        if (
            ordered_mask_train.sum() > 0
            and ordered_mask_test.sum() > 0
            and len(np.unique(abnormal_train[ordered_mask_train])) > 1
        ):
            abnormal_model, pred_abnormal_ordered = fit_predict_logistic(
                X_train[ordered_mask_train],
                abnormal_train[ordered_mask_train].astype(int),
                X_test[ordered_mask_test],
            )
            # Score all test rows so Task 3 can combine order × abnormality risk.
            pred_abnormal = abnormal_model.predict_proba(X_test)[:, 1].astype(np.float32)
            y_test_abnormal = abnormal_test[ordered_mask_test].astype(int)
            abnormal_auc   = safe_auc(y_test_abnormal, pred_abnormal_ordered)
            abnormal_auprc = average_precision_score(y_test_abnormal, pred_abnormal_ordered)
            add_calibration(calibration_frames, lab_name, "abnormal_if_ordered",
                            y_test_abnormal, pred_abnormal_ordered)
            add_roc(roc_frames, lab_name, "abnormal_if_ordered",
                    y_test_abnormal, pred_abnormal_ordered)
        else:
            print(f"Skipping abnormality model for {lab_name}: not enough classes/cases.")

        # Task 3: expected abnormal yield = P(ordered) × P(abnormal | ordered).
        expected_abnormal    = pred_order * pred_abnormal
        observed_and_abnormal = ((y_test_order == 1) & (abnormal_test == 1)).astype(int)
        joint_mask = ~np.isnan(expected_abnormal)

        joint_auc   = np.nan
        joint_auprc = np.nan
        joint_brier = np.nan
        if joint_mask.sum() > 0 and len(np.unique(observed_and_abnormal[joint_mask])) > 1:
            y_joint    = observed_and_abnormal[joint_mask]
            pred_joint = expected_abnormal[joint_mask]
            joint_auc   = safe_auc(y_joint, pred_joint)
            joint_auprc = average_precision_score(y_joint, pred_joint)
            joint_brier = brier_score_loss(y_joint, pred_joint)
            add_calibration(calibration_frames, lab_name, "joint_observed_abnormal",
                            y_joint, pred_joint)
            add_roc(roc_frames, lab_name, "joint_observed_abnormal", y_joint, pred_joint)

        # Severity decile: bin by pred_order, report observed abnormal rate among ordered.
        order_decile = pd.qcut(pred_order, q=10, labels=False, duplicates="drop")
        test_frame = pd.DataFrame({
            "lab_name":            lab_name,
            "ordered":             y_test_order,
            "abnormal":            abnormal_test,
            "pred_order":          pred_order,
            "pred_abnormal":       pred_abnormal,
            "expected_abnormal":   expected_abnormal,
            "observed_and_abnormal": observed_and_abnormal,
            "order_decile":        order_decile,
        })
        deciles = (
            test_frame[test_frame["ordered"] == 1]
            .groupby("order_decile", dropna=True)
            .agg(
                lab_name=("lab_name", "first"),
                n_rows=("ordered", "size"),
                mean_pred_order=("pred_order", "mean"),
                observed_abnormal_rate=("abnormal", "mean"),
            )
            .reset_index()
        )

        metrics.append({
            "lab_name":            lab_name,
            "n_total":             len(z),
            "n_ordered":           int(ordered.sum()),
            "n_abnormal_ordered":  int(np.nansum(abnormal[ordered == 1])),
            "order_auroc":         order_auc,
            "order_brier":         order_brier,
            "abnormal_auroc":      abnormal_auc,
            "abnormal_auprc":      abnormal_auprc,
            "joint_auroc":         joint_auc,
            "joint_auprc":         joint_auprc,
            "joint_brier":         joint_brier,
        })
        prediction_frames.append(test_frame)
        decile_frames.append(deciles)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    deciles_df = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    calibration_df = (
        pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames
        else pd.DataFrame(columns=["task", "lab_name", "mean_pred", "frac_positive"])
    )
    roc_df = (
        pd.concat(roc_frames, ignore_index=True)
        if roc_frames
        else pd.DataFrame(columns=["task", "lab_name", "fpr", "tpr"])
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    deciles_df.to_csv(output_dir / "order_decile_abnormal_yield.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_curves.csv", index=False)
    roc_df.to_csv(output_dir / "roc_curves.csv", index=False)

    plot_metric_bar(
        metrics_df,
        "order_auroc",
        "Next-Day Lab Order Prediction AUROC",
        output_dir / "bar_order_auroc.png",
    )
    plot_metric_bar(
        metrics_df,
        "order_brier",
        "Next-Day Lab Order Prediction Brier Score",
        output_dir / "bar_order_brier.png",
    )
    plot_metric_bar(
        metrics_df,
        "abnormal_auroc",
        "Next-Day Abnormality Prediction AUROC Among Ordered Labs",
        output_dir / "bar_abnormal_auroc.png",
    )
    plot_metric_bar(
        metrics_df,
        "abnormal_auprc",
        "Next-Day Abnormality Prediction AUPRC Among Ordered Labs",
        output_dir / "bar_abnormal_auprc.png",
    )
    plot_metric_bar(
        metrics_df,
        "joint_auroc",
        "Joint Observed-And-Abnormal Prediction AUROC",
        output_dir / "bar_joint_auroc.png",
    )
    plot_metric_bar(
        metrics_df,
        "joint_auprc",
        "Joint Observed-And-Abnormal Prediction AUPRC",
        output_dir / "bar_joint_auprc.png",
    )
    plot_metric_bar(
        metrics_df,
        "joint_brier",
        "Joint Observed-And-Abnormal Prediction Brier Score",
        output_dir / "bar_joint_brier.png",
    )

    plot_curves(
        roc_df,
        "order",
        "fpr",
        "tpr",
        "False positive rate",
        "True positive rate",
        "ROC: Next-Day Lab Order Prediction",
        output_dir / "roc_order_prediction.png",
    )
    plot_curves(
        roc_df,
        "abnormal_if_ordered",
        "fpr",
        "tpr",
        "False positive rate",
        "True positive rate",
        "ROC: Next-Day Abnormality Prediction Among Ordered Labs",
        output_dir / "roc_abnormal_if_ordered.png",
    )
    plot_curves(
        roc_df,
        "joint_observed_abnormal",
        "fpr",
        "tpr",
        "False positive rate",
        "True positive rate",
        "ROC: Joint Observed-And-Abnormal Prediction",
        output_dir / "roc_joint_observed_abnormal.png",
    )
    plot_curves(
        calibration_df,
        "order",
        "mean_pred",
        "frac_positive",
        "Mean predicted probability",
        "Observed frequency",
        "Calibration: Next-Day Lab Order Prediction",
        output_dir / "calibration_order_prediction.png",
    )
    plot_curves(
        calibration_df,
        "abnormal_if_ordered",
        "mean_pred",
        "frac_positive",
        "Mean predicted probability",
        "Observed frequency",
        "Calibration: Next-Day Abnormality Among Ordered Labs",
        output_dir / "calibration_abnormal_if_ordered.png",
    )
    plot_curves(
        calibration_df,
        "joint_observed_abnormal",
        "mean_pred",
        "frac_positive",
        "Mean predicted probability",
        "Observed frequency",
        "Calibration: Joint Observed-And-Abnormal Prediction",
        output_dir / "calibration_joint_observed_abnormal.png",
    )

    n_labs = deciles_df["lab_name"].nunique()
    plt.figure(figsize=(8, 5))
    for lab_name, lab_deciles in deciles_df.groupby("lab_name"):
        plt.plot(
            lab_deciles["mean_pred_order"],
            lab_deciles["observed_abnormal_rate"],
            marker="o",
            label=lab_name if n_labs <= 12 else None,
        )
    plt.xlabel("Predicted ordering probability")
    plt.ylabel("Observed abnormality rate among ordered")
    if n_labs <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "order_decile_abnormal_yield.png", dpi=200)

    print(metrics_df.to_string(index=False))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
