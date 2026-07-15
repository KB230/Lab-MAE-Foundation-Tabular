import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


# Normal ranges for all panel labs. Keys are MIMIC lab item IDs (npval_<id> columns).
# Labs normally absent use (0.0, 0.0): any detected value is flagged abnormal.
# Excluded: Anion Gap (50868) — calculated value; Atypical Lymphocytes (51143) — qualitative;
#   PEEP/Tidal Volume/FiO2 (50819/50826/50816) — machine settings; Tacrolimus (50986) /
#   Vancomycin (51009) — drug levels with indication-specific therapeutic ranges;
#   H/I/L (50934/50947/51678) — undocumented; urine panels — different reference context.
# Labs not present as npval_last_* in X_test.csv are filtered out automatically at runtime.
NORMAL_RANGES = {
    # BMP (Basic Metabolic Panel)
    50882: ("Bicarbonate",                23.0,  28.0),
    50893: ("Calcium Total",               8.5,  10.5),
    50902: ("Chloride",                   98.0, 107.0),
    50912: ("Creatinine",                  0.7,   1.3),
    50931: ("Glucose",                    70.0, 140.0),
    50971: ("Potassium",                   3.5,   5.0),
    50983: ("Sodium",                    136.0, 145.0),
    51006: ("Urea Nitrogen",               8.0,  20.0),
    # CBC (Complete Blood Count)
    51221: ("Hematocrit",                 36.0,  50.0),
    51222: ("Hemoglobin",                 12.0,  18.0),
    51248: ("MCH",                        27.0,  33.0),
    51249: ("MCHC",                       31.5,  35.7),
    51250: ("MCV",                        80.0, 100.0),
    51265: ("Platelet Count",            150.0, 450.0),
    51277: ("RDW",                        11.5,  14.5),
    51279: ("Red Blood Cells",             4.2,   5.9),
    51301: ("White Blood Cells",           4.0,  11.0),
    52172: ("RDW-SD",                     39.0,  46.0),
    51133: ("Absolute Lymphocyte Count",   1.0,   4.8),
    51144: ("Bands",                       0.0,   5.0),
    51146: ("Basophils",                   0.0,   1.0),
    51200: ("Eosinophils",                 0.0,   7.0),
    51244: ("Lymphocytes",                20.0,  40.0),
    51251: ("Metamyelocytes",              0.0,   0.0),
    51254: ("Monocytes",                   2.0,  10.0),
    51255: ("Myelocytes",                  0.0,   0.0),
    51256: ("Neutrophils",                50.0,  70.0),
    51257: ("Nucleated Red Cells",         0.0,   0.0),
    52069: ("Absolute Basophil Count",     0.0,   0.1),
    52073: ("Absolute Eosinophil Count",   0.0,   0.5),
    52074: ("Absolute Monocyte Count",     0.2,   0.9),
    52075: ("Absolute Neutrophil Count",   1.8,   7.7),
    52135: ("Immature Granulocytes",       0.0,   0.0),
    # LFT (Liver Function Tests)
    50861: ("ALT",                         7.0,  56.0),
    50862: ("Albumin",                     3.5,   5.0),
    50863: ("Alkaline Phosphatase",       44.0, 147.0),
    50878: ("AST",                        10.0,  40.0),
    50884: ("Bilirubin Indirect",          0.1,   0.8),
    50885: ("Bilirubin Total",             0.2,   1.2),
    # Coagulation Panel
    51214: ("Fibrinogen",                200.0, 400.0),
    51237: ("INR",                         0.8,   1.2),
    51274: ("PT",                         11.0,  15.0),
    51275: ("PTT",                        25.0,  35.0),
    # Blood Gas (ABG/VBG) — machine settings (50816/50819/50826) excluded
    50802: ("Base Excess",                -2.0,   2.0),
    50804: ("Calculated Total CO2",       22.0,  29.0),
    50806: ("Chloride Whole Blood",       98.0, 107.0),
    50808: ("Free Calcium",                1.15,  1.35),
    50809: ("Glucose Blood Gas",          70.0, 140.0),
    50810: ("Hematocrit Calculated",      36.0,  50.0),
    50811: ("Hemoglobin Blood Gas",       12.0,  18.0),
    50813: ("Lactate",                     0.5,   2.0),
    50817: ("Oxygen Saturation",          95.0, 100.0),
    50818: ("pCO2",                       35.0,  45.0),
    50820: ("pH",                          7.35,  7.45),
    50821: ("pO2",                        80.0, 100.0),
    50822: ("Potassium Whole Blood",       3.5,   5.0),
    50824: ("Sodium Whole Blood",        136.0, 145.0),
    50825: ("Temperature",                36.5,  37.5),
    # Electrolytes add-on
    50960: ("Magnesium",                   1.7,   2.2),
    50970: ("Phosphate",                   2.5,   4.5),
    # Creatine Kinase
    50910: ("Creatine Kinase",            30.0, 200.0),
    50911: ("Creatine Kinase MB",          0.0,   5.0),
    # Lactate Dehydrogenase
    50954: ("Lactate Dehydrogenase",     135.0, 225.0),
    # Troponin-T
    51003: ("Troponin T",                  0.0,   0.01),
    # Iron Studies
    50924: ("Ferritin",                   12.0, 300.0),
    50952: ("Iron",                       60.0, 170.0),
    50953: ("Iron Binding Capacity",     240.0, 360.0),
    50998: ("Transferrin",               170.0, 370.0),
    # Pancreatic Enzymes
    50867: ("Amylase",                    30.0, 110.0),
    50956: ("Lipase",                      7.0,  60.0),
    # Other targeted labs
    50852: ("Hemoglobin A1c",              4.0,   5.6),
    50889: ("C-Reactive Protein",          0.0,   1.0),
    50993: ("Thyroid Stimulating Hormone", 0.5,   4.5),
    51000: ("Triglycerides",               0.0, 150.0),
    50964: ("Osmolality",                280.0, 300.0),
    51007: ("Uric Acid",                   2.5,   7.0),
}

ID_COLS = ["first_race", "chartyear", "hadm_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build lab ordering/abnormality dataset using current-row Lab-MAE embeddings."
    )
    parser.add_argument("--input", default=str(PROJECT_DIR / "data" / "X_test.csv"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "data" / "order_abnormality_dataset.csv"))
    parser.add_argument("--summary-output", default=str(EXPERIMENT_DIR / "data" / "order_abnormality_summary.csv"))
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "epoch390_checkpoint"))
    parser.add_argument("--weights", default=str(PROJECT_DIR / "model_checkpoint.zip"))
    parser.add_argument("--norm-parameters", default=str(PROJECT_DIR / "norm_parameters.pkl"))
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Stop after this many input rows. Omit to process the full file.")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Rows read from CSV at a time. Controls peak RAM usage.")
    parser.add_argument("--min-current-observed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, cuda, or mps. auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only create labels/counts. Useful before installing Lab-MAE dependencies.",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    return parser.parse_args()


def abnormal_label(value, low, high):
    if pd.isna(value):
        return np.nan
    return int(value < low or value > high)


def resolve_device(device):
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_imputer(args, dim):
    from MAEImputer import ReMaskerStep

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    imputer = ReMaskerStep(
        dim=dim,
        mask_ratio=args.mask_ratio,
        max_epochs=1,
        save_path=args.save_path,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        weigths=args.weights,
        device=device,
    )

    with open(args.norm_parameters, "rb") as file:
        imputer.norm_parameters = pickle.load(file)

    return imputer


def pooled_embeddings(imputer, X, batch_size):
    embeddings = imputer.extract_embeddings(X, eval_batch_size=batch_size).numpy()
    observed = X.notna().to_numpy(dtype=np.float32)

    pooled = []
    for i in range(embeddings.shape[0]):
        mask = observed[i].astype(bool)
        if mask.sum() == 0:
            pooled.append(np.full(embeddings.shape[-1], np.nan))
        else:
            # Lab-MAE returns one embedding per input column. We pool observed
            # current columns into one row-level representation z.
            # Important: do not multiply missing embeddings by 0, because
            # NaN * 0 is still NaN. Select observed columns before averaging.
            pooled.append(np.nanmean(embeddings[i][mask], axis=0))

    return np.asarray(pooled)


def get_target_itemids(df):
    itemids = []
    for itemid in NORMAL_RANGES:
        current_col = f"npval_{itemid}"
        future_col = f"npval_last_{itemid}"
        if current_col in df.columns and future_col in df.columns:
            itemids.append(itemid)
    return itemids


def make_model_input(df, model_cols):
    X = df[model_cols].copy()

    # The dataset already contains next-day answers in npval_last_* columns.
    # We'll re asign future columns "last" to NaN to avoid data leakage
    last_cols = [col for col in X.columns if "_last_" in col]
    X[last_cols] = np.nan
    return X


def build_labels(df, target_itemids, current_value_cols, min_current_observed):
    rows = []
    # Count how many current lab values are observed for each row.
    current_observed = df[current_value_cols].notna().sum(axis=1)

    for source_row, row in df.iterrows():
        # Verify that the current row has at least min_current_observed lab values observed.
        if current_observed.loc[source_row] < min_current_observed:
            continue
        
        patient_meta = {
            "source_row": source_row,
            "hadm_id": row.get("hadm_id", np.nan),
            "first_race": row.get("first_race", np.nan),
            "chartyear": row.get("chartyear", np.nan),
            "current_observed": int(current_observed.loc[source_row]),
        }

        # for each target lab, create a row with the current value, next-day value, and abnormality label.
        for itemid in target_itemids:
            lab_name, low, high = NORMAL_RANGES[itemid]
            current_value_col = f"npval_{itemid}"
            current_time_col = f"nptime_{itemid}"
            next_value_col = f"npval_last_{itemid}"
            next_time_col = f"nptime_last_{itemid}"

            next_value = row[next_value_col]

            # ordered means this lab exists in the precomputed next-day column.
            # If it was not ordered/measured next day, abnormality is undefined.
            ordered = int(pd.notna(next_value))
            abnormal = abnormal_label(next_value, low, high) if ordered else np.nan

            rows.append(
                {
                    **patient_meta,
                    "lab_itemid": itemid,
                    "lab_name": lab_name,
                    "current_lab_value": row.get(current_value_col, np.nan),
                    "current_lab_time": row.get(current_time_col, np.nan),
                    "ordered": ordered,
                    "abnormal": abnormal,
                    "next_lab_value": next_value if ordered else np.nan,
                    "next_lab_time": row.get(next_time_col, np.nan) if ordered else np.nan,
                    "normal_low": low,
                    "normal_high": high,
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Read only the header to get column layout without loading the full file.
    header_df = pd.read_csv(input_path, nrows=0)
    model_cols = [col for col in header_df.columns if col not in ID_COLS]
    current_value_cols = [
        col for col in model_cols if col.startswith("npval_") and "_last_" not in col
    ]
    target_itemids = get_target_itemids(header_df)
    print(f"Target labs: {target_itemids}")
    print(f"Model columns: {len(model_cols)}")

    # Load the imputer once before the chunk loop.
    if not args.skip_embeddings:
        imputer = load_imputer(args, dim=len(model_cols))
        print("Extracting and pooling Lab-MAE embeddings...")

    first_chunk = True
    rows_processed = 0

    for chunk_df in pd.read_csv(input_path, chunksize=args.chunk_size):
        if args.max_rows is not None and rows_processed >= args.max_rows:
            break
        if args.max_rows is not None:
            chunk_df = chunk_df.head(args.max_rows - rows_processed)

        # Shift index so source_row is globally unique across chunks.
        chunk_df.index = range(rows_processed, rows_processed + len(chunk_df))

        X_current = make_model_input(chunk_df, model_cols)
        meta = build_labels(chunk_df, target_itemids, current_value_cols, args.min_current_observed)

        if meta.empty:
            rows_processed += len(chunk_df)
            continue

        used_source_rows = meta["source_row"].drop_duplicates().to_numpy()
        X_for_embed = X_current.loc[used_source_rows].reset_index(drop=True)
        source_to_embedding_row = {sr: i for i, sr in enumerate(used_source_rows)}
        meta["embedding_row"] = meta["source_row"].map(source_to_embedding_row)

        if args.skip_embeddings:
            out = meta.drop(columns=["embedding_row"])
        else:
            z = pooled_embeddings(imputer, X_for_embed, args.batch_size)
            z_cols = [f"z_{i}" for i in range(z.shape[1])]
            z_df = pd.DataFrame(z, columns=z_cols)
            z_df["embedding_row"] = np.arange(len(z_df))
            out = meta.merge(z_df, on="embedding_row", how="left")
            out = out.drop(columns=["embedding_row"])

        out.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        rows_processed += len(chunk_df)
        print(f"Processed {rows_processed} rows...")

    if first_chunk:
        raise RuntimeError("No eligible rows were created.")

    # Compute summary without reloading the z columns.
    result_df = pd.read_csv(output_path, usecols=["lab_itemid", "lab_name", "ordered", "abnormal"])
    summary = (
        result_df.groupby(["lab_itemid", "lab_name"], dropna=False)
        .agg(
            rows=("ordered", "size"),
            ordered=("ordered", "sum"),
            abnormal=("abnormal", "sum"),
            ordered_rate=("ordered", "mean"),
            abnormal_rate_among_ordered=("abnormal", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Wrote dataset: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
