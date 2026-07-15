"""
Build a panel-level dataset for the group ordering/abnormality experiment.

Reuses z embeddings from order_abnormality_dataset.csv, then reads X_test.csv
in chunks to compute panel-level labels for each patient.

Output: group_dataset.csv with one row per patient per panel.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Anion Gap (50868) excluded — calculated value. Atypical Lymphocytes (51143) — qualitative.
# Machine settings (50816/50819/50826) excluded. Drug levels (50986/51009) excluded.
# Urine panels and unknowns (50934/50947/51678) excluded.
# Labs normally absent use (0.0, 0.0): any detected value is flagged abnormal.
PANEL_LABS = {
    "BMP": {
        50882: ("Bicarbonate",                23.0,  28.0),
        50893: ("Calcium Total",               8.5,  10.5),
        50902: ("Chloride",                   98.0, 107.0),
        50912: ("Creatinine",                  0.7,   1.3),
        50931: ("Glucose",                    70.0, 140.0),
        50971: ("Potassium",                   3.5,   5.0),
        50983: ("Sodium",                    136.0, 145.0),
        51006: ("Urea Nitrogen",               8.0,  20.0),
    },
    "CBC": {
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
    },
    # Liver Function Tests: ordered for suspected hepatic pathology, drug toxicity, or jaundice.
    "LFT": {
        50861: ("ALT",                  7.0,   56.0),
        50862: ("Albumin",              3.5,    5.0),
        50863: ("Alkaline Phosphatase", 44.0, 147.0),
        50878: ("AST",                 10.0,   40.0),
        50884: ("Bilirubin Indirect",   0.1,    0.8),
        50885: ("Bilirubin Total",      0.2,    1.2),
    },
    # Coagulation Panel: ordered for anticoagulation monitoring, pre-procedure, or coagulopathy.
    "Coagulation": {
        51214: ("Fibrinogen",  200.0, 400.0),
        51237: ("INR",           0.8,   1.2),
        51274: ("PT",           11.0,  15.0),
        51275: ("PTT",          25.0,  35.0),
    },
    # Blood Gas: ordered for acid-base and respiratory assessment. Machine settings excluded.
    "ABG": {
        50802: ("Base Excess",           -2.0,   2.0),
        50804: ("Calculated Total CO2",  22.0,  29.0),
        50806: ("Chloride Whole Blood",  98.0, 107.0),
        50808: ("Free Calcium",           1.15,  1.35),
        50809: ("Glucose Blood Gas",     70.0, 140.0),
        50810: ("Hematocrit Calculated", 36.0,  50.0),
        50811: ("Hemoglobin Blood Gas",  12.0,  18.0),
        50813: ("Lactate",                0.5,   2.0),
        50817: ("Oxygen Saturation",     95.0, 100.0),
        50818: ("pCO2",                  35.0,  45.0),
        50820: ("pH",                     7.35,  7.45),
        50821: ("pO2",                   80.0, 100.0),
        50822: ("Potassium Whole Blood",  3.5,   5.0),
        50824: ("Sodium Whole Blood",   136.0, 145.0),
        50825: ("Temperature",           36.5,  37.5),
    },
    # Mg and Phos are co-ordered as add-ons not included in the standard BMP.
    "Electrolytes": {
        50960: ("Magnesium",  1.7, 2.2),
        50970: ("Phosphate",  2.5, 4.5),
    },
    # Iron Studies: ordered together for anemia workup or iron overload evaluation.
    "Iron Studies": {
        50924: ("Ferritin",               12.0, 300.0),
        50952: ("Iron",                   60.0, 170.0),
        50953: ("Iron Binding Capacity", 240.0, 360.0),
        50998: ("Transferrin",           170.0, 370.0),
    },
    # CK and CK-MB are co-ordered for myocardial injury or rhabdomyolysis evaluation.
    "Creatine Kinase": {
        50910: ("Creatine Kinase",    30.0, 200.0),
        50911: ("Creatine Kinase MB",  0.0,   5.0),
    },
    # Amylase and Lipase are co-ordered for pancreatitis evaluation.
    "Pancreatic Enzymes": {
        50867: ("Amylase",  30.0, 110.0),
        50956: ("Lipase",    7.0,  60.0),
    },
    # Single-biomarker targeted panels — each represents a distinct clinical question.
    "LDH": {
        50954: ("Lactate Dehydrogenase", 135.0, 225.0),
    },
    "Troponin": {
        51003: ("Troponin T", 0.0, 0.01),
    },
    "HbA1c": {
        50852: ("Hemoglobin A1c", 4.0, 5.6),
    },
    "CRP": {
        50889: ("C-Reactive Protein", 0.0, 1.0),
    },
    "TSH": {
        50993: ("Thyroid Stimulating Hormone", 0.5, 4.5),
    },
    "Lipid Panel": {
        51000: ("Triglycerides", 0.0, 150.0),
    },
    "Osmolality": {
        50964: ("Osmolality", 280.0, 300.0),
    },
    "Uric Acid": {
        51007: ("Uric Acid", 2.5, 7.0),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build panel-level group ordering/abnormality dataset."
    )
    parser.add_argument(
        "--individual-dataset",
        default=str(EXPERIMENT_DIR / "data" / "order_abnormality_dataset.csv"),
        help="Individual lab dataset produced by build_order_abnormality_dataset.py (source of z embeddings).",
    )
    parser.add_argument("--input", default=str(PROJECT_DIR / "data" / "X_test.csv"))
    parser.add_argument("--output", default=str(EXPERIMENT_DIR / "data" / "group_dataset.csv"))
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


def build_group_rows(chunk_df, valid_source_rows):
    rows = []
    for source_row, row in chunk_df.iterrows():
        if source_row not in valid_source_rows:
            continue
        for panel_name, labs in PANEL_LABS.items():
            n_total = len(labs)

            # Group next-day labs by their draw timestamp. Labs from the same
            # panel order share an identical time (same blood tube). We take the
            # dominant draw — the timestamp with the most labs present — so that
            # a spot single-lab check on the same day doesn't inflate the count.
            draws = {}  # time -> list of (value, low, high)
            for itemid, (_, low, high) in labs.items():
                next_val = row.get(f"npval_last_{itemid}", np.nan)
                next_time = row.get(f"nptime_last_{itemid}", np.nan)
                if pd.notna(next_val) and pd.notna(next_time):
                    draws.setdefault(next_time, []).append((next_val, low, high))

            if draws:
                # Pick the draw with the most labs; break ties by earliest time.
                dominant_time = max(draws, key=lambda t: (len(draws[t]), -t))
                dominant_labs = draws[dominant_time]
                n_ordered = len(dominant_labs)
                n_abnormal = sum(1 for val, low, high in dominant_labs
                                 if val < low or val > high)
            else:
                dominant_time = np.nan
                n_ordered = 0
                n_abnormal = 0

            rows.append({
                "source_row":       source_row,
                "panel":            panel_name,
                "n_panel_labs":     n_total,
                "panel_draw_time":  dominant_time,
                "n_ordered":        n_ordered,
                "ordered_any":      int(n_ordered > 0),
                "ordered_majority": int(n_ordered > n_total / 2),
                "n_abnormal":       n_abnormal,
                "abnormal_any":     int(n_abnormal > 0) if n_ordered > 0 else np.nan,
                "abnormal_rate":    n_abnormal / n_ordered if n_ordered > 0 else np.nan,
            })
    return rows


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load z embeddings — one vector per patient reused across panels.
    print(f"Loading z embeddings from {args.individual_dataset}...")
    ind_df = pd.read_csv(args.individual_dataset)
    z_cols = [c for c in ind_df.columns if c.startswith("z_")]
    if not z_cols:
        raise RuntimeError("No z_ columns found. Run build_order_abnormality_dataset.py first.")
    z_per_row = (
        ind_df[["source_row"] + z_cols]
        .drop_duplicates("source_row")
        .set_index("source_row")
    )
    valid_source_rows = set(z_per_row.index)
    print(f"Loaded embeddings for {len(valid_source_rows)} unique patients.")

    # Stream X_test.csv in chunks, compute group labels, join z, write incrementally.
    print(f"Reading {args.input} in chunks of {args.chunk_size}...")
    first_chunk = True
    rows_processed = 0
    patients_written = 0

    for chunk_df in pd.read_csv(args.input, chunksize=args.chunk_size):
        chunk_df.index = range(rows_processed, rows_processed + len(chunk_df))
        rows_processed += len(chunk_df)

        chunk_rows = build_group_rows(chunk_df, valid_source_rows)
        if not chunk_rows:
            continue

        chunk_group_df = pd.DataFrame(chunk_rows)
        chunk_group_df = chunk_group_df.join(z_per_row, on="source_row")
        chunk_group_df.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        patients_written += chunk_group_df["source_row"].nunique()
        print(f"  {rows_processed} CSV rows scanned, {patients_written} patients written...")

    if first_chunk:
        raise RuntimeError("No rows written. Check that source_row indices match between datasets.")

    print(f"\nWrote {output_path}")
    result = pd.read_csv(output_path, usecols=["panel", "ordered_any", "ordered_majority", "abnormal_any"])
    print(result.groupby("panel").agg(
        n=("ordered_any", "size"),
        ordered_any_rate=("ordered_any", "mean"),
        ordered_majority_rate=("ordered_majority", "mean"),
        abnormal_any_rate=("abnormal_any", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
