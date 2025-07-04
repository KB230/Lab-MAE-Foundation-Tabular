import pandas as pd
import numpy as np
import os

# --- Configuration ---
# all of these are the blood, chemistry variants
LABS_TO_ANALYZE = {
    'Potassium': 'npval_50971',
    'Creatinine': 'npval_50912',
    'INR': 'npval_51237',
    'Troponin_I': 'npval_51002',
    'Troponin_T': 'npval_51003'
}

FEATURE_EMBEDDINGS_FILE = '100_Labs_Train_0.25Mask_L_V3/test_feature_embeddings_dev.npy'
PREPROCESSED_DATA_FILE = '../data/X_test.csv'
DEV_MODE_ROWS = 10000
OUTPUT_EXCEL_FILE = 'embedding_analysis_data.xlsx'

def clean_missing(df, threshold=20 + 3, missing_per_col=500, cols_to_remove=None):
    df_clean = df.copy()
    df_clean = df_clean.dropna(thresh=threshold)
    
    if not isinstance(cols_to_remove, list):
        if missing_per_col and not cols_to_remove:
            sparse_cols = df_clean.columns[df_clean.notna().sum() < missing_per_col].tolist()
            ids_to_remove = set(['_' + col.split('_')[-1] for col in sparse_cols])
            def ids_in_string(value_set, target_string):
                for value in value_set:
                    if value in target_string: return True
                return False
            cols_to_remove = [col for col in df_clean.columns if ids_in_string(ids_to_remove, col)]
    
    if cols_to_remove:
        print(f'Removing {len(cols_to_remove)} sparse columns...')
        df_clean.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        
    return df_clean, []

print("Loading and aligning data (this might take a moment)...")
feature_embeddings = np.load(FEATURE_EMBEDDINGS_FILE)
df_original_raw = pd.read_csv(PREPROCESSED_DATA_FILE, nrows=DEV_MODE_ROWS)
df_aligned, _ = clean_missing(df_original_raw)

print(f"Data alignment successful. Shape: {df_aligned.shape}")
assert feature_embeddings.shape[0] == df_aligned.shape[0], "Shape mismatch after cleaning!"

# Check if lab is in the current sample of data
print("\n--- Verifying Lab Availability in the Sample ---")
for sheet_name, column_name in LABS_TO_ANALYZE.items():
    if column_name in df_aligned.columns:
        # Count how many non-NaN values exist for this lab
        valid_count = df_aligned[column_name].notna().sum()
        if valid_count > 0:
            print(f"  [SUCCESS] Found {valid_count} valid entries for '{sheet_name}' ({column_name}).")
        else:
            print(f"  [WARNING] Column '{column_name}' exists but has NO valid entries. Increase DEV_MODE_ROWS.")
    else:
        print(f"  [ERROR] Column '{column_name}' for '{sheet_name}' NOT FOUND in the data. Check spelling or increase DEV_MODE_ROWS.")




# --- Create Excel file with a sheet for each lab ---
print(f"Creating Excel file: {OUTPUT_EXCEL_FILE}")
with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
    for sheet_name, column_name in LABS_TO_ANALYZE.items():
        print(f"  Processing sheet for: {sheet_name} ({column_name})")

        # Find the integer index of the lab's column in the aligned dataframe
        try:
            col_idx = df_aligned.columns.get_loc(column_name)
        except KeyError:
            print(f"    WARNING: Column '{column_name}' not found in the data. Skipping.")
            continue

        # Extract the specific embeddings and true values
        X_embeddings = feature_embeddings[:, col_idx, :]
        y_true_values = df_aligned[column_name].values

        # Filter out rows where the true value is NaN
        valid_indices = ~np.isnan(y_true_values)
        X_clean = X_embeddings[valid_indices]
        y_clean = y_true_values[valid_indices]

        # Create the final DataFrame for this sheet
        # First column is the true value
        df_sheet = pd.DataFrame({'true_value': y_clean})

        # Create column names for the 64 embedding dimensions
        embedding_cols = {f'emb_{i}': X_clean[:, i] for i in range(X_clean.shape[1])}
        
        # Add the embedding dimensions to the DataFrame
        df_sheet = df_sheet.assign(**embedding_cols)

        # Write this DataFrame to a new sheet in the Excel file
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

print("Excel file created successfully.")