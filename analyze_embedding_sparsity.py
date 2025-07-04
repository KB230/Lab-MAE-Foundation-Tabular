import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================= ANALYSIS CONFIGURATION =======================
# 1. SET THE TARGET LAB ITEMID HERE
# 50971 = Potassium, 50912 = Creatinine, 51237 = INR
TARGET_LAB_ITEMID = 51237 

# 2. SET THE THRESHOLD FOR "NEAR ZERO"
# We'll count the percentage of values whose absolute value is less than this.
SPARSITY_THRESHOLD = 1e-4
# ======================================================================

# --- Configuration ---
FEATURE_EMBEDDINGS_FILE = '100_Labs_Train_0.25Mask_L_V3/test_feature_embeddings_dev.npy'
PREPROCESSED_DATA_FILE = '../data/X_test.csv'
DEV_MODE_ROWS = 10000

# --- The proven data alignment pipeline ---
def clean_missing(df, threshold=20 + 3, missing_per_col=500, cols_to_remove=None):
    # This is the full, original cleaning logic that we know works.
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
        
    return df_clean, cols_to_remove

# --- 1. Load and Align Data ---
print("Loading and aligning data...")
feature_embeddings = np.load(FEATURE_EMBEDDINGS_FILE)
df_original_raw = pd.read_csv(PREPROCESSED_DATA_FILE, nrows=DEV_MODE_ROWS)
df_aligned, _ = clean_missing(df_original_raw)
print(f"Data alignment successful. Shape: {df_aligned.shape}")
assert feature_embeddings.shape[0] == df_aligned.shape[0], "Shape mismatch after cleaning!"

# --- 2. Isolate Embeddings for the Target Lab ---
column_name = f'npval_{TARGET_LAB_ITEMID}'
print(f"\n--- Starting Sparsity Analysis for Lab: {column_name} ---")
if column_name not in df_aligned.columns:
    print(f"FATAL ERROR: Column '{column_name}' not found.")
    exit()

feature_columns = [col for col in df_aligned.columns if 'npval_' in col and '_last_' not in col]
col_idx = feature_columns.index(column_name)
# This gets all embeddings for our target lab: shape (num_samples, 64)
target_embeddings = feature_embeddings[:, col_idx, :]

# Clean out any rows where the embedding itself is all NaN (can happen with sparse data)
target_embeddings_clean = target_embeddings[~np.isnan(target_embeddings).all(axis=1)]


# --- 3. Perform Sparsity Analysis ---
# Flatten the 2D array of embeddings into a single 1D array of all values
all_embedding_values = target_embeddings_clean.flatten()

# Calculate the percentage of values that are very close to zero
near_zero_mask = np.isclose(all_embedding_values, 0, atol=SPARSITY_THRESHOLD)
sparsity_percentage = near_zero_mask.mean() * 100

print("\n--- Sparsity Analysis Results ---")
print(f"Total number of embedding values analyzed: {len(all_embedding_values)}")
print(f"Percentage of values within +/- {SPARSITY_THRESHOLD} of zero: {sparsity_percentage:.2f}%")


# --- 4. Visualize the Distribution ---
print("Generating histogram...")
plt.figure(figsize=(10, 6))
sns.histplot(all_embedding_values, bins=100, kde=True) # kde=True adds a smooth density curve
plt.title(f"Distribution of Embedding Values for '{column_name}'")
plt.xlabel("Value within Embedding Vector")
plt.ylabel("Frequency (Count)")
plt.grid(True)
plt.axvline(0, color='r', linestyle='--', label='Zero') # Add a line at zero for reference
plt.legend()
plt.show()

print("Done.")