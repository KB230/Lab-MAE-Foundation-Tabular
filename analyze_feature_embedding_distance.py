import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import cosine, euclidean

# --- Configuration ---
FEATURE_EMBEDDINGS_FILE = '100_Labs_Train_0.25Mask_L_V3/test_feature_embeddings_dev.npy'
PREPROCESSED_DATA_FILE = '../data/X_test.csv'
DEV_MODE_ROWS = 10000
# 50971 = Potassium, 50912 = Creatinine, 51237 = INR
TARGET_LAB_ITEMID = 50971
# ======================================================================

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

# --- 2. Isolate Data for the Target Lab ---
column_name = f'npval_{TARGET_LAB_ITEMID}'
print(f"\n--- Starting Analysis for Lab: {column_name} ---")
if column_name not in df_aligned.columns:
    print(f"FATAL ERROR: Column '{column_name}' not found.")
    exit()

feature_columns = [col for col in df_aligned.columns if 'npval_' in col and '_last_' not in col]
col_idx = feature_columns.index(column_name)

X = feature_embeddings[:, col_idx, :]
y = df_aligned[column_name].values

# Clean the data: 
# Filter out rows where the TARGET VALUE (y) is NaN
print(f"Total samples before cleaning: {len(y)}")
valid_y_indices = ~np.isnan(y)
X_pre_clean = X[valid_y_indices]
y_pre_clean = y[valid_y_indices]
print(f"Samples after removing NaN target values: {len(y_pre_clean)}")
# Filter out rows where the INPUT FEATURES (X) contain any NaNs.
# np.isnan(X).any(axis=1) creates a boolean mask that is True for any row
# in X that contains at least one NaN value.
nan_in_x_mask = np.isnan(X_pre_clean).any(axis=1)
valid_x_indices = ~nan_in_x_mask
# Apply this final filter to both X and y to keep them aligned
X_clean = X_pre_clean[valid_x_indices]
y_clean = y_pre_clean[valid_x_indices]
print(f"Samples after removing NaN X values: {len(y_clean)}")


#NEW CODE STARTS HERE (ABOVE IS FROM analyze_feature_embedding_w_regression.py)
# --- Binning, Centroid, and Distance Calculation ---

# Define the bins and labels for the target lab
# This is where you will change the values for different labs
if TARGET_LAB_ITEMID == 51237: # INR
    bins = [0, 1.2, 3.5, np.inf] # Bins for INR
    labels = ['Normal', 'Therapeutic', 'High']
elif TARGET_LAB_ITEMID == 50912: # Creatinine
    bins = [0, 1.3, 4.0, np.inf] # Bins for Creatinine
    labels = ['Normal', 'High', 'Critical']
else:
    print("Warning: No clinical bins defined for this lab. Using generic quartiles.")
    # Fallback for other labs: split data into 4 equal-sized groups
    _, bins = pd.qcut(y_clean, 4, retbins=True, duplicates='drop')
    labels = ['Quartile 1', 'Quartile 2', 'Quartile 3', 'Quartile 4']

print(f"\nBinning data for {column_name} using ranges: {bins}")
df_binned = pd.DataFrame({
    'value': y_clean,
    'embedding_idx': range(len(y_clean)), # Use a simple range index
    'bin': pd.cut(y_clean, bins=bins, labels=labels, right=False)
})

# Filter out any values that didn't fall into a bin
df_binned.dropna(subset=['bin'], inplace=True)

# Calculate Centroids
print("\n--- Centroid Analysis ---")
centroids = {}
for bin_name in labels:
    # Get the indices of the embeddings that belong to this bin
    indices_in_bin = df_binned[df_binned['bin'] == bin_name]['embedding_idx'].values
    
    if len(indices_in_bin) > 0:
        # Get the corresponding embeddings from the clean data
        embeddings_in_bin = X_clean[indices_in_bin]
        # Calculate the mean vector (the centroid)
        centroids[bin_name] = embeddings_in_bin.mean(axis=0)
        print(f"  Calculated centroid for bin '{bin_name}' from {len(indices_in_bin)} samples.")
    else:
        print(f"  WARNING: No samples found for bin '{bin_name}'. It will be skipped.")

# Measure Distances between available centroids
print("\n--- Cosine Distance Between Clinical Range Centroids ---")
# Use combinations to automatically compare all available pairs
from itertools import combinations

if len(centroids) < 2:
    print("  Not enough centroids to calculate distances.")
else:
    for (bin1, centroid1), (bin2, centroid2) in combinations(centroids.items(), 2):
        dist = cosine(centroid1, centroid2)
        print(f"  Distance ({bin1} vs. {bin2}): {dist:.4f}")

# Hypothesis: The distance between Normal and Critical should be the largest.
# The distance between High and Critical might be larger than Normal vs. High,
# showing the model learned the non-linear jump in severity.