import pandas as pd
import numpy as np
import os
import torch
import h5py # Import the HDF5 library
from MAEImputer import ReMaskerStep

# ======================= CONFIGURATION =======================
# Use the full dataset by setting nrows=None
# Or use a large number like 250000 to match your test
INPUT_FILE = '../data/X_test.csv'
N_ROWS_TO_PROCESS = 250000 

# The output file will now be a single HDF5 file
OUTPUT_H5_FILE = 'embeddings_250k.h5' 

# Process the data in chunks of this size to keep memory low
CHUNK_SIZE = 4096 
# =================================================================

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
df_original_raw = pd.read_csv(INPUT_FILE, nrows=N_ROWS_TO_PROCESS)
df_aligned, _ = clean_missing(df_original_raw)
print(f"Data alignment successful. Final shape to process: {df_aligned.shape}")

# --- 2. Initialize the Imputer ---
# You need to load the MIMIC-trained model checkpoint here
MIMIC_CHECKPOINT_PATH = '100_Labs_Train_0.25Mask_L_V3/model_checkpoint.zip' # Example path
# Get architecture params from the original project
columns = df_aligned.drop(columns=['hadm_id', 'chartyear', 'first_race']).shape[1]
imputer = ReMaskerStep(dim=columns, weigths=MIMIC_CHECKPOINT_PATH)
imputer.load_norm_parameters('100_Labs_Train_0.25Mask_L_V3/norm_parameters.pkl') # Example path

# --- 3. Process Data in Chunks and Save to HDF5 ---
print(f"Starting chunk-based embedding generation. Results will be saved to {OUTPUT_H5_FILE}")

# Delete the file if it exists to start fresh
if os.path.exists(OUTPUT_H5_FILE):
    os.remove(OUTPUT_H5_FILE)

num_rows = len(df_aligned)
# This is the main loop
for i in range(0, num_rows, CHUNK_SIZE):
    chunk_end = min(i + CHUNK_SIZE, num_rows)
    print(f"Processing rows {i} to {chunk_end-1}...")

    # Get a chunk of the dataframe
    chunk_df = df_aligned.iloc[i:chunk_end]
    # Prepare the feature-only data for the imputer
    chunk_features = chunk_df.drop(columns=['hadm_id', 'chartyear', 'first_race'])
    
    # Run inference ONLY on this chunk
    feature_embeddings_chunk, cls_embeddings_chunk = imputer.extract_embeddings(chunk_features)

    # Save the chunk directly to the HDF5 file
    with h5py.File(OUTPUT_H5_FILE, 'a') as hf:
        # --- Save Feature Embeddings ---
        if 'feature_embeddings' not in hf:
            # For the first chunk, create the dataset
            hf.create_dataset(
                'feature_embeddings', 
                data=feature_embeddings_chunk.cpu().numpy(), 
                maxshape=(None, feature_embeddings_chunk.shape[1], feature_embeddings_chunk.shape[2]), 
                chunks=True
            )
        else:
            # For subsequent chunks, resize and append
            dset = hf['feature_embeddings']
            dset.resize((dset.shape[0] + feature_embeddings_chunk.shape[0]), axis=0)
            dset[-feature_embeddings_chunk.shape[0]:] = feature_embeddings_chunk.cpu().numpy()

        # --- Save CLS Embeddings ---
        if 'cls_embeddings' not in hf:
            hf.create_dataset(
                'cls_embeddings', 
                data=cls_embeddings_chunk.cpu().numpy(), 
                maxshape=(None, cls_embeddings_chunk.shape[1]), 
                chunks=True
            )
        else:
            dset = hf['cls_embeddings']
            dset.resize((dset.shape[0] + cls_embeddings_chunk.shape[0]), axis=0)
            dset[-cls_embeddings_chunk.shape[0]:] = cls_embeddings_chunk.cpu().numpy()

# Finally, save the aligned IDs so you can match them later
aligned_ids_path = 'aligned_ids_250k.csv'
df_aligned[['hadm_id']].to_csv(aligned_ids_path, index=False)
print(f"Saved corresponding IDs to {aligned_ids_path}")

print("\n--- Embedding Generation Complete! ---")