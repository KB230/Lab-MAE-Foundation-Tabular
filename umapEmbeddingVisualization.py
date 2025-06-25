import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import umap
import os

# ======================= DEVELOPMENT SWITCH =======================
# Set to True to load the '_dev' files from a partial run.
DEV_MODE = True
# ==================================================================

# --- Configuration ---
save_path = '100_Labs_Train_0.25Mask_L_V3'
original_data_file = '../data/X_test.csv'
file_suffix = '_dev' if DEV_MODE else ''
cls_embeddings_file = os.path.join(save_path, f'test_cls_embeddings{file_suffix}.npy')

# --- 1. Define the EXACT same cleaning function from your other script ---
# (This is crucial for recreating the data state perfectly)
def clean_missing(df, threshold=20 + 3, missing_per_col=100, cols_to_remove=None):
    # Make an explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Remove rows with less than 'threshold' non-null values
    df_filtered = df.dropna(thresh=threshold)
    
    # The column removal logic from your script (can be kept or simplified if not needed for this step)
    if type(cols_to_remove) != list:
        if missing_per_col and not cols_to_remove:
            columns_all_nan = df_filtered.columns[df_filtered.notna().sum() < missing_per_col].tolist()
            ids = ['_' + col.split('_')[-1] for col in columns_all_nan]
            def ids_in_string(value_list, target_string):
                for value in value_list:
                    if value in target_string: return True
                return False
            cols_to_remove = [col for col in df_filtered.columns if ids_in_string(ids, col)]
    
    df_filtered.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    return df_filtered, cols_to_remove


# --- 2. Load Embeddings and Re-filter Original Data ---
if DEV_MODE:
    print("--- DEV MODE: Loading partial '_dev' files. ---")
else:
    print("--- PRODUCTION MODE: Loading full dataset files. ---")

print(f"Loading embeddings from: {cls_embeddings_file}")
cls_embeddings = np.load(cls_embeddings_file)

# Load the full original data that corresponds to the dev/prod run
num_rows_to_read = 10000 if DEV_MODE else None
print(f"Loading original data from: {original_data_file}")
df_original = pd.read_csv(original_data_file, nrows=num_rows_to_read)

# Re-run the exact same cleaning step to get a perfectly aligned DataFrame
print("Re-filtering original data to match the embeddings...")
df_viz, _ = clean_missing(df_original, threshold=20 + 3, missing_per_col=500, cols_to_remove=[])


# --- 3. Sanity Check (The Final Word) ---
print(f"Embeddings loaded shape: {cls_embeddings.shape}")
print(f"Visualization DF shape: {df_viz.shape}")

if len(cls_embeddings) != len(df_viz):
    print("\n\nCRITICAL ERROR! Shape mismatch after re-filtering.")
    print("Ensure the `clean_missing` function here is IDENTICAL to the one in your extraction script.")
    exit()
else:
    print("Shapes match perfectly. Proceeding to UMAP.")


# --- 4. Scale and Run UMAP ---
print("Scaling data...")
# We need to handle potential NaNs in the color column before plotting
color_column = 'npval_50971'
df_viz[color_column] = pd.to_numeric(df_viz[color_column], errors='coerce')

scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(cls_embeddings)

print("Running UMAP...")
umap_model = umap.UMAP(n_components=2, random_state=42)
embeddings_2d_umap = umap_model.fit_transform(embeddings_scaled)

# --- 5. Visualize ---
print("Creating visualization...")
df_viz['UMAP Component 1'] = embeddings_2d_umap[:, 0]
df_viz['UMAP Component 2'] = embeddings_2d_umap[:, 1]

title = f'UMAP Visualization of CLS Embeddings ({ "Dev" if DEV_MODE else "Full"} Data)'
fig_umap = px.scatter(
    df_viz,
    x='UMAP Component 1',
    y='UMAP Component 2',
    color=color_column,
    hover_data=['hadm_id'],
    title=title
)

fig_umap.show()
print("Done.")