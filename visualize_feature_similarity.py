import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import umap
import os

# ======================= DEVELOPMENT SWITCH =======================
# Set to True to use the '_dev' files.
DEV_MODE = True
# ==================================================================

# --- Configuration ---
save_path = '100_Labs_Train_0.25Mask_L_V3'
original_data_file = '../data/X_test.csv' # We need this for the column names

file_suffix = '_dev' if DEV_MODE else ''
feature_embeddings_file = os.path.join(save_path, f'test_feature_embeddings{file_suffix}.npy')

# --- 1. Load Data ---
if DEV_MODE:
    print("--- DEV MODE: Loading partial '_dev' files. ---")
else:
    print("--- PRODUCTION MODE: Loading full dataset files. ---")

print(f"Loading feature embeddings from: {feature_embeddings_file}")
# This array has the shape (num_patients, num_features, embedding_dim)
feature_embeddings = np.load(feature_embeddings_file)
print(f"Original feature embeddings shape: {feature_embeddings.shape}")

# Load the original data just to get the column names
num_rows_to_read = 10000 if DEV_MODE else None
df_original = pd.read_csv(original_data_file, nrows=num_rows_to_read)
# We need to get the feature names in the correct order, excluding ID columns
feature_names = [col for col in df_original.columns if col not in ['first_race', 'chartyear', 'hadm_id']]
# A check to make sure the number of features match
assert feature_embeddings.shape[1] == len(feature_names), "Mismatch between number of features in embeddings and column names!"


# --- 2. Calculate the Average Embedding for Each Lab Test ---
print("Averaging embeddings across all patients to get a single vector per lab test...")
# We use np.nanmean to gracefully handle any NaN values that might be in the embeddings.
# axis=0 tells numpy to average across the first dimension (the patients).
mean_feature_embeddings = np.nanmean(feature_embeddings, axis=0)
print(f"Mean feature embeddings shape: {mean_feature_embeddings.shape}") # Should be (num_features, embedding_dim)


# --- 3. Run UMAP on the Mean Embeddings ---
print("Scaling and running UMAP on the mean feature embeddings...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(mean_feature_embeddings)

umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_2d_umap = umap_model.fit_transform(embeddings_scaled)


# --- 4. Create Visualization DataFrame and Plot ---
print("Creating visualization...")
# Create a DataFrame to hold our results for plotting
df_viz = pd.DataFrame({
    'lab_test': feature_names,
    'UMAP Component 1': embeddings_2d_umap[:, 0],
    'UMAP Component 2': embeddings_2d_umap[:, 1]
})

title = 'UMAP Visualization of Lab Value (Feature) Similarity'
fig = px.scatter(
    df_viz,
    x='UMAP Component 1',
    y='UMAP Component 2',
    # We use 'text' to label the points directly on the plot
    text='lab_test',
    # 'hover_data' shows info when you mouse over a point
    hover_name='lab_test',
    title=title
)

# This makes the text labels easier to read
fig.update_traces(textposition='top center')
fig.update_layout(
    height=800, # Make the plot taller to prevent text overlap
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2"
)

fig.show()
print("Done.")