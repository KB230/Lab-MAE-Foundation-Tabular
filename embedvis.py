import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import umap

# Load the embeddings data
file_path = '100_Labs_Train_0.25Mask_L_V3/embeddings_test.csv'
embeddings_df = pd.read_csv(file_path)

# Convert the embeddings from string to list of floats
def convert_to_float_list(x):
    if pd.isna(x):
        return np.nan  # Return NaN if the value is NaN
    try:
        return np.array(eval(x))
    except Exception as e:
        print(f"Conversion failed for value: {x} with error: {e}")
        return np.nan

# Apply the conversion to the embedding columns
for col in embeddings_df.columns[:5]:  # Adjust this range if needed
    embeddings_df[col] = embeddings_df[col].apply(convert_to_float_list)

# Collect valid embeddings, skipping NaNs and calculating the maximum length
embeddings_list = []
max_length = 0
for _, row in embeddings_df.iloc[:, :5].iterrows():
    valid_row = [x for x in row if isinstance(x, np.ndarray) and not np.isnan(x).any()]
    if valid_row:  # Use rows that have valid data
        concatenated = np.concatenate(valid_row)
        embeddings_list.append(concatenated)
        max_length = max(max_length, len(concatenated))

# Pad the embeddings to have the same length
padded_embeddings = [np.pad(embedding, (0, max_length - len(embedding)), mode='constant', constant_values=np.nan) for embedding in embeddings_list]

# Convert the list to a numpy array
embeddings = np.array(padded_embeddings)

# Apply scaling to the embeddings before UMAP
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(np.nan_to_num(embeddings))

# Apply UMAP to reduce dimensions to 2
umap_model = umap.UMAP(n_components=2, random_state=42)
embeddings_2d_umap = umap_model.fit_transform(embeddings_scaled)

# Assuming the dataset has 'npval_50971' column
npval = embeddings_df['npval_50971']  # Replace with the actual 'npval' column name

npval_numeric = npval.apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan)

# Create a DataFrame for UMAP visualization
umap_df = pd.DataFrame(data=embeddings_2d_umap, columns=['UMAP Component 1', 'UMAP Component 2'])
umap_df['npval'] = npval_numeric

# Visualize the UMAP results using Plotly
fig_umap = px.scatter(umap_df, x='UMAP Component 1', y='UMAP Component 2', color='npval',
                      title='UMAP Visualization of Embeddings colored by npval')

# Show the interactive plots
fig_umap.show()
