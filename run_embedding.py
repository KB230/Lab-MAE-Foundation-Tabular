import pandas as pd
from sklearn.model_selection import train_test_split
from MAEImputer import ReMaskerStep
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from math import sqrt
import os
import torch

'''
################ Read Dataset ################
df_test = pd.read_csv('../data/X_test.csv')
print(f'Test values shape: {df_test.shape}')
'''

# ======================= DEVELOPMENT SWITCH =======================
# Set to a number (e.g., 5000) to run on a small subset for fast testing.
# Set to None to run on the entire file.
DEV_MODE_ROWS = 10000 
# ==================================================================

# --- Configuration ---
CHUNK_SIZE = 2048 # A good chunk size for your 32GB RAM system
save_path = '100_Labs_Train_0.25Mask_L_V3'
input_data_path = '../data/X_test.csv'
eval_batch_size = 256 # Batch size for inference

# --- Load initial data ---
print("Loading initial data...")
if DEV_MODE_ROWS:
    print(f"--- DEV MODE: Processing only the first {DEV_MODE_ROWS} rows. ---")
    df_test = pd.read_csv(input_data_path, nrows=DEV_MODE_ROWS)
else:
    print("--- PRODUCTION MODE: Processing the full dataset. ---")
    df_test = pd.read_csv(input_data_path)

print(f'Test values shape: {df_test.shape}')

################ Clean Missing Data ################
def clean_missing(df, threshold=20 + 3, missing_per_col=100, cols_to_remove=None):
    # Remove rows with less than 20 values
    df = df.dropna(thresh=threshold)
    print(f"DataFrame after removing rows with at least 20 missing values: {df.shape}")
    
    if type(cols_to_remove) != list:
        if missing_per_col and not cols_to_remove:
            # Get columns where at least 100 values are not missing
            columns_all_nan = df.columns[df.notna().sum() < missing_per_col].tolist()
            # Identify columns that end with a number after the last underscore
            ids = ['_' + col.split('_')[-1] for col in columns_all_nan]

            def ids_in_string(value_list, target_string):
                for value in value_list:
                    if value in target_string:
                        return True
                return False

            cols_to_remove = []
            for column in df.columns:
                if ids_in_string(ids, column):
                    cols_to_remove.append(column)

    print(f'Removing columns: {cols_to_remove}')

    df.drop(columns=cols_to_remove, inplace=True)
    
    return df, cols_to_remove

print("Cleaning data and removing rows with excessive missing values...")
df_test, _ = clean_missing(df_test, threshold=20 + 3, missing_per_col=500, cols_to_remove=[])
print(f'Shape after cleaning: {df_test.shape}') # This will now show the smaller number (e.g., 9661)


# Create a list of columns to ignore
columns_ignore = ['first_race', 'chartyear', 'hadm_id']


################ Create Imputer Instance ################
columns = df_test.shape[1] - 3 # + 3 because of: first_race, chartyear, hadm_id
mask_ratio = 0.25
max_epochs = 300
save_path = '100_Labs_Train_0.25Mask_L_V3'
weigths = '100_Labs_Train_0.25Mask_L_V3/model_checkpoint.zip'


batch_size=256 
embed_dim=64
depth=8
decoder_depth=4
num_heads=8
mlp_ratio=4.0


imputer = ReMaskerStep(dim=columns, mask_ratio=mask_ratio, max_epochs=max_epochs, save_path=save_path, batch_size=batch_size,
                      embed_dim=embed_dim, depth=depth, decoder_depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      weigths=weigths)


with open('100_Labs_Train_0.25Mask_L_V3/norm_parameters.pkl', 'rb') as file:
    loaded_norm_parameters = pickle.load(file)
    
imputer.norm_parameters = loaded_norm_parameters

###### Extract Embeddings with Batch-Processing #######
print("Extracting embeddings...")
input_df = df_test.drop(columns=['first_race', 'chartyear', 'hadm_id'])
# Call the function once. It will loop through the data in batches for us.
# It now returns two values, thanks to our fix in Step 1.
final_feature_embeddings_t, final_cls_embeddings_t = imputer.extract_embeddings(
    X_raw=input_df, 
    eval_batch_size=eval_batch_size
)


# Convert the final tensors to numpy arrays
final_feature_embeddings = final_feature_embeddings_t.numpy()
final_cls_embeddings = final_cls_embeddings_t.numpy()

print(f"Final feature embeddings shape: {final_feature_embeddings.shape}")
print(f"Final CLS (row) embeddings shape: {final_cls_embeddings.shape}")

# --- Save the outputs ---
file_suffix = '_dev' if DEV_MODE_ROWS else ''
feature_embedding_path = os.path.join(save_path, f'test_feature_embeddings{file_suffix}.npy')
cls_embedding_path = os.path.join(save_path, f'test_cls_embeddings{file_suffix}.npy')
ids_path = os.path.join(save_path, f'test_ids{file_suffix}.csv')

print(f"Saving feature embeddings to {feature_embedding_path}")
np.save(feature_embedding_path, final_feature_embeddings)
print(f"Saving CLS embeddings to {cls_embedding_path}")
np.save(cls_embedding_path, final_cls_embeddings)
print(f"Saving IDs to {ids_path}")
df_test[['hadm_id']].to_csv(ids_path, index=False)

print("Script finished successfully.")

'''
################ Extract embeddings ################

eval_batch_size = 256

print('Extracting embeddings...')
embeddings = imputer.extract_embeddings(df_test.drop(columns=['first_race', 'chartyear', 'hadm_id']), eval_batch_size=eval_batch_size).cpu().numpy()


print(f'Converting embeddings to csv...')
# Convert the embeddings to csv:
embeddings_df = df_test.copy()

# Convert the first 400 columns to dtype `object` to store lists
embeddings_df.iloc[:, :400] = embeddings_df.iloc[:, :400].astype(object)

# Replace the values in the first 400 columns with the corresponding embeddings
for i, index in enumerate(embeddings_df.index):
    
    if i % 100 == 0:
        print(f'{i} embeddings processed')
        
    print(f'Row: {i}')
    for j in range(400):
        column = embeddings_df.columns[j]
        if not pd.isna(embeddings_df.loc[index, column]):  # Check if the value is not NaN
            embeddings_df.at[index, column] = embeddings[i, j].tolist()


embeddings_df.to_csv(os.path.join(save_path, 'embeddings_test.csv'), index=False)
'''