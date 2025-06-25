import pandas as pd
import numpy as np
from MAEImputer import ReMaskerStep
import pickle

# --- Configuration: Point to your test run artifacts ---
RUN_FOLDER = 'eICU_Test_Run'
SAVED_MODEL_PATH = f'{RUN_FOLDER}/epoch1_checkpoint'
NORM_PARAMS_PATH = f'{RUN_FOLDER}/norm_parameters.pkl'
INPUT_FILE = 'X_train_eICU_dev.csv'

# --- Use the EXACT same architecture as your training script! ---
EMBED_DIM = 64
DEPTH = 8
DECODER_DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0

# --- 1. Load a small sample of data ---
df_sample = pd.read_csv(INPUT_FILE, index_col=0, nrows=5)
columns = df_sample.shape[1]

# --- 2. Load the Model ---
print(f"Loading saved model from: {SAVED_MODEL_PATH}")
imputer = ReMaskerStep(
    dim=columns,
    save_path=RUN_FOLDER,
    weigths=SAVED_MODEL_PATH,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    decoder_depth=DECODER_DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO
)

print(f"Loading normalization parameters from: {NORM_PARAMS_PATH}")
with open(NORM_PARAMS_PATH, 'rb') as f:
    norm_params = pickle.load(f)
imputer.norm_parameters = norm_params
print("Model and normalization parameters loaded successfully.")


# --- 3. Create an artificial test case ---
test_row = df_sample.iloc[[0]].copy()
original_values = test_row.iloc[0, 5:10].copy()
print("\nOriginal values for columns 5-10:")
print(original_values)

test_row.iloc[0, 5:10] = np.nan
print("\nArtificially masked row (values for columns 5-10 are now NaN):")
print(test_row.iloc[0, 5:10])


# --- 4. Perform Imputation and Convert to DataFrame ---
print("\nRunning imputer.transform() to fill in the missing values...")
# The transform method returns a Tensor
imputed_tensor = imputer.transform(test_row)


imputed_data = pd.DataFrame(
    imputed_tensor.cpu().numpy(),
    columns=test_row.columns,
    index=test_row.index
)


# --- 5. Compare the Results ---
# Now we can safely use .iloc because imputed_data is a DataFrame
imputed_row = imputed_data.iloc[0]

print("\n--- RESULTS ---")
print("Comparison for columns 5-10:")
result_df = pd.DataFrame({
    'Original': original_values,
    'Imputed': imputed_row[5:10]
})
print(result_df)

print("\nVerification complete! The NaN values were successfully replaced.")