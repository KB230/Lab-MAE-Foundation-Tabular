import pandas as pd
from sklearn.model_selection import train_test_split
from MAEImputer import ReMaskerStep
import numpy as np

# ======================= DEV SETTINGS =======================
# All the parameters you might want to change are here at the top.
#
# --- File Paths ---
INPUT_FILE = 'X_train_eICU_dev.csv' # The output from your preprocessing script
SAVE_PATH = 'eICU_Test_Run' # A new folder to save the test model's checkpoints

# --- Training Parameters ---
# For a quick test, 1-3 epochs is perfect.
MAX_EPOCHS = 2
BATCH_SIZE = 128 # You might use a smaller batch size on a local machine
MASK_RATIO = 0.75 # The paper uses a high mask ratio

# --- Model Architecture ---
# These should match the model you want to train
EMBED_DIM = 64
DEPTH = 8
DECODER_DEPTH = 4
NUM_HEADS = 8
MLP_RATIO = 4.0
# =============================================================

# --- 1. Load Your Preprocessed Data ---
print(f"Loading preprocessed data from: {INPUT_FILE}")
# We set the first column (patientunitstayid) as the index.
df = pd.read_csv(INPUT_FILE, index_col=0)
print(f'Data shape: {df.shape}')


# --- 2. Create Train-Validation Split ---
# We split our single dev file into a set for training and a smaller set for validation.
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)


# --- 3. Create Imputer Instance ---
# The number of columns in your data is the `dim` for the model.
columns = train_df.shape[1]

print("Initializing model to train from scratch...")
imputer = ReMaskerStep(
    dim=columns, 
    mask_ratio=MASK_RATIO, 
    max_epochs=MAX_EPOCHS, 
    save_path=SAVE_PATH, 
    batch_size=BATCH_SIZE,
    embed_dim=EMBED_DIM, 
    depth=DEPTH, 
    decoder_depth=DECODER_DEPTH, 
    num_heads=NUM_HEADS, 
    mlp_ratio=MLP_RATIO,
    weigths=None  # CRITICAL: Set to None to train from scratch, not fine-tune.
)


# --- 4. Train the Model (The Moment of Truth) ---
# The .fit() method expects DataFrames without any extra ID columns, which we've
# already handled by setting the index.
print(f"\nStarting training for {MAX_EPOCHS} epochs...")
imputer.fit(train_df, val_df)


# --- 5. Done! ---
# We will skip the complex evaluation from the original script for this test.
# If the script gets here without crashing, the pipeline works!
print("\n-----------------------------------------")
print("Training test completed successfully!")
print("Check the folder '{}' for model checkpoints.".format(SAVE_PATH))
print("-----------------------------------------")