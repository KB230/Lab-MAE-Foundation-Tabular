import pandas as pd
from sklearn.model_selection import train_test_split
from MAE import MaskedAutoencoder, MaskedAutoencoder_NoPos # this one does not add previous positional embeddings together
import numpy as np
import argparse
from MAEImputer import ReMaskerStep
from functools import partial
import torch.nn as nn 

# --- 1. Setup Command-Line Argument Parser ---
parser = argparse.ArgumentParser(description='Train LabMAE on eICU data.')
parser.add_argument(
    '--model_type', 
    type=str, 
    required=True, 
    choices=['mae', 'no_pos'],
    help='Type of model to train: "mae" (with positional embeddings) or "no_pos" (without).'
)
args = parser.parse_args()


# ======================= PARAMETERS =======================
INPUT_FILE = 'X_train_eICU_daily_dev.csv' # Use the one made with 100 labs
SAVE_PATH = f'eICU_Run_100_Labs_{args.model_type}'
MAX_EPOCHS = 10
BATCH_SIZE = 128
MASK_RATIO = 0.75 

# --- Model Architecture --- (referencing mae_large in MAE.py)
EMBED_DIM = 64
DEPTH = 8
DECODER_EMBED_DIM = 64 
DECODER_DEPTH = 4
NUM_HEADS = 8
DECODER_NUM_HEADS = 4 
MLP_RATIO = 4.0
# =============================================================

# --- 2. Load and Prepare Data ---
print(f"Loading preprocessed data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) 
id_columns = ['patientunitstayid', 'day_number']
X_train = train_df.drop(columns=id_columns) # assumes train_test_split returns arrays instead of dataframes
X_val = val_df.drop(columns=id_columns)
columns = X_train.shape[1]

# --- 3. Initialize the Correct Model (Corrected Version) ---
print(f"\n--- Initializing model of type: {args.model_type.upper()} ---")

if args.model_type == 'mae':
    model_class = MaskedAutoencoder
else: # 'no_pos'
    model_class = MaskedAutoencoder_NoPos
    
# First, create a basic imputer instance. It will create a default model inside
# that we are about to replace, but we need it for its other methods and parameters.
imputer = ReMaskerStep(
    dim=columns, 
    save_path=SAVE_PATH, 
    max_epochs=MAX_EPOCHS, 
    batch_size=BATCH_SIZE,
    mask_ratio=MASK_RATIO, # Pass this here too
    weigths=None
)


norm_layer_partial = partial(nn.LayerNorm, eps=1e-6)

imputer.model = model_class(
    rec_len=columns,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    decoder_embed_dim=DECODER_EMBED_DIM,
    decoder_depth=DECODER_DEPTH,
    decoder_num_heads=DECODER_NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    norm_layer=norm_layer_partial # Pass the partial object here
)
imputer.model.to(imputer.device)

# --- 4. Train the Model ---
# (This part is unchanged)
print(f"Starting training. Checkpoints and logs will be saved in: {SAVE_PATH}")
imputer.fit(X_train, X_val)

print("\n--- Training complete! ---")