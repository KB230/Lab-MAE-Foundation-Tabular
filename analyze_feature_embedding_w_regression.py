import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
FEATURE_EMBEDDINGS_FILE = '100_Labs_Train_0.25Mask_L_V3/test_feature_embeddings_dev.npy'
PREPROCESSED_DATA_FILE = '../data/X_test.csv'
BASE_OUTPUT_DIR = 'analysis_results'
DEV_MODE_ROWS = 10000
# 50971 = Potassium, 50912 = Creatinine, 51237 = INR
TARGET_LAB_ITEMID = 51237
NUM_INFLUENTIAL_DIMS= 5
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


X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Check if there's any data left to train on
if len(X_train) == 0:
    print("\nFATAL ERROR: No valid data remains after cleaning. Cannot train model.")
    exit()


# --- 3. Train Model and Get Coefficients ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
coefficients = model.coef_

# --- 4. Analyze and Print Results ---
print(f"\n--- ANALYSIS RESULTS for '{column_name}' ---")
print(f"  R-squared (R2 Score): {r2:.4f}")
print(f"  Linear Model Intercept: {model.intercept_:.4f}")
# ============================ ADDED THIS LINE BACK ============================
print(f"  All Coefficients {coefficients}")
# ============================================================================


# --- 5. Analyze Prediction Bands (As Requested) ---
print("\n--- Analysis of Prediction Bands ---")
df_results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
band_analysis = df_results.groupby('actual')['predicted'].agg(['mean', 'std']).reset_index()
print("Summary of predictions for each actual value band:")
print(band_analysis.round(4))

# --- 6. Analyze Influential Dimensions ---
influential_indices = np.argsort(np.abs(coefficients))[::-1]

print(f"\nTop {NUM_INFLUENTIAL_DIMS} Most Influential Embedding Dimensions:")
print("-" * 50)
print(f"{'Dimension':<12} | {'Coefficient Value':<20}")
print("-" * 50)
for i in influential_indices[:NUM_INFLUENTIAL_DIMS]:
    print(f"Dim {i:<8} | {coefficients[i]:<20.4f}")
print("-" * 50)



# --- 7. Setup Output Directory and Generate Plots ---
print("Generating and saving individual plots...")

# Create a specific directory for this lab's results
lab_output_dir = os.path.join(BASE_OUTPUT_DIR, column_name)
os.makedirs(lab_output_dir, exist_ok=True)

# --- Plot 1: Regression Performance ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([band_analysis['actual'].min(), band_analysis['actual'].max()], 
        [band_analysis['actual'].min(), band_analysis['actual'].max()], 
        '--r', linewidth=2, label='Perfect Prediction')
ax.plot(band_analysis['actual'], band_analysis['mean'], 'o', color='royalblue', label='Mean Prediction')
ax.fill_between(
    band_analysis['actual'],
    band_analysis['mean'] - band_analysis['std'],
    band_analysis['mean'] + band_analysis['std'],
    color='royalblue', alpha=0.2, label='Std. Dev.'
)
ax.set_title(f"Regression Performance for {column_name} (R2: {r2:.3f})")
ax.set_xlabel("Actual Values (Ground Truth)")
ax.set_ylabel("Predicted Values")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(lab_output_dir, '1_regression_performance.png'), dpi=150)
plt.show() # Display the plot

# --- Plot 2: All Coefficient Magnitudes ---
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=range(len(coefficients)), y=coefficients, ax=ax, color='purple')
ax.set_title(f"Learned Coefficients for {column_name}")
ax.set_xlabel("Embedding Dimension (0-63)")
ax.set_ylabel("Coefficient Value")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(lab_output_dir, '2_all_coefficients.png'), dpi=150)
plt.show()

# --- Plots 3 onwards: Deeper Dive into Top N Dimensions ---
influential_indices = np.argsort(np.abs(coefficients))[::-1]
print(f"\nAnalyzing top {NUM_INFLUENTIAL_DIMS} most influential dimensions...")

for i in range(NUM_INFLUENTIAL_DIMS):
    dim_index = influential_indices[i]
    dim_coeff = coefficients[dim_index]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_clean, y=X_clean[:, dim_index], alpha=0.3, ax=ax)
    ax.set_title(f"Behavior of Dim {dim_index} for {column_name} (Coeff: {dim_coeff:.2f})")
    ax.set_xlabel("Actual Lab Value")
    ax.set_ylabel(f"Value in Embedding Dim {dim_index}")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(lab_output_dir, f'3_influential_dim_{dim_index}.png'), dpi=150)
    plt.show()

print(f"\nAnalysis complete. All plots saved to '{lab_output_dir}'")