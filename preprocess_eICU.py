import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

# ======================= DEVELOPMENT SWITCH =======================
# Set to a number (e.g., 500000) for a fast local run on a subset.
# Set to None to run on the entire 2.4GB file.
DEV_MODE_ROWS = 500000
# ==================================================================


# --- Step 1: Load and Filter ---
# We use the DEV_MODE_ROWS switch here.
cols_to_load = ['patientunitstayid', 'labname', 'labresult', 'labresultoffset']

if DEV_MODE_ROWS:
    print(f"--- DEV MODE: Loading only the first {DEV_MODE_ROWS} rows for testing. ---")
    df_lab = pd.read_csv('../eICU-data/lab.csv', usecols=cols_to_load, nrows=DEV_MODE_ROWS)
else:
    print("--- PRODUCTION MODE: Loading the full dataset. This may take a while... ---")
    df_lab = pd.read_csv('../eICU-data/lab.csv', usecols=cols_to_load)


# Curation Step
TOP_N_LABS = 100
top_labs = df_lab['labname'].value_counts().nlargest(TOP_N_LABS).index.tolist()
print(f"Top {TOP_N_LABS} labs selected.")
df_filtered = df_lab[df_lab['labname'].isin(top_labs)].copy()

# Initial Cleaning
df_filtered['labresult'] = pd.to_numeric(df_filtered['labresult'], errors='coerce')
df_filtered = df_filtered[df_filtered['labresult'] >= 0]
assert isinstance(df_filtered, pd.DataFrame)
df_filtered.dropna(subset=['labresult', 'labresultoffset'], inplace=True)


# --- Step 2: Find First and Follow-up Values using rank() ---
print("Ranking events to find first and follow-up tests...")
df_filtered.sort_values(['patientunitstayid', 'labname', 'labresultoffset'], inplace=True)
df_filtered['time_rank'] = df_filtered.groupby(['patientunitstayid', 'labname'])['labresultoffset'].rank(method='first', ascending=True)

df_first = df_filtered[df_filtered['time_rank'] == 1.0].copy()
df_followup = df_filtered[df_filtered['time_rank'] == 2.0].copy()


# --- Step 3: Pivot the DataFrames ---
print("Pivoting data to create wide format...")
df_val_first = df_first.pivot_table(index='patientunitstayid', columns='labname', values='labresult')
df_time_first = df_first.pivot_table(index='patientunitstayid', columns='labname', values='labresultoffset')
df_val_followup = df_followup.pivot_table(index='patientunitstayid', columns='labname', values='labresult')
df_time_followup = df_followup.pivot_table(index='patientunitstayid', columns='labname', values='labresultoffset')


# --- Step 4: Apply Quantile Normalization ---
print("Applying Quantile Normalization...")
qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)

if not df_val_first.empty:
    df_val_first[:] = qt.fit_transform(df_val_first)
if not df_val_followup.empty:
    df_val_followup[:] = qt.fit_transform(df_val_followup)


# --- Step 5: Combine, Rename, and Reorder ---
print("Combining, renaming, and saving final data...")
df_val_first.columns = ['npval_' + str(col) for col in df_val_first.columns]
df_time_first.columns = ['nptime_' + str(col) for col in df_time_first.columns]
df_val_followup.columns = ['npval_last_' + str(col) for col in df_val_followup.columns]
df_time_followup.columns = ['nptime_last_' + str(col) for col in df_time_followup.columns]

df_final = pd.concat([df_val_first, df_time_first, df_val_followup, df_time_followup], axis=1)

interleaved_cols = []
for lab in top_labs:
    col_map = {
        'val': f'npval_{lab}', 'time': f'nptime_{lab}',
        'val_last': f'npval_last_{lab}', 'time_last': f'nptime_last_{lab}'
    }
    for col in col_map.values():
        if col in df_final.columns:
            interleaved_cols.append(col)

df_final = df_final[interleaved_cols]

# We can also add a suffix to the output file to avoid confusion
file_suffix = '_dev' if DEV_MODE_ROWS else ''
output_filename = f'X_train_eICU{file_suffix}.csv'
df_final.to_csv(output_filename)

print(f"\nPreprocessing complete! Saved to {output_filename}")
print("Final data shape:", df_final.shape)