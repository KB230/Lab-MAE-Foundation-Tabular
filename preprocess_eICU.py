import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

# ======================= DEVELOPMENT SWITCH =======================
# Set to a number (e.g., 500000) for a fast local run on a subset.
# Set to None to run on the entire 2.4GB file.
DEV_MODE_ROWS = 500000
# ==================================================================


# --- Step 1: Load and Filter ---
cols_to_load = ['patientunitstayid', 'labname', 'labresult', 'labresultoffset'] # Assumes labresult offset is the time

if DEV_MODE_ROWS:
    print(f"--- DEV MODE: Loading only the first {DEV_MODE_ROWS} rows for testing. ---")
    df_lab = pd.read_csv('../eICU-data/lab.csv', usecols=cols_to_load, nrows=DEV_MODE_ROWS)
else:
    print("--- PRODUCTION MODE: Loading the full dataset. ---")
    df_lab = pd.read_csv('../eICU-data/lab.csv', usecols=cols_to_load)



TOP_N_LABS = 100 # just to test the training
top_labs = df_lab['labname'].value_counts().nlargest(TOP_N_LABS).index.tolist()
df_filtered = df_lab[df_lab['labname'].isin(top_labs)].copy()

df_filtered['labresult'] = pd.to_numeric(df_filtered['labresult'], errors='coerce')
df_filtered = df_filtered[df_filtered['labresult'] >= 0]
assert isinstance(df_filtered, pd.DataFrame)
df_filtered.dropna(subset=['labresult', 'labresultoffset'], inplace=True)

# --- Step 2: Aggregate by Day ---
# Create a 'day_number' based on the offset (1440 minutes = 24 hours)
df_filtered['day_number'] = df_filtered['labresultoffset'] // 1440

print("Aggregating lab events into daily averages...")
# Group by patient, day, and lab test, and calculate the mean for that day.

daily_agg = df_filtered.groupby(['patientunitstayid', 'day_number', 'labname']).agg(
    labresult=('labresult', 'mean'),
    labresultoffset=('labresultoffset', 'mean')
).reset_index()

# Sort to ensure the 'shift' operation works correctly within each patient's timeline
daily_agg.sort_values(['patientunitstayid', 'day_number'], inplace=True)

# For each patient, pull up the data from the next day (-1 means shift up by 1 row)
daily_agg['labresult_last'] = daily_agg.groupby('patientunitstayid')['labresult'].shift(-1)
daily_agg['labresultoffset_last'] = daily_agg.groupby('patientunitstayid')['labresultoffset'].shift(-1)

# At this point, the last day for each patient will have NaN for the 'last' columns, which is correct.


# --- Step 4: Pivot into Final Wide Format ---
print("Pivoting data into the final wide format...")
# Now we can do a single, powerful pivot
df_pivoted = daily_agg.pivot_table(
    index=['patientunitstayid', 'day_number'],
    columns='labname',
    values=['labresult', 'labresultoffset', 'labresult_last', 'labresultoffset_last']
)

# --- Step 5: Flatten and Rename Columns ---
# The pivot creates multi-level columns, e.g., ('labresult', 'potassium'). We flatten them.
print("Flattening and renaming columns...")
df_pivoted.columns = [f'{val_type}_{lab_name}' for val_type, lab_name in df_pivoted.columns]
# Rename 'labresult' to 'npval' and 'labresultoffset' to 'nptime' to match original format
df_pivoted.columns = df_pivoted.columns.str.replace('labresult_last', 'npval_last')
df_pivoted.columns = df_pivoted.columns.str.replace('labresult', 'npval')
df_pivoted.columns = df_pivoted.columns.str.replace('labresultoffset_last', 'nptime_last')
df_pivoted.columns = df_pivoted.columns.str.replace('labresultoffset', 'nptime')

# --- Step 6: Apply Quantile Normalization ---
print("Applying Quantile Normalization...")
# We only want to normalize the value columns, not the time columns
val_cols = [col for col in df_pivoted.columns if 'npval' in col]
qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
df_pivoted[val_cols] = qt.fit_transform(df_pivoted[val_cols])

# --- Step 7: Save the Final Data ---
# Reset index to make patientid and day_number regular columns for saving
df_final = df_pivoted.reset_index()

file_suffix = '_daily_dev' if DEV_MODE_ROWS else '_daily'
output_filename = f'X_train_eICU{file_suffix}.csv'
df_final.to_csv(output_filename, index=False) # index=False because IDs are now columns

print(f"\nPreprocessing complete! Saved to {output_filename}")
print("Final data shape:", df_final.shape)