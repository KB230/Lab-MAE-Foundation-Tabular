import pandas as pd

# Load the MIMIC dictionary file
try:
    df_dict = pd.read_csv('../data/d_labitems.csv')
except FileNotFoundError:
    print("Error: d_labitems.csv not found. Please place it in the '../data/' directory.")
    exit()


search_terms = ['potassium', 'creatinine', 'troponin i', 'troponin t', 'inr']

print("--- Lab ID Lookup Results ---")

for term in search_terms:
    # Use .str.contains for flexible searching, case=False to ignore capitalization
    results = df_dict[df_dict['label'].str.contains(term, case=False, na=False)]
    
    print(f"\nResults for '{term}':")
    if not results.empty:
        # Print the relevant columns
        print(results[['itemid', 'label', 'fluid', 'category']])
    else:
        print("  No direct matches found.")

# Note: INR is often part of the 'PT' test, so let's search for that too.
print("\n--- Searching for 'Prothrombin Time' as a proxy for INR ---")
pt_results = df_dict[df_dict['label'].str.contains('Prothrombin Time', case=False, na=False)]
print(pt_results[['itemid', 'label', 'fluid', 'category']])