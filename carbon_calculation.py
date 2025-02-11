import pandas as pd
import numpy as np
import os
import joblib
import pickle
from MAEImputer import ReMaskerStep
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

# Define locations, including latitude and longitude for Popayán, Colombia
locations = {
    "South America - Popayán, Colombia": {"country_iso_code": "COL", "region": "Cauca"},
    "North America - Cambridge, MA, USA": {"country_iso_code": "USA", "region": "Massachusetts"},
    "Europe - Paris, France": {"country_iso_code": "FRA", "region": "Ile-de-France"},
    "Africa - Uganda, Mbarara": {"country_iso_code": "UGA", "region": None},
    "Asia - Manila, Philippines": {"country_iso_code": "PHL", "region": None},
    "Australia - Sydney, Australia": {"country_iso_code": "AUS", "region": "New South Wales"}
}

################# Helper Functions #################
def load_xgboost_model(filename):
    """Load XGBoost model from file."""
    loaded_model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return loaded_model

################ Load and Prepare Data ################
df_test = pd.read_csv('../data/X_test.csv')
print(f'Test values shape: {df_test.shape}')

columns_ignore = ['first_race', 'chartyear', 'hadm_id']
df_test = df_test.drop(columns=columns_ignore)
df_test = df_test[df_test.notna().sum(axis=1) >= 20]
df_test = df_test.sample(n=64, random_state=42).reset_index(drop=True)


################ Load MAE Imputer ################
columns = df_test.shape[1]
mask_ratio = 0.25
max_epochs = 300
save_path = '100_Labs_Train_0.25Mask_L_V3'
weigths = '100_Labs_Train_0.25Mask_L_V3/epoch390_checkpoint'


batch_size = 256 
embed_dim = 64
depth = 8
decoder_depth = 4
num_heads = 8
mlp_ratio = 4.0

imputer = ReMaskerStep(dim=columns, mask_ratio=mask_ratio, max_epochs=max_epochs, save_path=save_path, batch_size=batch_size,
                      embed_dim=embed_dim, depth=depth, decoder_depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      weigths=weigths)

with open('100_Labs_Train_0.25Mask_L_V3/norm_parameters.pkl', 'rb') as file:
    loaded_norm_parameters = pickle.load(file)
    
imputer.norm_parameters = loaded_norm_parameters




################ Define Inference Function ################
def perform_inference(imputer, xgboost_models, data, batch_sizes):
    for batch_size in batch_sizes:
        print("\n" + "="*50)
        print(f"\nPerforming inference for batch size: {batch_size}")

        # Select a subset for the given batch size
        batch_data = data.iloc[:batch_size]
        print(f"\nBatch Data (Batch Size {batch_size}):")
        print(batch_data.head())
        
        # MAE Model Predictions
        imputed_data = pd.DataFrame(imputer.transform(batch_data).cpu().numpy(), columns=batch_data.columns)
        print(f"\nMAE Imputer Results (Batch Size {batch_size}):")
        print(imputed_data.head())

        # XGBoost Predictions
        for idx, model in enumerate(xgboost_models):
            batch_data_no_target = batch_data.drop(columns=[batch_data.columns[idx*2]])
            predictions = model.predict(batch_data_no_target)
            print(f"\nXGBoost Model {idx} Results (Batch Size {batch_size}):")
            print(predictions[:5])
                
                
                


# Updated Inference Function to Track Emissions per Batch Size
def perform_inference_with_tracking_per_batch(imputer, xgboost_models, data, batch_sizes):
    for batch_size in batch_sizes:
        print("\n" + "="*50)
        print(f"\nPerforming inference for batch size: {batch_size}")

        # Select a subset for the given batch size
        batch_data = data.iloc[:batch_size]

        # Track emissions for MAE
        mae_tracker = EmissionsTracker(project_name=f"MAE Imputer - Batch Size {batch_size}")
        mae_tracker.start()
        imputed_data = pd.DataFrame(imputer.transform(batch_data).cpu().numpy(), columns=batch_data.columns)
        mae_emissions = mae_tracker.stop()
        #print(f"\nMAE Imputer Results (Batch Size {batch_size}):")
        #print(imputed_data.head())
        print(f"\nMAE Total Emissions for Batch Size {batch_size}: {mae_emissions:.4f} kg CO2")

        # Track emissions for XGBoost
        xgboost_tracker = EmissionsTracker(project_name=f"XGBoost Models - Batch Size {batch_size}")
        xgboost_tracker.start()
        for idx, model in enumerate(xgboost_models):
            batch_data_no_target = batch_data.drop(columns=[batch_data.columns[idx * 2]])
            predictions = model.predict(batch_data_no_target)
            #print(f"\nXGBoost Model {idx} Results (Batch Size {batch_size}):")
            #print(predictions[:5])
        xgboost_emissions = xgboost_tracker.stop()
        print(f"\nXGBoost Total Emissions for Batch Size {batch_size}: {xgboost_emissions:.4f} kg CO2")


# Updated Inference Function to Handle Geo-Location
def perform_inference_with_location_tracking(imputer, xgboost_models, data, batch_sizes, locations):
    for location, settings in locations.items():
        print("\n" + "="*50)
        print(f"\nSimulating emissions for location: {location}")
        for batch_size in batch_sizes:
            print(f"\nPerforming inference for batch size: {batch_size}")

            # Select a subset for the given batch size
            batch_data = data.iloc[:batch_size]

            # Track emissions for MAE
            if "geo_location" in settings and settings["geo_location"]:
                mae_tracker = EmissionsTracker(
                    project_name=f"MAE Imputer - {location} - Batch Size {batch_size}"
                )
            else:
                mae_tracker = OfflineEmissionsTracker(
                    project_name=f"MAE Imputer - {location} - Batch Size {batch_size}",
                    country_iso_code=settings.get("country_iso_code"),
                    region=settings.get("region")
                )
            mae_tracker.start()
            imputed_data = pd.DataFrame(imputer.transform(batch_data).cpu().numpy(), columns=batch_data.columns)
            mae_emissions = mae_tracker.stop()
            print(f"\nMAE Total Emissions for {location}, Batch Size {batch_size}: {mae_emissions:.4f} kg CO2")

            # Track emissions for XGBoost
            if "geo_location" in settings and settings["geo_location"]:
                xgboost_tracker = EmissionsTracker(
                    project_name=f"XGBoost Models - {location} - Batch Size {batch_size}"
                )
            else:
                xgboost_tracker = OfflineEmissionsTracker(
                    project_name=f"XGBoost Models - {location} - Batch Size {batch_size}",
                    country_iso_code=settings.get("country_iso_code"),
                    region=settings.get("region")
                )
            xgboost_tracker.start()
            for idx, model in enumerate(xgboost_models):
                batch_data_no_target = batch_data.drop(columns=[batch_data.columns[idx * 2]])
                predictions = model.predict(batch_data_no_target)
            xgboost_emissions = xgboost_tracker.stop()
            print(f"\nXGBoost Total Emissions for {location}, Batch Size {batch_size}: {xgboost_emissions:.4f} kg CO2")


################ Load XGBoost Models ################
xgboost_models = []
for i in range(100):
    model_path = f"/Users/davidrestrepo/MAE Tabular/xgboost/{i}best_model.pkl"
    if not os.path.exists(model_path):
        continue
    xgboost_models.append(load_xgboost_model(model_path))

################ Perform Inference with Carbon Tracking per Batch ################
batch_sizes = [1, 32, 64]
#perform_inference(imputer, xgboost_models, df_test, batch_sizes)
#perform_inference_with_tracking_per_batch(imputer, xgboost_models, df_test, batch_sizes)
perform_inference_with_location_tracking(imputer, xgboost_models, df_test, batch_sizes, locations)
