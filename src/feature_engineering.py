import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from data_loader import load_train_data, SENSOR_COLS, SETTING_COLS

# Sensors with zero or near-zero variance in subset FD001
LOW_VARIANCE_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]

def process_features(df: pd.DataFrame, window_sizes=[5, 10], fit_scaler=True, scaler=None):
    # Drop invariant sensors
    useful_sensors = [s for s in SENSOR_COLS if s not in LOW_VARIANCE_SENSORS]
    
    # Scale feature ranges to [0, 1]
    features_to_scale = useful_sensors + SETTING_COLS
    if fit_scaler:
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        df[features_to_scale] = scaler.transform(df[features_to_scale])
        
    engineered_df = df.copy()
    
    # Compute rolling window statistics grouped by engine unit
    for w in window_sizes:
        # Rolling Averages
        rolling_mean = (
            df.groupby("unit_number")[useful_sensors]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_mean.columns = [f"{col}_mean_w{w}" for col in useful_sensors]
        
        # Rolling Standard Deviation
        rolling_std = (
            df.groupby("unit_number")[useful_sensors]
            .rolling(window=w, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        rolling_std.columns = [f"{col}_std_w{w}" for col in useful_sensors]
        
        # Concatenate features
        engineered_df = pd.concat([engineered_df, rolling_mean, rolling_std], axis=1)
        
    # Replace NaNs from initial rolling windows with zero
    engineered_df = engineered_df.fillna(0)
    
    return engineered_df, scaler

if __name__ == "__main__":
    train_path = "data/raw/train_FD001.txt"
    df_raw = load_train_data(train_path)
    
    print("Processing sensor features and generating rolling windows...")
    df_processed, scaler = process_features(df_raw, window_sizes=[5, 10], fit_scaler=True)
    
    # Ensure target output folders exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Export processed training set
    output_path = "data/processed/train_FD001_processed.csv"
    df_processed.to_csv(output_path, index=False)
    
    # Save fitted scaler for inference
    joblib.dump(scaler, "models/scaler_FD001.pkl")
    
    print(f"Feature engineering finished successfully!")
    print(f"Output saved to: {output_path}")
    print(f"Total features created: {df_processed.shape[1]}")