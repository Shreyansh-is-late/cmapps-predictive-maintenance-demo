import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
import joblib
import os

def train_model():
    # Load processed data
    data_path = "data/processed/train_FD001_processed.csv"
    df = pd.read_csv(data_path)
    
    # Define features and target
    exclude_cols = ["unit_number", "time_cycles", "RUL"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df["RUL"]
    
    # Group-aware split (ensure entire engines stay within train or validation set)
    unique_units = df["unit_number"].unique()
    train_units, val_units = train_test_split(unique_units, test_size=0.2, random_state=42)
    
    X_train = X[df["unit_number"].isin(train_units)]
    y_train = y[df["unit_number"].isin(train_units)]
    X_val = X[df["unit_number"].isin(val_units)]
    y_val = y[df["unit_number"].isin(val_units)]
    
    # Set MLflow experiment name
    mlflow.set_experiment("C-MAPSS_Predictive_Maintenance")
    
    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42
        }
        
        # Log hyperparameters
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        
        # Log evaluation metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        
        # Save local model binary
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgb_rul_model.pkl")
        
        # Log model with MLflow
        mlflow.xgboost.log_model(model, artifact_path="model")
        print("Model and metrics successfully logged to MLflow!")

if __name__ == "__main__":
    train_model()