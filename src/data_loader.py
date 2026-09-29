
import pandas as pd
import numpy as np

INDEX_COLS = ["unit_number","time_cycles"]
SETTING_COLS = ["op_setting_1","op_setting_2","op_setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1,22)]
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS

def load_train_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+",header=None,names=COLUMN_NAMES)

    max_cycles = df.groupby("unit_number")["time_cycles"].transform("max")
    df["RUL"]= max_cycles - df["time_cycles"]

    df["RUL"] = df["RUL"].clip(upper=125)
    return df


if __name__=="__main__":
    train_path = "data/raw/train_FD001.txt"
    try:
        df = load_train_data(train_path)
        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Total Engines: {df['unit_number'].nunique()}")
        print("\nFirst 5 rows: ")
        print(df[["unit_number","time_cycles","s_2","s_3","RUL"]].head())
    except FileNotFoundError:
        print(f"Error: Could not find {train_path}. Check file path in data/raw/")