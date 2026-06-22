import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional
import logging


# Load The Dataset For preprocessing 
def load_data(file_path: str)-> pd.DataFrame:
    df1=pd.read_csv(file_path, encoding='unicode_escape')
    return df1

def process_data(df: pd.DataFrame, x: Optional[List[str]] = None, cat: Optional[List[str]] = None, drop_columns: Optional[list[str]] = None,round_off:Optional[List[str]] = None):
    # 1. Handle numeric column filling
    if x is not None:
        for i in x:
            # Coerce to numeric first in case pandas read them as string/object (e.g., due to NA strings)
            df[i] = pd.to_numeric(df[i], errors='coerce')
            df = df.dropna(subset=[i])

    # 3. Drop columns
    if drop_columns is not None:
        existing_drop_columns = [col for col in drop_columns if col in df.columns]
        df = df.drop(columns=existing_drop_columns)
            
    # 4. Handle categorical column encoding
    encoders = {}  # Initialize encoders so it's always defined
    if cat is not None:
        for i in cat:
            le = LabelEncoder()
            # Ensure column is treated as string to avoid type conflicts during fitting
            df[i] = df[i].astype(str)
            df[i] = le.fit_transform(df[i])
            encoders[i] = le
    if round_off is not None:
        for i in round_off:
            df[i] = df[i].round(2)

    # 5. Save the outputs using forward slashes (works on both Windows and Linux)
    df.to_csv("data/processed/processed_data1.csv", index=False)
    return df, encoders
    
