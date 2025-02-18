# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """Load dataset from a given file path."""
    return pd.read_csv(path)

def preprocess_data(X, y):
    """Preprocess features and target variables."""
    # Label encode the target variable if necessary
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded
