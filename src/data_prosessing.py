import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data.ffill(inplace=True)
    
    data = pd.get_dummies(data, columns=['AQI_Bucket'], drop_first=True)
    
    return data

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def apply_smote(X_train, y_train):
    class_counts = y_train.value_counts()
    min_samples = class_counts.min()
    
    smote = SMOTE(random_state=42, k_neighbors=min_samples - 1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def process_data(file_path):
    """Main function to process the data."""
    data = load_data(file_path)
    
    data = preprocess_data(data)
    
    X = data.drop(columns=[col for col in data.columns if col.startswith('AQI_Bucket_')])
    y = data[[col for col in data.columns if col.startswith('AQI_Bucket_')]].idxmax(axis=1)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test
