# src/data_preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(raw_path):
    df = pd.read_csv(raw_path)
    return df

def preprocess_data(df):
    # Create a copy and scale the 'Time' and 'Amount' features.
    df = df.sort_values('Time').reset_index(drop=True)
    df_processed = df.copy()
    scaler = StandardScaler()
    df_processed[['Time', 'Amount']] = scaler.fit_transform(df_processed[['Time', 'Amount']])
    return df_processed

def split_and_save(df, output_dir, test_size=0.2, random_state=42):
    # Separate features and label
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified split because of heavy imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    train_df = X_train.copy()
    train_df['Class'] = y_train
    test_df = X_test.copy()
    test_df['Class'] = y_test

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Saved training data to {train_file}")
    print(f"Saved test data to {test_file}")

def main():
    raw_data_path = os.path.join('data', 'raw', 'creditcard.csv')
    processed_data_dir = os.path.join('data', 'processed')
    df = load_data(raw_data_path)
    print(f"Loaded raw data with shape: {df.shape}")
    df_processed = preprocess_data(df)
    print("Data preprocessing completed.")
    split_and_save(df_processed, processed_data_dir)

if __name__ == '__main__':
    main()
