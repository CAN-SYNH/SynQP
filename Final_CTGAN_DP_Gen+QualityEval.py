import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from ctgan import CTGAN
import matplotlib.pyplot as plt

synthetic_nodp = pd.read_csv("ct_synthetic_data_nodp.csv").round(0).astype(int)
# Increase recursion limit
sys.setrecursionlimit(30000)

# Add Differential Privacy Manually in the DataFrame
# --------------------------------------------------------------
def add_dp_noise(df, numeric_cols, cost=0.8):
    noisy_df = df.copy()
    scale = 1.0 / cost
    for col in numeric_cols:
        noise = np.random.laplace(loc=df[col].mean(), scale=df[col].std(), size=noisy_df.shape[0])
        noisy_df[col] = noisy_df[col] * (1 - cost) + noise * cost
    return noisy_df

# Data Simulation Utilities
# --------------------------------------------------------------
def ensure_numeric_column(df, col, simulation_func):
    if col not in df.columns or df[col].isna().all():
        df[col] = simulation_func(len(df))
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        missing_mask = df[col].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, col] = simulation_func(missing_mask.sum())
    return df[col]

def add_additional_feature(df, col, simulation_func):
    if col not in df.columns:
        df[col] = simulation_func(len(df))
    else:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass
        missing_mask = df[col].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, col] = simulation_func(missing_mask.sum())
    return df[col]


# Linking External Data Functions
# --------------------------------------------------------------
def link_bmi_data(sim_df, bmi_file='bmi.csv'):
    bmi_df = pd.read_csv(bmi_file)
    bmi_df.columns = [col.lower() for col in bmi_df.columns]
    if 'height' not in bmi_df.columns or 'weight' not in bmi_df.columns:
        raise KeyError("BMI CSV must contain 'height' and 'weight' columns.")
    bmi_df['bmi'] = bmi_df['weight'] / ((bmi_df['height'] / 100) ** 2)
    
    def sample_bmi(row):
        gender_val = row['gender']
        subset = bmi_df[bmi_df['gender'] == gender_val]
        if subset.empty:
            return row
        sample = subset.sample(n=1).iloc[0]
        row['height'] = sample['height']
        row['weight'] = sample['weight']
        row['BMI'] = sample['bmi']
        return row
    sim_df = sim_df.apply(sample_bmi, axis=1)
    return sim_df

def link_diabetes_data(sim_df, diabetes_file='diabetes.csv'):
    diabetes_df = pd.read_csv(diabetes_file)
    diabetes_df.columns = [col.lower() for col in diabetes_df.columns]
    def sample_diabetes(row):
        a = row['age']
        bmi_val = row['BMI']
        distances = np.sqrt((diabetes_df['age'] - a)**2 + (diabetes_df['bmi'] - bmi_val)**2)
        idx = distances.idxmin()
        return row
    sim_df = sim_df.apply(sample_diabetes, axis=1)
    return sim_df


# Data Generation, Linking, Synthetic Data Generation, and Evaluation
# --------------------------------------------------------------
def main():
    # Load the initial diabetic dataset (seed)
    df = pd.read_csv('diabetic_data.csv')
    print("Original diabetic dataset shape:", df.shape)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    print("Dataset shape after limiting to 10,000 rows:", df.shape)
    
    # Process quasi-identifiers
    df['age'] = ensure_numeric_column(df, 'age', lambda n: np.random.randint(20, 81, size=n))
    df['BMI'] = ensure_numeric_column(df, 'BMI', lambda n: np.clip(np.random.normal(loc=27, scale=4, size=n), 15, 50))
    df['pregnancies'] = ensure_numeric_column(df, 'pregnancies', lambda n: np.random.poisson(lam=1, size=n))
    if 'gender' not in df.columns or df['gender'].isna().all():
        df['gender'] = np.random.choice([0, 1], size=len(df))
    else:
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
        missing_mask = df['gender'].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, 'gender'] = np.random.choice([0, 1], size=missing_mask.sum())
    if 'marital_status' not in df.columns or df['marital_status'].isna().all():
        df['marital_status'] = np.random.choice([0, 1, 2], size=len(df), p=[0.3, 0.6, 0.1])
    else:
        df['marital_status'] = pd.to_numeric(df['marital_status'], errors='coerce')
        missing_mask = df['marital_status'].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, 'marital_status'] = np.random.choice([0, 1, 2], size=missing_mask.sum(), p=[0.3, 0.6, 0.1])
    
    # Process additional features
    df['number_of_medications'] = add_additional_feature(df, 'number_of_medications', lambda n: np.random.randint(0, 11, size=n))
    df['number_of_lab_procedures'] = add_additional_feature(df, 'number_of_lab_procedures', lambda n: np.random.randint(1, 101, size=n))
    df['time_in_hospital'] = add_additional_feature(df, 'time_in_hospital', lambda n: np.random.randint(1, 15, size=n))
    df['number_of_inpatient_visits'] = add_additional_feature(df, 'number_of_inpatient_visits', lambda n: np.random.randint(0, 6, size=n))
    df.fillna(method='ffill', inplace=True)
    
    # Create final simulated dataset
    qi_cols = ['age', 'gender', 'BMI', 'marital_status', 'pregnancies']
    additional_features = ['number_of_medications', 'number_of_lab_procedures', 'time_in_hospital', 'number_of_inpatient_visits']
    simulated_df = df[qi_cols + additional_features].copy()
    simulated_df['BMI'] = simulated_df['BMI'].round(1)
    simulated_df['gender'] = simulated_df['gender'].astype(object)
    simulated_df['marital_status'] = simulated_df['marital_status'].astype(object)
    simulated_df['target'] = (simulated_df['BMI'] > 30).astype(int)
    
    # Link external data
    simulated_df = link_bmi_data(simulated_df, bmi_file='bmi.csv')
    simulated_df = link_diabetes_data(simulated_df, diabetes_file='diabetes.csv')
    print("\nSimulated dataset head after linking external data:")
    print(simulated_df.head())
    
    # Split Data & Train TVAE Synthesizer with Metadata
    # --------------------------------------------------------------
    final_cols = qi_cols + additional_features + ['target']
    train_df, holdout_df = train_test_split(simulated_df[final_cols], test_size=0.3, random_state=42)
    holdout_df=holdout_df.sample(frac=0.3,random_state=42)
    print("\nTrainset size:", len(train_df))
    print("Holdout set size:", len(holdout_df))

    # Add Differential Privacy Noise
    # --------------------------------------------------------------
    numeric_cols_dp = ['age', 'BMI', 'pregnancies', 'number_of_medications', 
                       'number_of_lab_procedures', 'time_in_hospital', 'number_of_inpatient_visits']
    train_df_1 = add_dp_noise(train_df, numeric_cols_dp, cost=0.8)
    print("\nSimulated dataset head (after DP noise):")
    print(simulated_df.head())
    
    # Generate Synthetic Data with CTGAN
    # --------------------------------------------------------------
    model = CTGAN(epochs=200,verbose=True, generator_dim=(128, 128), discriminator_dim=(128, 128))
    model.fit(train_df_1)
    synthetic_df = model.sample(len(train_df))
    
    synthetic_df['age'] = synthetic_df['age'].round(0).astype(int)
    synthetic_df['gender'] = synthetic_df['gender'].round(0).astype(int).astype(object)
    synthetic_df['BMI'] = synthetic_df['BMI'].round(1).astype(int)
    synthetic_df['target'] = synthetic_df['target'].round(0).astype(int)
    
    print("\nSynthetic dataset head:")
    print(synthetic_df.head())

    simulated_df.to_csv("simulated_data.csv", index=False)
    synthetic_df.to_csv("ct_synthetic_data_withdp.csv", index=False)
    train_df.to_csv('ct_train_data_withdp.csv', index=False)
    holdout_df.to_csv('ct_holdout_data_withdp.csv', index=False)
    
    # Utility evaluation using logistic regression
    # --------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    util_features = qi_cols + additional_features  # exclude target
    X_syn = synthetic_df[util_features]
    y_syn = synthetic_df['target']
    X_holdout = holdout_df[util_features]
    y_holdout = holdout_df['target']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_syn, y_syn)
    y_pred_proba = clf.predict_proba(X_holdout)[:,1]
    auc = roc_auc_score(y_holdout, y_pred_proba)
    print("\nUtility Evaluation (AUC of logistic regression trained on synthetic data): {:.8f}".format(auc))
    
    # Fidelity evaluation using Hellinger distance for numeric columns
    # --------------------------------------------------------------
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
    numeric_cols = ['age', 'BMI', 'pregnancies', 'number_of_medications',
                    'number_of_lab_procedures', 'time_in_hospital', 'number_of_inpatient_visits']
    print("\nFidelity Evaluation (Hellinger distances):")
    for col in numeric_cols:
        holdout_hist, bin_edges = np.histogram(train_df[col], bins=10, density=True)
        synthetic_hist, _ = np.histogram(synthetic_df[col], bins=bin_edges, density=True)
        holdout_hist = holdout_hist / holdout_hist.sum()
        synthetic_hist = synthetic_hist / synthetic_hist.sum()
        h_dist = hellinger_distance(holdout_hist, synthetic_hist)
        print("  {}: Hellinger distance = {:.8f}".format(col, h_dist))
    for col in numeric_cols:
    # Combine both datasets to define common bin edges
    # --------------------------------------------------------------
        data_combined = np.concatenate([train_df[col].values, train_df_1[col].values,synthetic_df[col].values,synthetic_nodp[col].values])
        bin_edges = np.histogram_bin_edges(data_combined, bins=40)
    
        plt.figure(figsize=(4,2))
        
        plt.hist(synthetic_df[col], bins=bin_edges, alpha=0.5, density=False, label='Syn')
        plt.hist(train_df_1[col], bins=bin_edges, alpha=0.5, density=False, label='Real')
        plt.title(f"Distribution of {col} with DP")
        plt.xlabel(col)
        plt.yscale('log')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(4, 2))
        plt.hist(synthetic_nodp[col], bins=bin_edges, alpha=0.5, density=False, label='Syn')
        plt.hist(train_df[col], bins=bin_edges, alpha=0.5, density=False, label='Real')
        plt.title(f"Distribution of {col} without DP")
        plt.xlabel(col)
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main() 



