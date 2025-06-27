import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


train = pd.read_csv("ct_train_data_withdp.csv").round(0).astype(int)
holdout = pd.read_csv("ct_holdout_data_withdp.csv").round(0).astype(int)
synthetic = pd.read_csv("ct_synthetic_data_withdp.csv").round(0).astype(int)

# Test each of the two combinations for synthetic data generated from each model and with different DP settings 
# simulated = holdout set, synthetic = synthetic -> holdout set IDR and FIDR
# simulated = training set, synthetic = synthetic -> training set IDR and FIDR

# age,gender,BMI,marital_status,pregnancies,number_of_medications,number_of_lab_procedures,time_in_hospital,number_of_inpatient_visits,target

# Quasi ID
QI = [
        "gender", 
        "marital_status", 
        "age",
        "BMI",
        "pregnancies",
        "number_of_medications",
        #"number_of_lab_procedures",
        #"time_in_hospital",
        #"number_of_inpatient_visits"
]

# Quasi ID + Sensitive variables
qi_sv = [
        "gender", 
        "marital_status", 
        "age",
        "BMI",
        "pregnancies",
        "number_of_medications",
        #"number_of_lab_procedures",
        #"time_in_hospital",
        #"number_of_inpatient_visits"
]

numerical = [
    "age",
    #"BMI",
    #"pregnancies",
    #"number_of_lab_procedures",
]

def add(df, u):
    c = df.copy()
    ldf = df[qi_sv].values.tolist()
    ldf = ["_".join([str(int(v)) for v in r]) in u for r in ldf] # in u or test_u
    c['ls_rs'] = np.where(ldf, 1, 0)
    ldf = df[QI].values.tolist()
    ldf = ["_".join([str(v) for v in r]) for r in ldf]
    cldf = Counter(ldf)
    c['QI'] = ldf
    c['QI_count'] = [cldf[r] for r in c['QI']]
    return c

def combinations(df):
    col = QI + ['key', 'count']
    ldf = df[QI].values.tolist()
    ldf = ["_".join([str((v)) for v in r]) for r in ldf]
    cldf = Counter(ldf)
    ldf = [k.split("_") + [k, v] for k, v in list(cldf.items())]
    fdf = pd.DataFrame(ldf, columns=col).sort_values(by=['count'], ascending=False)
    return fdf

def compute_idr(real, syn, numerical, noise):
    categorical = list(set(QI) - set(numerical))
    cols = categorical + numerical

    real = real[cols]
    syn = syn[cols]

    rdf = combinations(real)
    sdf = combinations(syn)

    sd = set(sdf['key'].tolist())
    rd = set(rdf['key'].tolist())

    sd_dict = {}
    for v in sd:
        v = v.split("_")
        k = "_".join([v[i] for i in range(len(categorical))])
        if k not in sd_dict:
            sd_dict[k] = []
        sd_dict[k].append([float(v[i]) for i in range(len(categorical), len(cols))])

    rd_dict = {}
    for v in rd:
        v = v.split("_")
        k = "_".join([v[i] for i in range(len(categorical))])
        if k not in rd_dict:
            rd_dict[k] = []
        rd_dict[k].append([float(v[i]) for i in range(len(categorical), len(cols))])

    test_u = []
    for k,v in sd_dict.items():
        if k in rd_dict:
            w = rd_dict[k]
            
            v = np.array(v).astype(int)
            w = np.array(w).astype(int)

            d = cdist(w,v, metric="cityblock")
            d = np.where(d <= noise, 1, 0)
            w_add = d.sum(axis=1)
            v_add = d.sum(axis=0)
            
            
            for i, c in enumerate(w_add):
                if c > 0:
                    n = "_".join([k] + [str(w[i][idx]) for idx in range(len(cols)-len(categorical))])
                    test_u.append(n)
            
            for i, c in enumerate(v_add):
                if c > 0:
                    n = "_".join([k] + [str(v[i][idx]) for idx in range(len(cols)-len(categorical))])
                    test_u.append(n)

    test_u = set(test_u)
    u = set.intersection(sd,rd)

    ss = sdf.sum()
    rs = rdf.sum()

    rdf = add(real, u)
    sdf = add(syn, u)
    sr = (1/sdf['QI_count']).mul(sdf['ls_rs']/rs['count']).sum()
    rr = (1/rdf['QI_count']).mul(rdf['ls_rs']/ss['count']).sum()
    
    
    print("idr", f"{sr:.3e}", f"{rr:.3e}")


    rdf = add(real, test_u)
    sdf = add(syn, test_u)
    sr = (1/sdf['QI_count']).mul(sdf['ls_rs']/rs['count']).sum()
    rr = (1/rdf['QI_count']).mul(rdf['ls_rs']/ss['count']).sum()
    
    
    print("fidr", f"{sr:.3e}", f"{rr:.3e}")
    

    
compute_idr(train, holdout, numerical, 1)
compute_idr(train, synthetic, numerical, 1)

compute_idr(train, holdout, numerical, 2)
compute_idr(train, synthetic, numerical, 2)

compute_idr(train, holdout, numerical, 3)
compute_idr(train, synthetic, numerical, 3)

# --- Additional Code for Subtraction of IDR and FIDR ---

def compute_idr_values(real, syn, numerical, noise):
    # Use the same QI and qi_sv as defined earlier.
    categorical = list(set(QI) - set(numerical))
    cols = categorical + numerical

    real_sub = real[cols]
    syn_sub = syn[cols]

    rdf = combinations(real_sub)
    sdf = combinations(syn_sub)

    sd = set(sdf['key'].tolist())
    rd = set(rdf['key'].tolist())

    sd_dict = {}
    for v in sd:
        parts = v.split("_")
        k = "_".join(parts[:len(categorical)])
        sd_dict.setdefault(k, []).append([float(x) for x in parts[len(categorical):]])
    rd_dict = {}
    for v in rd:
        parts = v.split("_")
        k = "_".join(parts[:len(categorical)])
        rd_dict.setdefault(k, []).append([float(x) for x in parts[len(categorical):]])

    test_u = []
    for k, v in sd_dict.items():
        if k in rd_dict:
            w = rd_dict[k]
            v_arr = np.array(v).astype(int)
            w_arr = np.array(w).astype(int)
            # Calculate pairwise Manhattan distances
            d = cdist(w_arr, v_arr, metric="cityblock")
            d = np.where(d <= noise, 1, 0)
            w_add = d.sum(axis=1)
            v_add = d.sum(axis=0)
            for i, count in enumerate(w_add):
                if count > 0:
                    n = "_".join([k] + [str(w_arr[i][j]) for j in range(v_arr.shape[1])])
                    test_u.append(n)
            for i, count in enumerate(v_add):
                if count > 0:
                    n = "_".join([k] + [str(v_arr[i][j]) for j in range(v_arr.shape[1])])
                    test_u.append(n)

    test_u = set(test_u)
    u = set.intersection(sd, rd)
    
    # Using the count sums from combinations DataFrames
    ss = sdf['count'].sum()
    rs = rdf['count'].sum()
    
    # IDR calculation using the intersection u
    rdf_u = add(real_sub, u)
    sdf_u = add(syn_sub, u)
    idr_sr = (1 / sdf_u['QI_count']).mul(sdf_u['ls_rs'] / rs).sum()
    idr_rr = (1 / rdf_u['QI_count']).mul(rdf_u['ls_rs'] / ss).sum()
    idr_val = max(idr_sr, idr_rr)
    
    # FIDR calculation using test_u
    rdf_t = add(real_sub, test_u)
    sdf_t = add(syn_sub, test_u)
    fidr_sr = (1 / sdf_t['QI_count']).mul(sdf_t['ls_rs'] / rs).sum()
    fidr_rr = (1 / rdf_t['QI_count']).mul(rdf_t['ls_rs'] / ss).sum()
    fidr_val = max(fidr_sr, fidr_rr)
    
    return idr_val, fidr_val

# For each noise level, compute IDR and FIDR for train vs holdout and train vs synthetic,
# then subtract the holdout value (larger one) from the synthetic value (larger one).
for noise in [1, 2, 3]:
    idr_holdout, fidr_holdout = compute_idr_values(train, holdout, numerical, noise)
    idr_synthetic, fidr_synthetic = compute_idr_values(train, synthetic, numerical, noise)
    
    diff_idr = idr_synthetic - idr_holdout
    diff_fidr = fidr_synthetic - fidr_holdout

    print(f"Noise {noise}: Difference in IDR (train vs synthetic - train vs holdout) = {diff_idr:.3e}")
    print(f"Noise {noise}: Difference in FIDR (train vs synthetic - train vs holdout) = {diff_fidr:.3e}")