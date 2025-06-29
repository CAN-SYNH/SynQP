// ─── Pipeline A: Generate & Evaluate Synthetic Data ───
function generate_and_evaluate_synthetic():
    # 1. Load real data (limit to 10K rows)
    real_df ← load_csv("diabetic_data.csv")
    real_df ← sample_if_large(real_df, max_rows=10000)

    # 2. Fill or simulate key features (age, BMI, etc.)
    real_df ← ensure_or_simulate(real_df, columns=[
        age, BMI, pregnancies, gender, marital_status,
        number_of_medications, number_of_lab_procedures,
        time_in_hospital, number_of_inpatient_visits
    ])

    # 3. Derive target and link external data
    real_df.target ← (real_df.BMI > 30)
    real_df ← link_external(real_df, ["bmi.csv","diabetes.csv"])

    # 4. Split → train vs holdout
    train_df, holdout_df ← split(real_df, test_frac=0.3)

    # 5. Add DP noise to train
    dp_train ← add_dp_noise(train_df, cost=0.8)

    # 6. Fit CTGAN on dp_train → sample synthetic
    model ← CTGAN(epochs=200)
    model.fit(dp_train)
    syn_df ← model.sample(n_rows = train_df.row_count)

    # 7. Utility check (logistic AUC)
    auc ← eval_auc(syn_df, holdout_df)

    # 8. Fidelity check (Hellinger distance)
    for col in numeric_cols:
        dist ← hellinger(train_df[col], syn_df[col])
        print(col, dist)

    return train_df, holdout_df, syn_df, auc
end function


// ─── Pipeline B: Compute IDR & FIDR Differences ───
function compute_privacy_metrics(train_df, holdout_df, syn_df):
    QI_cols ← [gender, marital_status, age, BMI, pregnancies, number_of_medications]
    numeric_sub ← [age]  # for distance

    for noise in [1,2,3]:
        # 1. Exact-match risk (IDR)
        idr_hold ← idr(train_df, holdout_df, QI_cols, numeric_sub, noise)
        idr_syn  ← idr(train_df, syn_df,    QI_cols, numeric_sub, noise)
        ΔIDR ← idr_syn − idr_hold

        # 2. Fuzzy-match risk (FIDR)
        fidr_hold ← fidr(train_df, holdout_df, QI_cols, numeric_sub, noise)
        fidr_syn  ← fidr(train_df, syn_df,    QI_cols, numeric_sub, noise)
        ΔFIDR ← fidr_syn − fidr_hold

        print("noise=", noise, "ΔIDR=", ΔIDR, "ΔFIDR=", ΔFIDR)
    end for
end function