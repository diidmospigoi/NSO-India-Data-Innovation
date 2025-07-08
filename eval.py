import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# from final_model import prepare_features_batch

# CONFIG 
TEST_CSV  = r"C:\Users\PC3\Desktop\Shubham-IIIT\Testing_data.csv"

# Outputs
TEST_OUT  = "test_raw.csv"

# ─── 2. Read the raw test data WITHOUT touching its TotalExpense ──────────────
test_df = pd.read_csv(TEST_CSV)
print(f"Raw test set loaded: {len(test_df)} rows (no Income_Class assigned)")

# Drop TotalExpense from test so it’s truly unlabeled
test_df.drop(columns=["TotalExpense"], inplace=True)
test_df.to_csv(TEST_OUT, index=False)
print(f"✔ Saved raw test set (no target) → {TEST_OUT}")


# ─── CONFIG ────────────────────────────────────────────────────────────────────
TEST_RAW_CSV            = "test_raw.csv"  # output of Step 1
REGRESSOR_PICKLE        = r"C:\Users\PC3\Desktop\Shubham-IIIT\models_regressor\sector_income_randomforestmodel.pkl"
CLASSIFIER_PICKLE       = r"C:\Users\PC3\Desktop\Shubham-IIIT\models_clf\sector_income_classifiers_tuned.pkl"
OUTPUT_PREDICTIONS_CSV  = "final_predictions.csv"

# ─── 1. Load the raw test set (no TotalExpense, no Income_Class) ──────────────
test_df = pd.read_csv(TEST_RAW_CSV)
print(f"Loaded raw test set: {len(test_df)} rows")

# ─── 2. Load regressors + feature_info ────────────────────────────────────────
reg_data    = joblib.load(REGRESSOR_PICKLE)
reg_models  = reg_data['models']        # dict {1: reg1, …, 4: reg4}
feat_info   = reg_data['feature_info']  # for encoding & scaling

# ─── 3. Load tuned classifiers ─────────────────────────────────────────────────
clf_data    = joblib.load(CLASSIFIER_PICKLE)
clf_rural   = clf_data['clf_rural']
clf_urban   = clf_data['clf_urban']

def prepare_features_batch(data, feature_info):
    """
    One-shot feature prep for a whole DataFrame:
      • OneHotEncode all categorical cols
      • Scale all numerical cols
    """
    categorical_cols = feature_info['categorical_cols']
    numerical_cols   = feature_info['numerical_cols']
    encoders         = feature_info['encoders']
    scaler           = feature_info['scaler']
    
    encoded = []
    # encode all categoricals
    for col in categorical_cols:
        if col in data:
            encoded.append(encoders[col].transform(data[[col]]))
    # scale numericals
    if numerical_cols:
        encoded.append(scaler.transform(data[numerical_cols]))
    
    return np.hstack(encoded)

# ─── 4. Batch‐prepare features ───────────────────────────────────────────────────
X_all = prepare_features_batch(test_df, feat_info)

# ─── 5. Predict binary “Upper” flags ────────────────────────────────────────────
# Masks for sector
mask_rural = test_df['Sector'] == 1
mask_urban = test_df['Sector'] == 2

# Allocate array of predicted classes
predicted_classes = np.zeros(len(test_df), dtype=int)

# Predict rural households
if mask_rural.any():
    rural_preds = clf_rural.predict(X_all[mask_rural])
    # rural_preds: 0=Lower, 1=Upper → map to class 1 or 2
    predicted_classes[mask_rural] = np.where(rural_preds==1, 2, 1)

# Predict urban households
if mask_urban.any():
    urban_preds = clf_urban.predict(X_all[mask_urban])
    # urban_preds: 0=Lower, 1=Upper → map to class 3 or 4
    predicted_classes[mask_urban] = np.where(urban_preds==1, 4, 3)

# ─── 6. Route into the appropriate regressor ───────────────────────────────────
predicted_expenses = np.zeros(len(test_df))

for class_id, model in reg_models.items():
    idx = predicted_classes == class_id
    if idx.any():
        predicted_expenses[idx] = model.predict(X_all[idx])

# ─── 7. Build and save the output DataFrame ────────────────────────────────────
output_df = pd.DataFrame({
    'HH_ID': test_df['HH_ID'],
    'Predicted_Class': predicted_classes,
    'Predicted_TotalExpense': predicted_expenses
})

output_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
print(f"✔ Final predictions saved to {OUTPUT_PREDICTIONS_CSV}")
print(output_df.head(10))

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, skipping zeros in y_true."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

def main():
    # ─── CONFIG ────────────────────────────────────────────────────────────────
    PRED_CSV = "final_predictions.csv"         # output of Step 3
    TEST_CSV = r"C:\Users\PC3\Desktop\Shubham-IIIT\Testing_data.csv"  # original with TotalExpense

    # ─── 1. Load data ──────────────────────────────────────────────────────────
    preds = pd.read_csv(PRED_CSV)
    test  = pd.read_csv(TEST_CSV)[["HH_ID", "TotalExpense"]]

    # ─── 2. Merge on HH_ID ─────────────────────────────────────────────────────
    merged = preds.merge(test, on="HH_ID", how="inner")
    if len(merged) != len(preds):
        print("Warning: some HH_IDs in predictions not matched in test set.")

    # ─── 3. Extract arrays ─────────────────────────────────────────────────────
    y_true = merged["TotalExpense"].values
    y_pred = merged["Predicted_TotalExpense"].values

    # ─── 4. Compute metrics ────────────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # ─── 5. Print results ──────────────────────────────────────────────────────
    print("\n===== Final Model Evaluation =====")
    print(f"Samples evaluated : {len(merged)}")
    print(f"RMSE   : {rmse:,.2f}")
    print(f"MAE    : {mae:,.2f}")
    print(f"MAPE   : {mape:.2f}%")
    print(f"R²     : {r2:.4f}")
    print("===================================")

    # Optional: show a few rows
    print("\nSample vs Actual:")
    print(merged[["HH_ID", "Predicted_TotalExpense", "TotalExpense"]].head(10))

if __name__ == "__main__":
    main()
