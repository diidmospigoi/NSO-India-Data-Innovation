import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from main import create_sector_income_classes

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE, safely skipping zeros"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

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

def main():
    # ─── CONFIG ────────────────────────────────────────────────────────────────
    TRAIN_CSV    = r"C:\Users\PC3\Desktop\Shubham-IIIT\Training_data.csv"
    TEST_CSV     = r"C:\Users\PC3\Desktop\Shubham-IIIT\Testing_data.csv"
    MODEL_PICKLE = r"C:\Users\PC3\Desktop\Shubham-IIIT\models\sector_income_model.pkl"

    # ─── 1. Tag train & test with Income_Class ─────────────────────────────────
    _, test_df, _ = create_sector_income_classes(
        train_path=TRAIN_CSV,
        test_path=TEST_CSV
    )
    
    # Extract the true targets and drop them
    y_true = test_df['TotalExpense'].values
    X_test = test_df.drop(columns=['TotalExpense']).copy()
    
    # ─── 2. Load models & feature pipeline ─────────────────────────────────────
    saved        = joblib.load(MODEL_PICKLE)
    models       = saved['models']        # dict: {1: model1, …, 4: model4}
    feature_info = saved['feature_info']  # contains encoders, scaler, etc.
    
    # ─── 3. Prepare ALL features at once ───────────────────────────────────────
    X_all = prepare_features_batch(X_test, feature_info)
    classes = X_test['Income_Class'].values
    
    # ─── 4. Predict by class mask ──────────────────────────────────────────────
    preds = np.full(len(X_all), np.nan)
    for class_id, model in models.items():
        mask = (classes == class_id)
        if mask.sum() > 0:
            preds[mask] = model.predict(X_all[mask])
    
    # ─── 5. Filter out any NaNs (if some class was missing) ────────────────────
    valid = ~np.isnan(preds)
    y_pred = preds[valid]
    y_true = y_true[valid]
    
    # ─── 6. Compute metrics ──────────────────────────────────────────────────
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # ─── 7. Display ──────────────────────────────────────────────────────────
    print("="*40)
    print("EVALUATION ON TEST SET")
    print("="*40)
    print(f"Samples evaluated: {len(y_pred)}")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print("="*40)
    
    return r2, mape, rmse, mae

if __name__ == "__main__":
    main()
