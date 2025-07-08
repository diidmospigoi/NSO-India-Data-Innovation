# Step 1: Saving the enriched training set (with Income_Class) and the raw test set (no TotalExpense, no Income_Class) to disk.
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from main import prepare_features
from main import create_sector_income_classes

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_CSV = r"C:\Users\PC3\Desktop\Shubham-IIIT\Training_data.csv"
TEST_CSV  = r"C:\Users\PC3\Desktop\Shubham-IIIT\Testing_data.csv"

# Outputs
TRAIN_OUT = "train_with_class.csv"
TEST_OUT  = "test_raw.csv"

# ─── 1. Compute Income_Class ONLY on the TRAINING DATA ─────────────────────────
# Note: test_path=None ensures we don't assign classes on test
train_df, _, sector_medians = create_sector_income_classes(
    train_path=TRAIN_CSV,
    test_path=None
)

# train_df now has a new column 'Income_Class' (1–4)
print(f"Training set enriched: {len(train_df)} rows with Income_Class labels")

# ─── 2. Read the raw test data WITHOUT touching its TotalExpense ──────────────
test_df = pd.read_csv(TEST_CSV)
print(f"Raw test set loaded: {len(test_df)} rows (no Income_Class assigned)")

# ─── 3. Save both for downstream steps ────────────────────────────────────────
train_df.to_csv(TRAIN_OUT, index=False)
print(f"✔ Saved enriched training set → {TRAIN_OUT}")

# Drop TotalExpense from test so it’s truly unlabeled
test_df.drop(columns=["TotalExpense"], inplace=True)
test_df.to_csv(TEST_OUT, index=False)
print(f"✔ Saved raw test set (no target) → {TEST_OUT}")

# pipeline_step1_tuning
""" Best Model: XGBoost with Randomized Search CV saving best models with best hyperparameters as sector_income_classifier_tuned.pkl"""


# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_WITH_CLASS = "train_with_class.csv"
REGRESSOR_PICKLE = r"C:\Users\PC3\Desktop\Shubham-IIIT\models_regressor\sector_income_randomforestmodel.pkl"
CLASSIFIER_OUT   = "sector_income_classifiers_tuned.pkl"
RANDOM_STATE     = 42

# ─── 1. Load data & feature pipeline ──────────────────────────────────────────
df = pd.read_csv(TRAIN_WITH_CLASS)
reg_saved = joblib.load(REGRESSOR_PICKLE)
feat_info = reg_saved['feature_info']

# Helper to tune one sector
def tune_sector(sector_df, positive_classes, sector_name):
    # Prepare X, y
    X = prepare_features(sector_df, feat_info)
    y = sector_df['Income_Class'].isin(positive_classes).astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Define models & parameter distributions
    models = {
        'rf': (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
              'n_estimators': [100, 200, 500],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]
            }
        ),
        'xgb': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
            {
              'n_estimators': [100, 200, 500],
              'max_depth': [3, 6, 10],
              'learning_rate': [0.01, 0.1, 0.3],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0]
            }
        )
    }
    
    best_models = {}
    for name, (clf, params) in models.items():
        print(f"\nTuning {sector_name} → {name.upper()}...")
        search = RandomizedSearchCV(
            clf, params,
            n_iter=20,
            scoring='accuracy',
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        y_pred = best.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"→ Best {name.upper()} params: {search.best_params_}")
        print(f"→ Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, target_names=['Lower','Upper']))
        best_models[name] = best
    return best_models

# ─── Tune Rural (classes 1 vs 2) ───────────────────────────────────────────────
rural_df = df[df['Sector']==1]
best_rural = tune_sector(rural_df, positive_classes=[2], sector_name="Rural")

# ─── Tune Urban (classes 3 vs 4) ───────────────────────────────────────────────
urban_df = df[df['Sector']==2]
best_urban = tune_sector(urban_df, positive_classes=[4], sector_name="Urban")

# ─── PICK & SAVE THE BEST PER SECTOR ──────────────────────────────────────────
# For simplicity, choose XGB if it outperformed RF, else RF:
final_rural = best_rural['xgb'] if best_rural['xgb'].score(
    prepare_features(rural_df, feat_info), rural_df['Income_Class'].isin([2])
) > best_rural['rf'].score(
    prepare_features(rural_df, feat_info), rural_df['Income_Class'].isin([2])
) else best_rural['rf']

final_urban = best_urban['xgb'] if best_urban['xgb'].score(
    prepare_features(urban_df, feat_info), urban_df['Income_Class'].isin([4])
) > best_urban['rf'].score(
    prepare_features(urban_df, feat_info), urban_df['Income_Class'].isin([4])
) else best_urban['rf']

joblib.dump({
    'clf_rural': final_rural,
    'clf_urban': final_urban
}, CLASSIFIER_OUT)

print(f"\n✔ Saved tuned classifiers → {CLASSIFIER_OUT}")

