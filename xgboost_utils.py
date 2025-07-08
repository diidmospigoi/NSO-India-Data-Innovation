import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from main import mean_absolute_percentage_error, create_sector_income_classes
from main import prepare_features
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from xgboost import XGBRegressor
import joblib

# Training XG-Boost Model for MPCE predictions
def train_class_based_models(train_df, sector_medians):
    """
    Train separate models for each sector-income class
    """
    # Define target variable
    target = 'TotalExpense'
    
    # Create one-hot encoding for categorical variables
    categorical_cols = ['Sector', 'State', 'NSS-Region', 'District', 
                        'Household Type', 'Religion of the head of the household',
                        'Social Group of the head of the household']
    
    # Check which categorical columns actually exist in the data
    categorical_cols = [col for col in categorical_cols if col in train_df.columns]
    
    # Get numerical columns (exclude HH_ID, target, and Income_Class)
    numerical_cols = [col for col in train_df.columns 
                     if col not in categorical_cols + ['HH_ID', target, 'Income_Class']]
    
    # Create preprocessing pipeline
    encoders = {}
    for col in categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(train_df[[col]])
        encoders[col] = encoder
    
    # Scale numerical features
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_cols])
    
    # Save feature information
    feature_info = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'encoders': encoders,
        'scaler': scaler,
        'sector_medians': sector_medians
    }
    
    # Train separate models for each class
    models = {}
    class_names = {1: "Lower Rural", 2: "Upper Rural", 3: "Lower Urban", 4: "Upper Urban"}
    
    for class_id in range(1, 5):  # 1 to 4
        print(f"\nTraining model for Class {class_id} ({class_names[class_id]})...")
        
        # Get data for this class
        class_data = train_df[train_df['Income_Class'] == class_id].copy()
        
        if len(class_data) < 10:
            print(f"Not enough samples for Class {class_id}. Skipping.")
            continue
        
        # Prepare features
        X_class = prepare_features(class_data, feature_info)
        y_class = class_data[target]
        
        # Train model
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_class, y_class)
        models[class_id] = model
        
        # Show feature importance
        feature_names = (
            [f"{col}_{cat}" for col in categorical_cols 
             for cat in encoders[col].categories_[0]] + 
            numerical_cols
        )
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(10, len(feature_names))
        
        print(f"Top {top_n} Feature Importances:")
        for i in range(top_n):
            idx = indices[i]
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return models, feature_info



# Main execution
if __name__ == "__main__":
    # Define paths
    train_path = r'C:\Users\PC3\Desktop\Shubham-IIIT\Training_data.csv'
 
    # Step 1: Create sector-income classes
    print("=== Creating Sector-Income Classes ===")
    train_df, test_df, sector_medians = create_sector_income_classes(train_path, None)
    
    # Step 2: Train class-based models
    print("\n=== Training Models by Class ===")
    models, feature_info = train_class_based_models(train_df, sector_medians)
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump({
        'models': models,
        'feature_info': feature_info
    }, 'models/sector_income_xgboostmodel.pkl')
    
    print("\nModel saved as 'models/sector_income_xgboostmodel.pkl'")
    print("\nPrediction pipeline complete!")

