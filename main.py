import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# New function to calculate MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    float: MAPE value
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle any zeros in y_true to avoid division by zero
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate absolute percentage errors
    ape = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered) * 100
    
    # Return the mean
    return np.mean(ape)

def prepare_features(data, feature_info):
    """
    Prepare features for model training or prediction
    """
    # Extract info
    categorical_cols = feature_info['categorical_cols']
    numerical_cols = feature_info['numerical_cols']
    encoders = feature_info['encoders']
    scaler = feature_info['scaler']
    
    # Process categorical features
    encoded_features = []
    for col in categorical_cols:
        if col in data.columns:
            # Apply one-hot encoding
            encoded = encoders[col].transform(data[[col]])
            encoded_features.append(encoded)
    
    # Process numerical features
    if numerical_cols:
        # Scale numerical features
        scaled_numerical = scaler.transform(data[numerical_cols])
        # Add to feature list
        encoded_features.append(scaled_numerical)
    
    # Combine all features
    if encoded_features:
        X = np.hstack(encoded_features)
    else:
        raise ValueError("No features available after preprocessing")
    
    return X


# Previous functions remain unchanged
def create_sector_income_classes(data, test_path=None):
    """
    Create classes based on sector (rural/urban) and income level (above/below median)
    """
    # Load the training data
    print("Loading training dataset...")
    train_df = pd.read_csv(data)
    
    # Handle missing values in NCO_3D and NIC_5D if needed
    if 'NCO_3D' in train_df.columns and 'NIC_5D' in train_df.columns:
        if train_df[['NCO_3D', 'NIC_5D']].isnull().any().any():
            print("Dropping rows with missing NCO_3D or NIC_5D values...")
            train_df = train_df.dropna(subset=['NCO_3D', 'NIC_5D'])
    
    # Verify that Sector column exists and has values 1 and 2
    if 'Sector' not in train_df.columns:
        raise ValueError("Sector column not found in the dataset")
    
    # Check Sector values
    sector_values = train_df['Sector'].unique()
    if not all(val in sector_values for val in [1, 2]):
        print(f"Warning: Expected Sector values 1 and 2, found {sector_values}")
    
    # Calculate median TotalExpense for each sector
    sector_medians = train_df.groupby('Sector')['TotalExpense'].median().to_dict()
    
    print("\nMedian TotalExpense by Sector:")
    for sector, median in sector_medians.items():
        sector_name = "Rural" if sector == 1 else "Urban" if sector == 2 else f"Unknown ({sector})"
        print(f"{sector_name}: {median:.2f}")
    
    # Create class labels (1: lower rural, 2: upper rural, 3: lower urban, 4: upper urban)
    def assign_class(row):
        if row['Sector'] == 1:  # Rural
            return 1 if row['TotalExpense'] <= sector_medians[1] else 2
        elif row['Sector'] == 2:  # Urban
            return 3 if row['TotalExpense'] <= sector_medians[2] else 4
        else:
            # Handle unexpected sector values
            print(f"Warning: Unexpected Sector value {row['Sector']}")
            return 0  # Unknown class
    
    # Add class column to training data
    train_df['Income_Class'] = train_df.apply(assign_class, axis=1)
    
    # Get class distribution
    class_distribution = train_df['Income_Class'].value_counts().sort_index()
    
    print("\nClass Distribution in Training Data:")
    class_names = {1: "Lower Rural", 2: "Upper Rural", 3: "Lower Urban", 4: "Upper Urban", 0: "Unknown"}
    for class_id, count in class_distribution.items():
        print(f"{class_names[class_id]}: {count} households ({count/len(train_df):.1%})")
    
    # If test data is provided, process it too
    if test_path:
        print("\nLoading test dataset...")
        test_df = pd.read_csv(test_path)
        
        # For test data, we only use the sector to determine rural/urban
        # Income level will be predicted
        # But we still calculate the actual class for evaluation
        test_df['Income_Class'] = test_df.apply(assign_class, axis=1)
        
        # Get class distribution for test data
        test_class_distribution = test_df['Income_Class'].value_counts().sort_index()
        
        print("\nClass Distribution in Test Data:")
        for class_id, count in test_class_distribution.items():
            print(f"{class_names[class_id]}: {count} households ({count/len(test_df):.1%})")
        
        return train_df, test_df, sector_medians
    
    return train_df, None, sector_medians

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
        model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    data = r'C:\Users\PC3\Desktop\Shubham-IIIT\Training_data.csv'

    # Step 1: Create sector-income classes
    print("=== Creating Sector-Income Classes ===")
    train_df, _, sector_medians = create_sector_income_classes(data, None)
    
    # Step 2: Train class-based models
    print("\n=== Training Models by Class ===")
    models, feature_info = train_class_based_models(train_df, sector_medians)
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump({
        'models': models,
        'feature_info': feature_info
    }, 'models/sector_income_randomforestmodel.pkl')
    
    print("\nModel saved as 'models/sector_income_randomforestmodel.pkl'")
    print("\nPrediction pipeline complete!")




