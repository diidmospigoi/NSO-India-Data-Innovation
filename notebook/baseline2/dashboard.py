import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="MPCE Prediction Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #0EA5E9;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #10B981;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #047857;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .sector-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Monthly Per Capita Expenditure (MPCE) Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive tool helps estimate the monthly per capita expenditure for Indian households based on their demographic and socioeconomic characteristics. Adjust the filters on the left to see how different factors influence household expenditure patterns.
""")

# Actual MPCE data for 2022-23 from the provided table
state_mpce_data = {
    1: {"name": "Jammu & Kashmir", "rural": 4296, "urban": 6179},
    2: {"name": "Himachal Pradesh", "rural": 5561, "urban": 8075},
    3: {"name": "Punjab", "rural": 5315, "urban": 6544},
    4: {"name": "Chandigarh (U.T.)", "rural": 7467, "urban": 12575},
    5: {"name": "Uttarakhand", "rural": 4641, "urban": 7004},
    6: {"name": "Haryana", "rural": 4859, "urban": 7911},
    7: {"name": "Delhi", "rural": 6576, "urban": 8217},
    8: {"name": "Rajasthan", "rural": 4263, "urban": 5913},
    9: {"name": "Uttar Pradesh", "rural": 3191, "urban": 5040},
    10: {"name": "Bihar", "rural": 3384, "urban": 4768},
    11: {"name": "Sikkim", "rural": 7731, "urban": 12105},
    12: {"name": "Arunachal Pradesh", "rural": 5276, "urban": 8636},
    13: {"name": "Nagaland", "rural": 4393, "urban": 7098},
    14: {"name": "Manipur", "rural": 4360, "urban": 4880},
    15: {"name": "Mizoram", "rural": 5224, "urban": 7655},
    16: {"name": "Tripura", "rural": 5206, "urban": 7405},
    17: {"name": "Meghalaya", "rural": 3514, "urban": 6433},
    18: {"name": "Assam", "rural": 3432, "urban": 6136},
    19: {"name": "West Bengal", "rural": 3239, "urban": 5267},
    20: {"name": "Jharkhand", "rural": 2763, "urban": 4931},
    21: {"name": "Odisha", "rural": 2950, "urban": 5187},
    22: {"name": "Chhattisgarh", "rural": 2466, "urban": 4483},
    23: {"name": "Madhya Pradesh", "rural": 3113, "urban": 4987},
    24: {"name": "Gujarat", "rural": 3798, "urban": 6621},
    25: {"name": "Dadra & Nagar Haveli and Daman & Diu", "rural": 4184, "urban": 6298},
    26: {"name": "Telangana", "rural": 4802, "urban": 8158},
    27: {"name": "Maharashtra", "rural": 4010, "urban": 6657},
    28: {"name": "Andhra Pradesh", "rural": 4870, "urban": 6782},
    29: {"name": "Karnataka", "rural": 4397, "urban": 7666},
    30: {"name": "Goa", "rural": 7367, "urban": 8734},
    31: {"name": "Lakshadweep (U.T.)", "rural": 5895, "urban": 5475},
    32: {"name": "Kerala", "rural": 5924, "urban": 7078},
    33: {"name": "Tamil Nadu", "rural": 5310, "urban": 7630},
    34: {"name": "Puducherry (U.T.)", "rural": 6590, "urban": 7706},
    35: {"name": "Andaman & Nicobar Islands (U.T.)", "rural": 7332, "urban": 10268},
    36: {"name": "Ladakh (U.T.)", "rural": 4035, "urban": 6215},
}

# National average
national_average = {"rural": 3773, "urban": 6459}

# Utility functions from your ML model
def prepare_features(data, feature_info):
    """
    Prepare features for model prediction
    
    Parameters:
    data (DataFrame): Data to prepare
    feature_info (dict): Information for preprocessing
    
    Returns:
    X (ndarray): Prepared features
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

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Load model from file
        model_path = 'models/sector_income_model.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data
        else:
            st.warning(f"Model file not found at {model_path}. Using a demo model instead.")
            return create_demo_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_demo_model()

def create_demo_model():
    """Create a demo model for testing when the real model isn't available"""
    # Use actual medians from your model output
    sector_medians = {1: 15507.77, 2: 21813.77}  # Rural, Urban (per your model output)
    
    # Feature importances from your model output
    # These will be used for visualization and explanation
    feature_importances = {
        1: {  # Lower Rural
            "HH Size (For FDQ)": 0.2727,
            "Is_HH_Have_Motorcycle_scooter": 0.0510,
            "Is_HH_Have_Mobile_handset": 0.0425,
            "avg_age": 0.0308,
            "internet_users_count": 0.0286,
            "NIC_5D": 0.0275,
            "avg_education": 0.0272,
            "person_count": 0.0268,
            "max_age": 0.0263,
            "min_age": 0.0256
        },
        2: {  # Upper Rural
            "Is_HH_Have_Motorcar_jeep_van": 0.0794,
            "person_count": 0.0455,
            "avg_education": 0.0451,
            "avg_age": 0.0421,
            "No. of meals taken during last 30 days on payment_sum": 0.0388,
            "No. of meals taken during last 30 days at home_sum": 0.0356,
            "min_age": 0.0348,
            "max_age": 0.0339,
            "internet_users_count": 0.0327,
            "Is_HH_Have_Refrigerator": 0.0304
        },
        3: {  # Lower Urban
            "HH Size (For FDQ)": 0.1638,
            "internet_users_count": 0.1072,
            "Is_HH_Have_Refrigerator": 0.0603,
            "NIC_5D": 0.0333,
            "avg_education": 0.0323,
            "avg_age": 0.0307,
            "NCO_3D": 0.0284,
            "Is_HH_Have_Motorcycle_scooter": 0.0283,
            "max_age": 0.0281,
            "min_age": 0.0281
        },
        4: {  # Upper Urban
            "Is_HH_Have_Motorcar_jeep_van": 0.0969,
            "NIC_5D": 0.0525,
            "NCO_3D": 0.0391,
            "No. of meals taken during last 30 days at home_sum": 0.0383,
            "avg_education": 0.0382,
            "Is_HH_Have_Laptop_PC": 0.0379,
            "No. of meals taken during last 30 days on payment_sum": 0.0379,
            "min_age": 0.0376,
            "avg_age": 0.0356,
            "max_age": 0.0336
        }
    }
    
    # Create a simple feature info structure
    feature_info = {
        'categorical_cols': ['Sector', 'State', 'Religion', 'Social Group'],
        'numerical_cols': ['HH Size (For FDQ)', 'education_score', 'internet_usage'],
        'encoders': {},
        'scaler': StandardScaler(),
        'sector_medians': sector_medians,
        'feature_importances': feature_importances
    }
    
    # Create dummy encoders
    for col in feature_info['categorical_cols']:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Fit with some dummy data
        dummy_data = pd.DataFrame({col: [1, 2, 3, 4]})
        encoder.fit(dummy_data)
        feature_info['encoders'][col] = encoder
    
    # Create dummy models
    models = {}
    for class_id in range(1, 5):
        models[class_id] = RandomForestRegressor(n_estimators=10, random_state=42)
    
    return {'models': models, 'feature_info': feature_info}

# Load the model
model_data = load_model()
models = model_data['models']
feature_info = model_data['feature_info']
sector_medians = feature_info['sector_medians']
feature_importances = feature_info.get('feature_importances', {})

# Function to predict MPCE
def predict_mpce(features_df):
    """
    Predict MPCE using the sector-income class models
    
    Parameters:
    features_df (DataFrame): Features for prediction
    
    Returns:
    dict: Prediction results
    """
    sector = features_df['Sector'].iloc[0]
    
    # Determine which models to use based on sector
    if sector == 1:  # Rural
        class_options = [1, 2]  # Lower rural, Upper rural
    elif sector == 2:  # Urban
        class_options = [3, 4]  # Lower urban, Upper urban
    else:  # Both sectors (this is new functionality)
        # Create separate predictions for rural and urban
        rural_df = features_df.copy()
        rural_df['Sector'] = 1
        urban_df = features_df.copy()
        urban_df['Sector'] = 2
        
        rural_result = predict_mpce(rural_df)
        urban_result = predict_mpce(urban_df)
        
        # Calculate combined (weighted average)
        # You could adjust the weights based on rural/urban population ratio
        rural_weight = 0.65  # Example: 65% rural
        urban_weight = 0.35  # Example: 35% urban
        combined = rural_result['prediction'] * rural_weight + urban_result['prediction'] * urban_weight
        
        return {
            'rural': rural_result['prediction'],
            'urban': urban_result['prediction'],
            'combined': combined,
            'class': None
        }
    
    # Prepare features
    try:
        X = prepare_features(features_df, feature_info)
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        # Return a default prediction based on national averages
        if sector == 1:  # Rural
            return {'prediction': national_average["rural"], 'class': None}
        elif sector == 2:  # Urban
            return {'prediction': national_average["urban"], 'class': None}
        else:
            return {'prediction': 0.65 * national_average["rural"] + 0.35 * national_average["urban"], 'class': None}
    
    # Try predictions from each applicable model
    class_predictions = {}
    for class_id in class_options:
        if class_id in models:
            pred = models[class_id].predict(X)[0]
            class_predictions[class_id] = pred
    
    if not class_predictions:
        # No valid models for this household, use state averages
        state_id = features_df['State'].iloc[0]
        if state_id in state_mpce_data:
            if sector == 1:  # Rural
                return {'prediction': state_mpce_data[state_id]["rural"], 'class': None}
            else:  # Urban
                return {'prediction': state_mpce_data[state_id]["urban"], 'class': None}
        else:
            # Use national average if state not found
            if sector == 1:  # Rural
                return {'prediction': national_average["rural"], 'class': None}
            else:  # Urban
                return {'prediction': national_average["urban"], 'class': None}
    
    # Rule-based approach to select the most likely class
    # If prediction is above the sector median, use upper model, else lower
    sector_median = sector_medians[sector]
    
    # Average all predictions as a baseline
    avg_prediction = sum(class_predictions.values()) / len(class_predictions)
    
    # Compare to sector median to refine the class
    if avg_prediction <= sector_median:
        # Lower income
        selected_class = 1 if sector == 1 else 3
    else:
        # Upper income
        selected_class = 2 if sector == 1 else 4
    
    # Check if we have a model for the selected class
    if selected_class in class_predictions:
        final_prediction = class_predictions[selected_class]
    else:
        # Use average if selected model is not available
        final_prediction = avg_prediction
    
    return {'prediction': final_prediction, 'class': selected_class}

# Sidebar filters
st.sidebar.markdown("## Household Filters")
st.sidebar.markdown("Adjust these parameters to see how they affect the predicted MPCE.")

# Location filters
st.sidebar.markdown("### Location")
sector = st.sidebar.selectbox(
    "Sector", 
    options=[1, 2, 3], 
    format_func=lambda x: "Rural" if x == 1 else ("Urban" if x == 2 else "Both (Rural & Urban)"),
    index=0
)

state_options = {
    1: "Jammu & Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh (U.T.)",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Dadra & Nagar Haveli and Daman & Diu",
    26: "Telangana",
    27: "Maharashtra",
    28: "Andhra Pradesh",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep (U.T.)",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry (U.T.)",
    35: "Andaman & Nicobar Islands (U.T.)",
    36: "Ladakh (U.T.)"
}


state = st.sidebar.selectbox(
    "State", 
    options=list(state_options.keys()),
    format_func=lambda x: state_options[x],
    index=13  # Maharashtra
)

# Socioeconomic filters
st.sidebar.markdown("### Socioeconomic Characteristics")

religion_options = {
    1: "Hinduism",
    2: "Islam",
    3: "Christianity",
    4: "Sikhism",
    5: "Jainism",
    6: "Buddhism",
    7: "Zoroastrianism",
    9: "Others",
    0: "Not reported"
}

religion = st.sidebar.selectbox(
    "Religion", 
    options=list(religion_options.keys()),
    format_func=lambda x: religion_options[x],
    index=0
)

social_group_options = {
    1: "Scheduled Tribe (ST)",
    2: "Scheduled Caste (SC)",
    3: "Other Backward Class (OBC)",
    9: "Others",
    0: "Not reported"
}

social_group = st.sidebar.selectbox(
    "Social Group", 
    options=list(social_group_options.keys()),
    format_func=lambda x: social_group_options[x],
    index=2
)

hh_size = st.sidebar.slider(
    "Household Size", 
    min_value=1, 
    max_value=15, 
    value=4,
    help="Number of members in the household"
)

# Assets ownership
st.sidebar.markdown("### Asset Ownership")
assets = {}

asset_types = [
    "Television", "Radio", "Laptop_PC", "Mobile_handset", 
    "Bicycle", "Motorcycle_scooter", "Motorcar_jeep_van", 
    "Refrigerator", "Washing_machine", "Airconditioner_aircooler"
]

# Create two columns for asset selection
col1, col2 = st.sidebar.columns(2)

for i, asset in enumerate(asset_types):
    if i % 2 == 0:
        assets[f"Is_HH_Have_{asset}"] = col1.checkbox(
            asset.replace("_", " "), 
            value=True if asset in ["Mobile_handset", "Television"] else False
        )
    else:
        assets[f"Is_HH_Have_{asset}"] = col2.checkbox(
            asset.replace("_", " "), 
            value=True if asset in ["Mobile_handset", "Television"] else False
        )

# Education level 
st.sidebar.markdown("### Education Level")
education_level = st.sidebar.slider(
    "Average Education Level of Household", 
    min_value=0, 
    max_value=17, 
    value=8,
    help="Average years of education of adult household members"
)

# Internet usage
internet_usage = st.sidebar.slider(
    "Proportion of Household Members Using Internet", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.4,
    step=0.1,
    help="Proportion of household members who used internet in the last 30 days"
)

# Online purchase behavior
st.sidebar.markdown("### Online Purchase Behavior")
online_categories = [
    "Clothing", "Footwear", "Mobile_Handset", "Personal_Goods"
]

online_purchases = {}
for category in online_categories:
    online_purchases[f"Is_online_{category}_Purchased_Last365"] = st.sidebar.checkbox(
        f"Purchased {category.replace('_', ' ')} online", 
        value=True if category == "Mobile_Handset" else False
    )

# Prepare input data for prediction
def prepare_input_data():
    data = {
        'Sector': [sector],
        'State': [state],
        'Religion of the head of the household': [religion],
        'Social Group of the head of the household': [social_group],
        'HH Size (For FDQ)': [hh_size],
        'education_score': [education_level],
        'internet_usage': [internet_usage]
    }
    
    # Add assets
    for asset_key, asset_value in assets.items():
        data[asset_key] = [1 if asset_value else 0]
    
    # Add online purchases
    for purchase_key, purchase_value in online_purchases.items():
        data[purchase_key] = [1 if purchase_value else 0]
    
    # Ensure columns match what's expected in feature_info
    needed_cols = feature_info['categorical_cols'] + feature_info['numerical_cols']
    for col in needed_cols:
        if col not in data:
            data[col] = [0]  # Default value for missing columns
    
    return pd.DataFrame(data)

# Make prediction
input_data = prepare_input_data()
prediction_result = predict_mpce(input_data)

# Main content
if sector == 3:  # Both Rural and Urban
    st.markdown('<h2 class="sub-header">Predicted Monthly Per Capita Expenditure by Sector</h2>', unsafe_allow_html=True)
    
    # Create columns for rural and urban
    col_rural, col_urban, col_combined = st.columns(3)
    
    with col_rural:
        st.markdown(f"""
        <div class="prediction-box" style="background-color: #F0FFF4;">
            <p>Rural Sector</p>
            <div class="prediction-value">₹{prediction_result['rural']:,.2f}</div>
            <p>estimated monthly per capita expenditure</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_urban:
        st.markdown(f"""
        <div class="prediction-box" style="background-color: #F0F9FF;">
            <p>Urban Sector</p>
            <div class="prediction-value">₹{prediction_result['urban']:,.2f}</div>
            <p>estimated monthly per capita expenditure</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_combined:
        st.markdown(f"""
        <div class="prediction-box" style="background-color: #F5F3FF;">
            <p>Combined (Population-Weighted Average)</p>
            <div class="prediction-value">₹{prediction_result['combined']:,.2f}</div>
            <p>estimated monthly per capita expenditure</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Setting the main prediction value for later use
    prediction = prediction_result['combined']
    
    # Visual comparison chart
    st.markdown('<h3 class="sub-header">Rural-Urban Comparison</h3>', unsafe_allow_html=True)
    
    comparison_data = pd.DataFrame({
        'Sector': ['Rural', 'Urban', 'Combined'],
        'MPCE': [prediction_result['rural'], prediction_result['urban'], prediction_result['combined']]
    })
    
    fig = px.bar(
        comparison_data,
        x='Sector',
        y='MPCE',
        color='Sector',
        color_discrete_map={
            'Rural': '#10B981',
            'Urban': '#3B82F6',
            'Combined': '#8B5CF6'
        },
        text_auto='.2s'
    )
    
    fig.update_layout(
        title='Rural-Urban MPCE Difference',
        xaxis_title='',
        yaxis_title='Monthly Per Capita Expenditure (₹)',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Urban-Rural differences analysis
    urban_rural_ratio = prediction_result['urban'] / prediction_result['rural']
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### Rural-Urban Insights")
    
    st.markdown(f"""
    - Urban MPCE is **{urban_rural_ratio:.2f}x** higher than Rural MPCE in {state_options[state]}
    - The urban-rural expenditure gap reflects differences in:
        - Cost of living
        - Income opportunities
        - Access to markets and services
        - Consumption patterns
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:  # Single sector (Rural or Urban)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Predicted Monthly Per Capita Expenditure</h2>', unsafe_allow_html=True)
        
        # Display prediction
        prediction = prediction_result['prediction']
        class_name = {
            1: "Lower Rural",
            2: "Upper Rural",
            3: "Lower Urban",
            4: "Upper Urban"
        }.get(prediction_result['class'], "")
        
        st.markdown(f"""
        <div class="prediction-box">
            <p>Based on the selected household characteristics</p>
            <div class="prediction-value">₹{prediction:,.2f}</div>
            <p>estimated monthly per capita expenditure</p>
            <p><strong>Income Class: {class_name}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Expenditure breakdown (shown for both options)
col1, col2 = st.columns([2, 1])

with col1:
    if sector != 3:  # Only show if not already shown above
        st.markdown('<h2 class="sub-header">Estimated Expenditure Breakdown</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="sub-header">Estimated Expenditure Breakdown (Combined)</h2>', unsafe_allow_html=True)
    
    # Categories for expenditure breakdown
    categories = [
        "Food", "Clothing", "Housing", "Education", 
        "Healthcare", "Transportation", "Communication", "Others"
    ]
    
    # Simplified logic to generate breakdown - updated with feature importance data
    current_sector = 1 if sector == 1 else (2 if sector == 2 else 3)  # 3 is combined
    
    # Base food expenditure varies by sector
    base_food = 0.42 if current_sector == 1 else (0.35 if current_sector == 2 else 0.38)  # Rural, Urban, Combined
    
    # Asset ownership affects food share (negative correlation)
    total_assets = sum([1 for a in assets.values() if a])
    asset_impact = 0.03 * (total_assets / len(assets))
    
    # Education level affects education expenditure (positive correlation)
    education_impact = 0.01 * education_level / 10
    
    # Transportation impact by vehicle ownership
    transport_impact = 0.02 * sum([assets[a] for a in assets if a in [
        "Is_HH_Have_Motorcycle_scooter", "Is_HH_Have_Motorcar_jeep_van"]])
    
    # Communication impact by internet usage
    communication_impact = 0.01 * internet_usage
    
    # Define base shares and apply impacts
    shares = {
        "Food": max(0.20, base_food * (1 - asset_impact)),
        "Clothing": 0.07,
        "Housing": 0.15 if current_sector == 1 else (0.25 if current_sector == 2 else 0.20),
        "Education": 0.05 + education_impact,
        "Healthcare": 0.08,
        "Transportation": 0.06 + transport_impact,
        "Communication": 0.05 + communication_impact,
        "Others": 0.12
    }
    
    # Normalize to ensure sum is 1
    total = sum(shares.values())
    shares = {k: v/total for k, v in shares.items()}
    
    # Calculate values
    values = {k: v * prediction for k, v in shares.items()}
    
    # Create DataFrame for plotting
    df_exp = pd.DataFrame({
        'Category': list(values.keys()),
        'Amount': list(values.values()),
        'Percentage': [f"{v*100:.1f}%" for v in shares.values()]
    })
    
    # Plot
    fig = px.bar(
        df_exp, 
        x='Category', 
        y='Amount',
        text='Percentage',
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='Monthly Expenditure (₹)',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)