import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory add store for user records
user_records = {}

# In-memory log for failed submissions
failed_submissions_log = []

def log_failed_submission(info):
    """Append a failed submission to the in-memory log (max 100 entries)."""
    if len(failed_submissions_log) > 100:
        failed_submissions_log.pop(0)
    failed_submissions_log.append(info)

def prepare_features(data, feature_info):
    """Prepare features for model prediction using encoders and scaler."""
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

# Load models and feature_info at startup and store globally
reg_models = None
feature_info = None
classifier = None
models_ready_event = asyncio.Event()

async def async_load_models():
    """Asynchronously load ML models and feature info from disk (only once)."""
    global reg_models, feature_info, classifier
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'Models', 'sector_income_model.pkl')
    model_data = await asyncio.to_thread(joblib.load, model_path)
    reg_models = model_data['models']
    feature_info = model_data['feature_info']
    model_path = os.path.join(base_dir, 'Models', 'sector_income_classifiers_tuned.pkl')
    classifier = await asyncio.to_thread(joblib.load, model_path)
    models_ready_event.set()

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(async_load_models())

@app.get("/")
async def landing():
#     """Render the landing page with options."""
    path = os.path.join( 'index.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/submit")
async def submit_form(request: Request):
    """Receive and store the latest survey record from frontend with server-side validation."""
    try:
        data = await request.json()
    except Exception as e:
        log_failed_submission({'error': 'Malformed JSON', 'exception': str(e)})
        raise HTTPException(status_code=400, detail="Malformed JSON")

    record_key = data.get('key')
    record = data.get('record')

    # Server-side required field/type validation
    required_fields = [
        'sector', 'state', 'region', 'district', 'household_type', 'religion', 'caste', 'hh_size',
        'max_age', 'min_age', 'gender_male', 'gender_female', 'gender_others',
        'max_edu', 'min_edu', 'meals_daily','meals_school', 'meals_employer',
        'meals_payment', 'meals_home', 'internet_users'
    ]
    missing = [f for f in required_fields if f not in record or record[f] in [None, '', []]]
    if missing:
        # Log missing required fields
        log_failed_submission({'error': 'Missing required fields', 'fields': missing, 'record': record})
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")

    # Type and value checks for gender and household size
    try:
        hh_size = int(record['hh_size'])
        gender_male = int(record['gender_male'])
        gender_female = int(record['gender_female'])
        gender_others = int(record['gender_others'])
    except Exception as e:
        # Log invalid type errors
        log_failed_submission({'error': 'Invalid type for gender/hh_size', 'exception': str(e), 'record': record})
        raise HTTPException(status_code=400, detail="Invalid type for gender/hh_size")

    # Gender sum check
    if (gender_male + gender_female + gender_others) != hh_size:
        log_failed_submission({'error': 'Gender counts do not sum to household size', 'record': record})
        raise HTTPException(status_code=400, detail="Gender counts do not sum to household size")

    # Remove all previous records and keep only the current one
    user_records.clear()
    user_records[record_key] = record
    return JSONResponse(content={'success': True, 'message': 'Record saved successfully.'})

@app.get("/records")
async def get_records():
    """Return all user records (only one at a time)."""
    return JSONResponse(content=user_records)

@app.get("/process")
async def process_record():
    """Process the current record and return prediction."""
    try:
        await models_ready_event.wait()  # Wait for models to be ready
        result = process_user_records(user_records)
        if result is None:
            return JSONResponse(content={'success': False, 'error': 'No record found'})
        return JSONResponse(content={'success': True, 'result': result})
    except Exception as e:
        log_failed_submission({'error': 'Exception in /process', 'exception': str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/logs")
async def admin_logs():
    """Admin route to view failed submission logs (last 100)."""
    return JSONResponse(content=failed_submissions_log)

def process_user_records(records):
    """
    Process the user_records dictionary.
    Returns prediction and error for the single (current) record.
    """
    if not records:
        return None
    # Get the only record (since only one is kept)
    key, record = next(iter(records.items()))
    # Use sector as integer directly (1 for rural, 2 for urban)
    sector = int(record.get('sector')) if record.get('sector') else None
    state_number = record.get('state')
    region_number = int(record.get('region')) if record.get('region') else None
    district_number = int(record.get('district')) if record.get('district') else None
    household_type_number = int(record.get('household_type')) if record.get('household_type') else None
    religion_number = int(record.get('religion')) if record.get('religion') else None
    caste_number = int(record.get('caste')) if record.get('caste') else None
    hh_size = int(record.get('hh_size')) if record.get('hh_size') else 1
    nco = int(record.get('nco')) if record.get('nco') else 0
    nic = int(record.get('nic')) if record.get('nic') else 0

    # Online purchases (convert to binary list)
    online_data = record.get('online_purchases', [])
    if isinstance(online_data, str):
        online_data = [online_data]
    purchase_list = ['clothing', 'footwear', 'furniture', 'mobile', 'personal', 'recreation', 'appliances', 'crockery', 'sports', 'medical', 'bedding']
    online_answers = [1 if item in online_data else 0 for item in purchase_list]

    # Assets (convert to binary list)
    asset_data = record.get('assets', [])
    if isinstance(asset_data, str):
        asset_data = [asset_data]
    assets_list = ['television', 'radio', 'laptop', 'mobile', 'bicycle', 'motorcycle', 'car', 'trucks', 'cart', 'refrigerator', 'washing_machine', 'cooler']
    asset_answers = [1 if item in asset_data else 0 for item in assets_list]

    person_count = hh_size
    max_age = int(record.get('max_age', 1)) if record.get('max_age') else 1
    min_age = int(record.get('min_age', 0)) if record.get('min_age') else 0
    avg_age = (max_age + min_age) / 2
    gender_1 = int(record.get('gender_male', 0))
    gender_2 = int(record.get('gender_female', 0))
    gender_3 = int(record.get('gender_others', 0))
    max_edu = int(record.get('max_edu', 0)) + 1
    min_edu = int(record.get('min_edu', 0)) + 1
    avg_edu = (max_edu + min_edu) / 2
    meal1_sum = int(record.get('meals_daily', 0))
    meal1_mean = meal1_sum / person_count if person_count > 0 else 0
    meal2_sum = int(record.get('meals_school', 0))
    meal2_mean = meal2_sum / person_count if person_count > 0 else 0
    meal3_sum = int(record.get('meals_employer', 0))
    meal3_mean = meal3_sum / person_count if person_count > 0 else 0
    meal4_sum = int(record.get('meals_payment', 0))
    meal4_mean = meal4_sum / person_count if person_count > 0 else 0
    meal5_sum = int(record.get('meals_home', 0))
    meal5_mean = meal5_sum / person_count if person_count > 0 else 0
    internet_users_count = int(record.get('internet_users', 0))

    global reg_models, feature_info, classifier

    # Build DataFrame for prediction
    new_row = pd.DataFrame([{
        "Sector": sector,
        "State": state_number,
        "NSS-Region": region_number,
        "District": district_number,
        "Household Type": household_type_number,
        "Religion of the head of the household": religion_number,
        "Social Group of the head of the household": caste_number,
        "HH Size (For FDQ)": hh_size,
        "NCO_3D": nco,
        "NIC_5D": nic,
        "Is_online_Clothing_Purchased_Last365": online_answers[0],
        "Is_online_Footwear_Purchased_Last365": online_answers[1],
        "Is_online_Furniture_fixturesPurchased_Last365": online_answers[2],
        "Is_online_Mobile_Handset_Purchased_Last365": online_answers[3],
        "Is_online_Personal_Goods_Purchased_Last365": online_answers[4],
        "Is_online_Recreation_Goods_Purchased_Last365": online_answers[5],
        "Is_online_Household_Appliances_Purchased_Last365": online_answers[6],
        "Is_online_Crockery_Utensils_Purchased_Last365": online_answers[7],
        "Is_online_Sports_Goods_Purchased_Last365": online_answers[8],
        "Is_online_Medical_Equipment_Purchased_Last365": online_answers[9],
        "Is_online_Bedding_Purchased_Last365": online_answers[10],
        "Is_HH_Have_Television": asset_answers[0],
        "Is_HH_Have_Radio": asset_answers[1],
        "Is_HH_Have_Laptop_PC": asset_answers[2],
        "Is_HH_Have_Mobile_handset": asset_answers[3],
        "Is_HH_Have_Bicycle": asset_answers[4],
        "Is_HH_Have_Motorcycle_scooter": asset_answers[5],
        "Is_HH_Have_Motorcar_jeep_van": asset_answers[6],
        "Is_HH_Have_Trucks": asset_answers[7],
        "Is_HH_Have_Animal_cart": asset_answers[8],
        "Is_HH_Have_Refrigerator": asset_answers[9],
        "Is_HH_Have_Washing_machine": asset_answers[10],
        "Is_HH_Have_Airconditioner_aircooler": asset_answers[11],
        "person_count": person_count,
        "avg_age": avg_age,
        "max_age": max_age,
        "min_age": min_age,
        "gender_1_count": gender_1,
        "gender_2_count": gender_2,
        "gender_3_count": gender_3,
        "avg_education": avg_edu,
        "max_education": max_edu,
        "No. of meals usually taken in a day_sum": meal1_sum,
        "No. of meals usually taken in a day_mean": meal1_mean,
        "No. of meals taken during last 30 days from school, balwadi etc._sum": meal2_sum,
        "No. of meals taken during last 30 days from school, balwadi etc._mean": meal2_mean,
        "No. of meals taken during last 30 days from employer as perquisites or part of wage_sum": meal3_sum,
        "No. of meals taken during last 30 days from employer as perquisites or part of wage_mean": meal3_mean,
        "No. of meals taken during last 30 days on payment_sum": meal4_sum,
        "No. of meals taken during last 30 days on payment_mean": meal4_mean,
        "No. of meals taken during last 30 days at home_sum": meal5_sum,
        "No. of meals taken during last 30 days at home_mean": meal5_mean,
        "internet_users_count": internet_users_count
    }])
    X = prepare_features(new_row, feature_info)

    # Determine class for prediction
    sec = new_row['Sector'].iloc[0]
    classifier_model = classifier['clf_rural'] if sec == 1 else classifier['clf_urban']
    inferred_class = classifier_model.predict(X)[0]

    inferred_class = inferred_class+1 if sec == 1 else inferred_class+3

    model = reg_models[inferred_class]
    pred = model.predict(X)[0]

    return {
        "predicted_expense": pred
    }