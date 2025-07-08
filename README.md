# MPCE Prediction Pipeline

This repository contains code and resources for predicting Monthly Per Capita Expenditure (MPCE) using machine learning models, with a focus on sector (rural/urban) and income class stratification. The project includes data preparation, model training, evaluation, and a frontend for user interaction.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Frontend](#frontend)
- [Testing](#testing)
- [Notebooks](#notebooks)
- [Utilities](#utilities)
- [Results & Reports](#results--reports)
- [Contact](#contact)

---

## Project Structure
```
.
├── main.py                      # Main training script (RandomForest pipeline)
├── classifier.py                # Income class assignment and classifier training
├── eval.py                      # Evaluation utilities and scripts
├── xgboost_utils.py             # XGBoost-based training and evaluation utilities
├── Testing_data.csv             # Test dataset (raw)
├── Training_data.csv            # Training dataset (raw)
├── models_regressor/            # Saved regressor models
├── models_clf/                  # Saved classifier models
├── dashboard/                    # Frontend (FastAPI/Streamlit app, static, templates)
│   ├── app.py
│   ├── streamlit/
│   ├── static/
│   ├── templates/
│   └── requirements.txt
├── testing/                     # Unit and integration tests
│   ├── test.py
│   └── dummy_test.py
├── src/                         # Source utilities and scripts
│   ├── topk_features.py
│   └── data_preparation/
├── notebook/                    # Jupyter notebooks (EDA, clustering, experiments)
│   └── baseline2/
├── eda/                         # Exploratory Data Analysis notebooks
├── report/                      # Reports and documentation
├── results/                     # Model outputs and results
└── images/                      # Images for documentation and frontend
```

---

## Setup & Installation

1. **Clone the repository:**
```sh
git clone <repo-url>
cd <repo-directory>
```

2. **Install dependencies:**
- For backend and model training:
```sh
pip install -r frontend/requirements.txt
```
- For Streamlit app:
```sh
pip install streamlit
```

3. **Prepare data:**
- Place `Training_data.csv` and `Testing_data.csv` in the project root or update paths in scripts as needed.

---

## Data Preparation
- Data cleaning, merging, and feature engineering are handled in the `src/data_preparation/` notebooks and scripts.
- Key columns: `Sector`, `State`, `NSS-Region`, `District`, `Household Type`, `Religion of the head of the household`, `Social Group of the head of the household`, `TotalExpense`, `NCO_3D`, `NIC_5D`, etc.

---

## Model Training

- **Main pipeline:** [`main.py`](main.py)
  - Assigns income classes based on sector medians.
  - Trains a separate `RandomForestRegressor` for each sector-income class.
  - Saves models and preprocessing info to [`models_regressor/`](models_regressor/).

- **Classifier pipeline:** [`classifier.py`](classifier.py)
  - Trains classifiers for income class prediction.

- **XGBoost pipeline:** [`xgboost_utils.py`](xgboost_utils.py)
  - Utilities for XGBoost-based regression and evaluation.

---

## Evaluation

- Use [`eval.py`](eval.py) to evaluate predictions against ground truth using metrics like RMSE, MAE, MAPE, and R².
- Evaluation scripts expect predictions and test data with `TotalExpense` for comparison.

---

## Frontend

- **FastAPI app:** [`frontend/app.py`](frontend/app.py)
- **Streamlit app:** [`frontend/streamlit/app.py`](frontend/streamlit/app.py)
- **Templates:** [`frontend/templates/`](frontend/templates/)
- **Static files:** [`frontend/static/`](frontend/static/)

### Running the Frontend
- **FastAPI:**
```sh
uvicorn frontend.app:app --reload
```
- **Streamlit:**
```sh
streamlit run frontend/streamlit/app.py
```

---

## Testing
- Unit and integration tests are in [`testing/`](testing/).
  - [`test.py`](testing/test.py): Batch feature preparation and model testing.
  - [`dummy_test.py`](testing/dummy_test.py): Dummy test file creation and prediction.

---

## Notebooks
- Experiments, clustering, and EDA are in [`notebook/baseline2/`](notebook/baseline2/) and [`eda/`](eda/).
- Notebooks include:
  - Clustering-based regression
  - Stacking/ensemble experiments
  - Feature importance analysis

---

## Utilities
- [`src/topk_features.py`](src/topk_features.py): Feature importance extraction and visualization.
- Data parsing and analysis notebooks in [`src/data_preparation/`](src/data_preparation/).

---

## Results & Reports
- Model outputs, predictions, and performance summaries are saved in [`results/`](results/).
- Reports and documentation in [`report/`](report/).

---

## Contact
For questions or contributions, please open an issue or contact the maintainers.
