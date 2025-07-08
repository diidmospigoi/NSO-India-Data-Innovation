# MPCE Prediction Pipeline - User Guide

## Overview

This project predicts Monthly Per Capita Expenditure (MPCE) for Indian households using machine learning. The pipeline includes data preparation, model training (with sector and income class stratification), evaluation, and frontend interfaces for predictions and insights.

---

## Project Structure

```
.
├── main.py                # Main training script (RandomForest pipeline)
├── classifier.py          # Income class assignment and classifier training
├── eval.py                # Evaluation utilities and scripts
├── xgboost_utils.py       # XGBoost-based training and evaluation utilities
├── Testing_data.csv       # Test dataset (raw)
├── Training_data.csv      # Training dataset (raw)
├── models_regressor/      # Saved regressor models
├── models_clf/            # Saved classifier models
├── frontend/              # Frontend (FastAPI/Streamlit app, static, templates)
├── testing/               # Unit and integration tests
├── src/                   # Source utilities and scripts
├── eda/                   # Exploratory Data Analysis notebooks
├── notebook/              # Jupyter notebooks (EDA, clustering, experiments)
├── report/                # Reports and documentation
├── results/               # Model outputs and results
└── images/                # Images for documentation and frontend
```

---

## Setup & Installation

1. **Clone the repository:**
    ```sh
    git clone <repo-url>
    cd <repo-directory>
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    For frontend:
    ```sh
    pip install -r frontend/requirements.txt
    ```

3. **Prepare data:**
    - Place `Training_data.csv` and `Testing_data.csv` in the project root or update paths in scripts as needed.

---

## Data Preparation

- Data is loaded from CSV files.
- The function `create_sector_income_classes` in [`main.py`](main.py) assigns each household to one of four classes based on sector (rural/urban) and whether their total expense is above or below the sector median.
- Missing values in key columns (`NCO_3D`, `NIC_5D`) are dropped.
- Class distribution and sector medians are printed for transparency.

---

## Model Training

- The main training script is [`main.py`](main.py).
- The function `train_class_based_models` trains a separate `RandomForestRegressor` for each income class.
- Features are preprocessed using one-hot encoding for categorical variables and scaling for numerical variables.
- Feature importances are displayed for each class.
- Trained models and preprocessing objects are saved to `models/sector_income_randomforestmodel.pkl`.

**To run training:**
```sh
python main.py
```

---

## Evaluation

- Use [`eval.py`](eval.py) and [`testing/`](testing/) scripts to evaluate predictions.
- Metrics computed include RMSE, MAE, MAPE, and R².
- Example evaluation code:
    ```python
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # y_true and y_pred should be numpy arrays
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true) * 100)
    r2 = r2_score(y_true, y_pred)
    ```

---

## Prediction on New Data

- Use the saved model and feature pipeline to predict on new data.
- See [`testing/dummy_test.py`](testing/dummy_test.py) for an example of preparing a test file and generating predictions.

---

## Frontend

- **FastAPI app:** [`frontend/app.py`](frontend/app.py)
    ```sh
    uvicorn frontend.app:app --reload
    ```
- **Streamlit app:** [`frontend/streamlit/app.py`](frontend/streamlit/app.py)
    ```sh
    streamlit run frontend/streamlit/app.py
    ```

---

## Exploratory Data Analysis

- EDA notebooks are in [`eda/`](eda/eda.ipynb) and [`notebook/`](notebook/).
- These provide insights into feature distributions, correlations, and key drivers of MPCE.

---

## Utilities

- [`src/topk_features.py`](src/topk_features.py): Feature importance extraction and visualization.
- Data parsing and analysis scripts in [`src/data_preparation/`](src/data_preparation/).

---

## Results & Reports

- Model outputs, predictions, and performance summaries are saved in [`results/`](results/).
- Reports and documentation in [`report/`](report/).

---

## Customization & Extension

- You can add new features or models by modifying `main.py` or adding new scripts in `src/`.
- For XGBoost-based pipelines, see [`xgboost_utils.py`](xgboost_utils.py).

---

## Troubleshooting

- Ensure all dependencies are installed.
- Check data paths in scripts.
- For errors related to missing columns, verify your CSV files match the expected schema.

---

## Contact

For questions or contributions, please open an issue or contact the maintainers.

---

**End of User