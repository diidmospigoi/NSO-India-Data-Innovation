import pandas as pd

# 1. Load final predictions
preds = pd.read_csv("/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/final_predictions.csv")
print(f"Loaded predictions: {preds.shape[0]} rows")

# 2. Load weights
weights = pd.read_excel("/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/HH_Data_with_Weight.xlsx")       # replace with your actual weights filename
print(f"Loaded weights:     {weights.shape[0]} rows")

# 3. Inner-merge predictions with weights on HH_ID
merged = preds.merge(weights, on="HH_ID", how="inner")
print(f"After merging weights: {merged.shape[0]} rows")

# 4. Load Sector, State, and TotalExpense from testing set
test_meta = pd.read_csv(
    "/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/Testing_data.csv",
    usecols=["HH_ID", "Sector", "State", "TotalExpense"]
)
test_meta.rename(columns={"TotalExpense": "Actual_TotalExpense"}, inplace=True)
print(f"Loaded test metadata: {test_meta.shape[0]} rows")

# 5. Merge in the actual TotalExpense
merged = merged.merge(test_meta, on="HH_ID", how="inner")
print(f"After merging actuals:  {merged.shape[0]} rows")

# 6. Inspect and save
print("\nSample:\n", merged.head())
merged.to_csv("/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/predictions_with_weights_and_actual.csv", index=False)
print("\n✔ Saved merged file → predictions_with_weights_and_actual.csv")

# ─── 1. Load the merged predictions+weights+actual file ───────────────────────
df = pd.read_csv("/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/predictions_with_weights_and_actual.csv")

# ─── 2. Identify the weight column ──────────────────────────────────────────
weight_cols = [c for c in df.columns if "weight" in c.lower()]
if not weight_cols:
    raise ValueError("No weight column found. Please ensure your merged CSV has a weight column.")
weight_col = weight_cols[0]
print(f"Using weight column: {weight_col}")

# ─── 3. Bring in household size from the test file ───────────────────────────
test_df = pd.read_csv("/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/Testing_data.csv")
df = df.merge(
    test_df[["HH_ID", "HH Size (For FDQ)"]],
    on="HH_ID", how="left"
)
if df["HH Size (For FDQ)"].isna().any():
    print(f"Warning: {df['HH Size (For FDQ)'].isna().sum()} HHs missing size.")

# ─── 4. Compute weighted sums ─────────────────────────────────────────────────
# For actual TotalExpense
df["WtdTotalExpense_actual"]  = df["Actual_TotalExpense"] * df[weight_col]
# For predicted TotalExpense
df["WtdTotalExpense_predict"] = df["Predicted_TotalExpense"] * df[weight_col]
# Weighted HH‐size
df["WtdHH_in_FDQ"]            = df["HH Size (For FDQ)"]   * df[weight_col]

# ─── 5. Group by State and compute MPCE via sum‐ratio ────────────────────────
# Build a small helper
def mpce_by(col):
    sums_exp = df.groupby("State")[col].sum()
    sums_size = df.groupby("State")["WtdHH_in_FDQ"].sum()
    return (sums_exp / sums_size).reset_index(name=col.replace("WtdTotalExpense_", "MPCE_"))

pred_mpce = mpce_by("WtdTotalExpense_predict").rename(columns={"MPCE_WtdTotalExpense_predict": "Predicted_MPCE"})
act_mpce  = mpce_by("WtdTotalExpense_actual").rename(columns={"MPCE_WtdTotalExpense_actual":  "Actual_MPCE"})

# ─── 6. Combine and save ─────────────────────────────────────────────────────
state_mpce = pred_mpce.merge(act_mpce, on="State")
state_mpce.to_csv(
    "/content/drive/MyDrive/MPCE_MoSPI/Shubham-IIIT/statewise_MPCE.csv",
    index=False
)
print("✔ Saved statewise MPCE to statewise_MPCE.csv")

# ─── 7. Display ───────────────────────────────────────────────────────────────
print("\nState-wise MPCE:")
print(state_mpce.to_string(index=False))