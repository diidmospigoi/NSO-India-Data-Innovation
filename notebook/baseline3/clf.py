import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle  

train_df = pd.read_csv("data\\final_train_v3.csv")
test_df = pd.read_csv("data\\final_test_v3.csv")

threshold = train_df['Total_Expense'].quantile(0.95)
train_df['is_top_5_percent'] = (train_df['Total_Expense'] >= threshold).astype(int)


X = train_df.drop(columns=['Total_Expense', 'is_top_5_percent'])
y = train_df['is_top_5_percent']

X = X.drop(columns=[col for col in X.columns if 'id' in col.lower()], errors='ignore')
test_X = test_df[X.columns]
test_df['is_top_5_percent'] = (test_df['Total_Expense'] >= threshold).astype(int)


model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=6, 
    random_state=42
)

model.fit(X, y)


test_preds = model.predict(test_X)


y_test = test_df['is_top_5_percent']
accuracy = accuracy_score(y_test, test_preds)
print(f"✅ Accuracy on Test Set: {accuracy:.4f}")


with open('model\\rf_classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)