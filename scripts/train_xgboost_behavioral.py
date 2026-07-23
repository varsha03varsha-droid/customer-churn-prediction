# train_xgboost_behavioral.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

INPUT = "processed/merged_LRFM_behavioral.csv"  # new file
MODEL_OUT = "models/xgb_churn_model_behavioral.pkl"
LE_OUT = "models/label_encoder_behavioral.pkl"

df = pd.read_csv(INPUT, index_col=0)

if 'churn_label' not in df.columns:
    raise KeyError("churn_label not found in dataset")

X = df.drop(columns=['churn_label', 'user_id'], errors='ignore')
y = df['churn_label']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
print("Training XGBoost on behavioral dataset...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(le, LE_OUT)
print("Saved:", MODEL_OUT, LE_OUT)
