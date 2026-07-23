# evaluate_telco_model.py

import matplotlib
matplotlib.use('TkAgg')  # ✅ Ensures interactive plots open properly
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from xgboost import plot_importance
import joblib

print("📥 Loading Telco churn model...")
model = joblib.load("models/telco_xgb_model.pkl")

# === Step 1: Load dataset ===
df = pd.read_csv("processed/telco_clean.csv")

# Convert churn labels to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(df.drop(columns=["customerID", "Churn"], errors="ignore"))
y = df["Churn"]

print(f"✅ Dataset loaded. Shape: {X.shape}")

# === Step 2: Predict & Evaluate ===
print("\n🚀 Evaluating Telco churn model...")
y_pred = model.predict(X)

# === Step 3: Classification Report ===
print("\n🔹 Classification Report:")
print(classification_report(y, y_pred))
print("✅ Accuracy:", round(accuracy_score(y, y_pred) * 100, 2), "%")

# === Step 4: Confusion Matrix ===
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Telco Model")
plt.show(block=True)

# === Step 5: Feature Importance ===
plt.figure(figsize=(8, 6))
plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances - Telco Model")
plt.show(block=True)

print("🎉 Telco model evaluation and visualization complete!")
