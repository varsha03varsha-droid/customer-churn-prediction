# evaluate_ecommerce_model.py

import matplotlib
matplotlib.use('TkAgg')  # Ensures interactive plots open properly
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_curve,
    auc
)
from xgboost import plot_importance
import joblib
import numpy as np

print("📥 Loading E-commerce churn model...")
model = joblib.load("models/xgb_churn_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# === Step 1: Load dataset ===
df = pd.read_csv("processed/merged_LRFM_labeled.csv")

# Drop unused columns
X = df.drop(columns=["churn_label", "user_id"], errors="ignore")
y = label_encoder.transform(df["churn_label"])  # numeric labels

print(f"✅ Dataset loaded. Shape: {X.shape}")

# === Step 2: Predict & Evaluate ===
print("\n🚀 Evaluating E-commerce churn model...")
y_pred = model.predict(X)

# Human-readable labels
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y)

# === Step 3: Classification Report ===
print("\n🔹 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
print("✅ Accuracy:", round(accuracy_score(y_true_labels, y_pred_labels) * 100, 2), "%")

# === Step 4: Confusion Matrix (Mandatory Graph #7) ===
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - E-commerce Model")
plt.show(block=True)

# === Step 5: Feature Importance ===
plt.figure(figsize=(8, 6))
plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances - E-commerce Model")
plt.show(block=True)

# === Step 6: ROC Curve (Mandatory Graph #8) ===
print("\n📈 Generating ROC Curve...")

# XGBoost outputs probability for each class: shape (n_samples, 3)
y_proba = model.predict_proba(X)

plt.figure(figsize=(8, 6))

for idx, class_label in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y == idx, y_proba[:, idx])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.title("ROC Curve - Multi-Class XGBoost Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show(block=True)

print("🎉 E-commerce model evaluation and all mandatory graphs generated!")
