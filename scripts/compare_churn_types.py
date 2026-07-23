# compare_churn_types.py
"""
Run this to:
 - Compute class-wise accuracy (Partial / Total / Non)
 - Create confusion matrix
 - Create class-wise accuracy bar chart
 - Create mandatory churn distribution graph
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("processed/merged_LRFM_labeled.csv", index_col=0)

# ================================
# 4️⃣ MANDATORY GRAPH: CHURN DISTRIBUTION
# ================================
plt.figure(figsize=(6,4))
sns.countplot(x="churn_label", data=df, palette="Blues")
plt.title("Churn Distribution (Mandatory Graph)")
plt.xlabel("Churn Label (0 = No Churn, 1 = Partial, 2 = Total)")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show(block=True)

print("📊 Churn Distribution graph generated successfully!")

# ---------------------------
# Prepare features and labels
# ---------------------------
X = df.drop(columns=["churn_label", "user_id"], errors="ignore")
y = df["churn_label"].astype(str)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

# Train XGBoost
model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# ---------------------------
# Class-wise accuracy
# ---------------------------
overall_acc = accuracy_score(y_test, y_pred)

class_accuracy = {}
for cls in le.classes_:
    mask = (y_test_labels == cls)
    if mask.sum() == 0:
        class_accuracy[cls] = np.nan
    else:
        class_accuracy[cls] = accuracy_score(y_test_labels[mask], y_pred_labels[mask])

metrics_df = pd.DataFrame([
    {"Class": cls, "Class_Accuracy(%)": round(class_accuracy[cls]*100,2)
     if not np.isnan(class_accuracy[cls]) else None}
    for cls in le.classes_
])

metrics_df.loc[len(metrics_df)] = {
    "Class": "Overall",
    "Class_Accuracy(%)": round(overall_acc*100,2)
}

print("\nClass-wise accuracy:\n")
print(metrics_df.to_string(index=False))

metrics_df.to_csv("processed/churn_type_metrics.csv", index=False)
print("\nSaved churn type metrics to processed/churn_type_metrics.csv")

# ---------------------------
# Classification Report
# ---------------------------
print("\nFull classification report:\n")
print(classification_report(y_test_labels, y_pred_labels))

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Churn Types)")
plt.tight_layout()
plt.show(block=True)

# ---------------------------
# Class-Wise Accuracy Bar Graph
# ---------------------------
plt.figure(figsize=(6,4))
plt.bar(metrics_df['Class'][:-1], metrics_df['Class_Accuracy(%)'][:-1],
        color=['#4CAF50', '#FF9800', '#F44336'])
plt.ylim(0,100)
plt.ylabel("Accuracy (%)")
plt.title("Class-wise Accuracy (Partial / Total / Non-Churn)")
plt.tight_layout()
plt.show(block=True)
