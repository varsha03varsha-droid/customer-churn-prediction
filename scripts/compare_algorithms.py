import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

print("\n🚀 Starting Hybrid Model Comparison...\n")

# ======================================================
# LOAD DATASET (Behavioral enhanced preferred)
# ======================================================
DATA_PATH = "processed/merged_LRFM_behavioral.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "processed/merged_LRFM_labeled.csv"

df = pd.read_csv(DATA_PATH, index_col=0)

# ======================================================
# Prepare features and labels
# ======================================================
X = df.drop(columns=["churn_label", "user_id"], errors="ignore")
y = df["churn_label"].astype(str)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc
)

# ======================================================
# Base ML models (including MLP Neural Network)
# ======================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

results = []
fitted_models = {}

# ======================================================
# Train and evaluate base models
# ======================================================
for name, model in models.items():
    print(f"Training & evaluating: {name} ...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2)
    })

    fitted_models[name] = model

# ======================================================
# Hybrid Stacking Ensemble
# ======================================================
print("\n🌟 Building Hybrid Stacking Ensemble...\n")

probas_train = []
probas_test = []

for name, model in fitted_models.items():
    if hasattr(model, "predict_proba"):
        probas_train.append(model.predict_proba(X_train))
        probas_test.append(model.predict_proba(X_test))
    else:
        probas_train.append(model.predict(X_train).reshape(-1, 1))
        probas_test.append(model.predict(X_test).reshape(-1, 1))

# Stack probabilities horizontally
stack_X_train = np.hstack(probas_train)
stack_X_test = np.hstack(probas_test)

# Meta learner
meta_model = LogisticRegression(max_iter=1500)
meta_model.fit(stack_X_train, y_train)
y_meta_pred = meta_model.predict(stack_X_test)

# Evaluate Ensemble
results.append({
    "Model": "Hybrid Stacking Ensemble",
    "Accuracy": round(accuracy_score(y_test, y_meta_pred) * 100, 2),
    "Precision": round(precision_score(y_test, y_meta_pred, average="weighted", zero_division=0) * 100, 2),
    "Recall": round(recall_score(y_test, y_meta_pred, average="weighted", zero_division=0) * 100, 2),
    "F1 Score": round(f1_score(y_test, y_meta_pred, average="weighted", zero_division=0) * 100, 2),
})

# ======================================================
# Display & Save Results
# ======================================================
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n📊 Final Model Comparison:\n")
print(results_df.to_string(index=False))

os.makedirs("processed", exist_ok=True)
results_df.to_csv("processed/model_comparison_results.csv", index=False)
print("\n💾 Saved model comparison to processed/model_comparison_results.csv")

# ======================================================
# 5️⃣ GRAPH: Model Accuracy Comparison (MANDATORY)
# ======================================================
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["Accuracy"], color="skyblue")
plt.title("Model Accuracy Comparison (%)")
plt.ylabel("Accuracy %")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# ======================================================
# 6️⃣ GRAPH: Precision–Recall–F1 Comparison (MANDATORY)
# ======================================================
metrics = ["Precision", "Recall", "F1 Score"]

plt.figure(figsize=(10, 5))

for index, row in results_df.iterrows():
    plt.plot(metrics,
             [row["Precision"], row["Recall"], row["F1 Score"]],
             marker="o",
             label=row["Model"])

plt.title("Precision–Recall–F1 Comparison")
plt.ylabel("Score (%)")
plt.ylim(0, 110)
plt.legend(loc="lower right", fontsize=8)
plt.grid(False)
plt.tight_layout()
plt.show()
