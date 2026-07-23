import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Paths for training & testing datasets
TRAIN_PATH = "datasets/train/training_data.csv"
TEST_PATH = "datasets/test/testing_data.csv"


def train_and_evaluate_xgboost():
    print("📥 Loading training & testing datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape:  {test_df.shape}")

    # -------------------------------------------------------
    # Prepare features and labels
    # -------------------------------------------------------
    X_train = train_df.drop(columns=["churn_label"])
    y_train = train_df["churn_label"]

    X_test = test_df.drop(columns=["churn_label"])
    y_test = test_df["churn_label"]

    # -------------------------------------------------------
    # Encode labels
    # -------------------------------------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\n🔢 Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # -------------------------------------------------------
    # Train XGBoost model
    # -------------------------------------------------------
    print("\n🚀 Training model...")
    model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train_enc)

    # -------------------------------------------------------
    # Evaluate model
    # -------------------------------------------------------
    preds = model.predict(X_test)

    print("\n📊 Accuracy:", accuracy_score(y_test_enc, preds))
    print("\n📘 Classification Report:")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))

    # -------------------------------------------------------
    # Save model + label encoder
    # -------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_churn_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("\n💾 Model saved successfully!")


if __name__ == "__main__":
    train_and_evaluate_xgboost()
