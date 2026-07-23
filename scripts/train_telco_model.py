# train_telco_model.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def prepare_telco_features_labels(df):
    print("✅ Starting Telco data preprocessing...")

    # Convert 'Churn' column to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop unnecessary columns
    features = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # One-hot encode categorical variables
    features = pd.get_dummies(features)

    labels = df['Churn']

    print("📊 Preprocessing complete. Feature count:", features.shape[1])
    return features, labels


if __name__ == '__main__':
    print("🚀 Telco script started running!")  # Confirms execution

    print("📥 Loading cleaned Telco dataset...")
    telco_df = pd.read_csv('processed/telco_clean.csv')
    print("✅ Telco data loaded successfully with shape:", telco_df.shape)

    # Run preprocessing
    X, y = prepare_telco_features_labels(telco_df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print("🚀 Training model...")

    # XGBoost model
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)

    print("\n✅ Evaluation Results:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/telco_xgb_model.pkl")
    print("💾 Telco model saved successfully to models/telco_xgb_model.pkl")

    print("\n🎉 Telco model training complete!")
