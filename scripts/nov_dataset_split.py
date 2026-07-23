import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Input file (NOVEMBER DATASET)
INPUT_PATH = "processed/nov_clean.csv"

# Output folder
OUTPUT_DIR = "dataset_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("📥 Loading November dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Total rows loaded: {len(df)}")

# Train-test split (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

# Save files
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Train/Test split completed successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")
print(f"📁 Files saved in folder: {OUTPUT_DIR}")
print(f" - {train_path}")
print(f" - {test_path}")

