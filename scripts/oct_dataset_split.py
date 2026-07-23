import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Input file (OCTOBER DATASET)
INPUT_PATH = "processed/oct_clean.csv"   # change to oct_LRFM.csv if required

# Output folder
OUTPUT_DIR = "oct_dataset_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📥 Loading October dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Total rows loaded: {len(df)}")

# Train-test split (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

# Save files
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("\n✅ October Train/Test split completed!")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")
print(f"Files saved inside folder: {OUTPUT_DIR}")
