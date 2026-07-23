# add_behavioral_features.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT = "processed/merged_LRFM_labeled.csv"
OUTPUT = "processed/merged_LRFM_behavioral.csv"

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT, index_col=0)

# Safe checks for required cols
for col in ["Frequency", "nov_frequency", "Length", "Monetary", "Recency"]:
    if col not in df.columns:
        raise KeyError(f"Required column missing in input: {col}")

# --------------------------
# New behavioral engineered features
# --------------------------
df["activity_drop_ratio"] = (df["Frequency"] - df["nov_frequency"]) / (df["Frequency"] + 1)

df["freq_norm"] = df["Frequency"] / (df["Frequency"].max() if df["Frequency"].max() != 0 else 1)
df["mon_norm"] = df["Monetary"] / (df["Monetary"].max() if df["Monetary"].max() != 0 else 1)

df["purchase_session_ratio"] = df["Frequency"] / (df["Length"] + 1)

df["recency_weight"] = np.exp(-df["Recency"] / (df["Recency"].max() if df["Recency"].max() else 1))

df["monetary_trend"] = df["nov_frequency"] - df["Frequency"]

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# --------------------------
# Save enhanced dataset
# --------------------------
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT)

print("✅ Behavioral features added successfully!")
print("Saved to:", OUTPUT)
print("New columns added (tail):")
print(df.columns[-8:].tolist())

# ======================================================
#                     GRAPH SECTION (ONLY 3 MANDATORY)
# ======================================================

print("\n📊 Generating mandatory graphs...")

# 1️⃣ Recency Distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Recency"], bins=30)
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Activity")
plt.ylabel("Customer Count")
plt.grid(False)
plt.show()

# 2️⃣ Frequency Distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Frequency"], bins=30)
plt.title("Frequency Distribution")
plt.xlabel("Number of Purchases / Sessions")
plt.ylabel("Customer Count")
plt.grid(False)
plt.show()

# 3️⃣ Monetary Distribution
plt.figure(figsize=(7, 4))
plt.hist(df["Monetary"], bins=30)
plt.title("Monetary Value Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Customer Count")
plt.grid(False)
plt.show()

print("\n✅ All 3 mandatory graphs generated successfully!")
