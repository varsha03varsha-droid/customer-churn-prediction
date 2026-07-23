import pandas as pd

# Load your sampled datasets
oct_data = pd.read_csv(r"D:\customer_project\base_data\2019-Oct-Sampled.csv")
nov_data = pd.read_csv(r"D:\customer_project\base_data\2019-Nov-Sampled.csv")

# Take smaller random samples
oct_small = oct_data.sample(n=100000, random_state=42)
nov_small = nov_data.sample(n=100000, random_state=42)

# Save them as new files
oct_small.to_csv(r"D:\customer_project\base_data\2019-Oct-Small.csv", index=False)
nov_small.to_csv(r"D:\customer_project\base_data\2019-Nov-Small.csv", index=False)

print("✅ Created smaller balanced files successfully!")
