import pandas as pd
import numpy as np

# --- File paths ---
oct_path = r"D:\customer_project\2019-Oct.csv"
nov_path = r"D:\customer_project\2019-Nov.csv"

# --- Output paths ---
oct_output = r"D:\customer_project\base_data\2019-Oct-Sampled.csv"
nov_output = r"D:\customer_project\base_data\2019-Nov-Sampled.csv"

# --- Sampling fraction (smaller fraction = smaller file, adjust as needed) ---
SAMPLE_FRAC = 0.01  # means ~1% of the entire dataset, roughly 500k–700k rows depending on size

# --- Function to read in chunks and sample evenly ---
def sample_csv(file_path, output_path, frac):
    print(f"\nProcessing {file_path} ...")
    chunksize = 1_000_000  # read 1 million rows at a time
    samples = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize, parse_dates=['event_time']):
        # Randomly sample a small fraction of each chunk
        samples.append(chunk.sample(frac=frac, random_state=42))
    
    df_sample = pd.concat(samples)
    
    # Shuffle to mix data across the month
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the smaller dataset
    df_sample.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df_sample)} rows to {output_path}")
    print(f"Date Range: {df_sample['event_time'].min()} → {df_sample['event_time'].max()}")
    return df_sample

# --- Run for both months ---
df_oct = sample_csv(oct_path, oct_output, SAMPLE_FRAC)
df_nov = sample_csv(nov_path, nov_output, SAMPLE_FRAC)

print("\n✅ Sampling completed successfully!")
