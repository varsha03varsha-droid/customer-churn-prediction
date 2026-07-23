import pandas as pd

def compute_LRFM(df, user_col='user_id', date_col='event_time', amount_col='price'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])  # remove rows with invalid dates
    max_date = df[date_col].max()
    grouped = df.groupby(user_col)

    # Length: Days between first and last purchase
    length = (grouped[date_col].max() - grouped[date_col].min()).dt.days

    # Recency: Days since last purchase relative to max_date in dataset
    recency = (max_date - grouped[date_col].max()).dt.days

    # Frequency: Number of purchases or events
    frequency = grouped.size()

    # Monetary: Total amount spent by the user
    monetary = grouped[amount_col].sum()

    LRFM = pd.DataFrame({
        'Length': length,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    })

    return LRFM


if __name__ == '__main__':
    oct_data = pd.read_csv(r"D:\customer_project\processed\oct_clean.csv")
    nov_data = pd.read_csv(r"D:\customer_project\processed\nov_clean.csv")

    print("⏳ Computing LRFM for October...")
    oct_LRFM = compute_LRFM(oct_data)

    print("⏳ Computing LRFM for November...")
    nov_LRFM = compute_LRFM(nov_data)

    oct_LRFM.to_csv(r"D:\customer_project\processed\oct_LRFM.csv")
    nov_LRFM.to_csv(r"D:\customer_project\processed\nov_LRFM.csv")

    print("\n✅ LRFM features computed and saved successfully.")
