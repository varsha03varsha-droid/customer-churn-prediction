import pandas as pd

def label_churn(oct_LRFM_path, nov_clean_path, oct_clustered_path=None):
    """
    Labels customers as Total Churn, Partial Churn, or Non-Churn 
    by comparing October LRFM (with optional clusters) against November activity.
    """
    # Load October LRFM and November raw (cleaned) data
    oct_df = pd.read_csv(oct_LRFM_path, index_col=0)
    nov_df = pd.read_csv(nov_clean_path)

    # Add cluster column from Oct if available
    if oct_clustered_path:
        oct_clustered = pd.read_csv(oct_clustered_path, index_col=0)
        oct_df['cluster'] = oct_clustered['cluster']

    # Compute how many events per user in November
    nov_activity = nov_df.groupby('user_id').size().rename('nov_frequency')
    merged = oct_df.join(nov_activity, how='left').fillna(0)

    # Define churn label logic
    def churn_label(row):
        if row['Frequency'] > 0 and row['nov_frequency'] == 0:
            return 'Total Churn'
        elif row['nov_frequency'] < 0.5 * row['Frequency']:
            return 'Partial Churn'
        else:
            return 'Non-Churn'

    # Apply churn labels
    merged['churn_label'] = merged.apply(churn_label, axis=1)

    print("✅ Churn labeling complete:")
    print(merged['churn_label'].value_counts())

    return merged


if __name__ == '__main__':
    merged_with_labels = label_churn(
        r"D:\customer_project\processed\oct_LRFM.csv",
        r"D:\customer_project\processed\nov_clean.csv",
        oct_clustered_path=r"D:\customer_project\processed\oct_LRFM_clustered.csv"  # Optional, include cluster info
    )

    output_path = r"D:\customer_project\processed\merged_LRFM_labeled.csv"
    merged_with_labels.to_csv(output_path)

    print(f"\n💾 Labeled churn data saved to: {output_path}")
