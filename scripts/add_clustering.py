import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def add_cluster_labels(lrfm_df, n_clusters=4):
    """
    Adds cluster labels to LRFM data using K-Means clustering.
    Automatically scales data before clustering.
    """
    # Step 1: Scale LRFM data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(lrfm_df[['Length', 'Recency', 'Frequency', 'Monetary']])

    # Step 2: Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # Step 3: Add cluster labels
    lrfm_df['cluster'] = clusters

    print(f"✅ Clustering complete — {n_clusters} clusters created.")
    print(lrfm_df['cluster'].value_counts())

    return lrfm_df, kmeans


if __name__ == '__main__':
    # Load LRFM data (October)
    oct_LRFM = pd.read_csv(r"D:\customer_project\processed\oct_LRFM.csv", index_col=0)

    print("⏳ Running K-Means clustering on October LRFM data...")
    oct_LRFM_clustered, kmeans_model = add_cluster_labels(oct_LRFM, n_clusters=4)

    # Save clustered output
    output_path = r"D:\customer_project\processed\oct_LRFM_clustered.csv"
    oct_LRFM_clustered.to_csv(output_path)

    print(f"\n💾 Clustered LRFM data saved successfully to: {output_path}")
