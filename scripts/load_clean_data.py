import pandas as pd

def load_data():
    # Define dataset paths (all inside base_data folder)
    oct_path = r"D:\customer_project\base_data\2019-Oct-Small.csv"
    nov_path = r"D:\customer_project\base_data\2019-Nov-Small.csv"
    telco_path = r"D:\customer_project\base_data\WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Load all datasets
    oct_data = pd.read_csv(oct_path)
    nov_data = pd.read_csv(nov_path)
    telco_data = pd.read_csv(telco_path)

    # Display shapes for confirmation
    print("✅ Data Loaded Successfully:")
    print("October dataset shape:", oct_data.shape)
    print("November dataset shape:", nov_data.shape)
    print("Telco dataset shape:", telco_data.shape)

    return oct_data, nov_data, telco_data


def clean_data(df):
    """Simple cleaning step: remove missing rows."""
    df_cleaned = df.dropna()
    return df_cleaned


if __name__ == '__main__':
    # Load datasets
    oct_data, nov_data, telco_data = load_data()

    # Clean each dataset
    oct_data = clean_data(oct_data)
    nov_data = clean_data(nov_data)
    telco_data = clean_data(telco_data)

    # Save cleaned datasets to 'processed' folder
    oct_data.to_csv(r"D:\customer_project\processed\oct_clean.csv", index=False)
    nov_data.to_csv(r"D:\customer_project\processed\nov_clean.csv", index=False)
    telco_data.to_csv(r"D:\customer_project\processed\telco_clean.csv", index=False)

    print("\n💾 All cleaned files saved successfully to 'processed' folder.")
