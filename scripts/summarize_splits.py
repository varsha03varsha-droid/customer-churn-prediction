import os
import pandas as pd

# Paths (change only if your folders differ)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESSED = os.path.join(PROJECT_ROOT, "processed")

OCT_SPLIT_DIR = os.path.join(PROJECT_ROOT, "oct_dataset_split")
NOV_SPLIT_DIR = os.path.join(PROJECT_ROOT, "nov_dataset_split")
TELCO_SPLIT_DIR = os.path.join(PROJECT_ROOT, "telco_dataset_split")

MERGED_LRFM_PATH = os.path.join(PROCESSED, "merged_LRFM_labeled.csv")  # has churn_label
TELCO_CLEAN_PATH = os.path.join(PROCESSED, "telco_clean.csv")         # has 'Churn' Yes/No

out_lines = []

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None

def summarize_split_folder(folder_path, label_map_from_merged=True, merged_df=None, id_col_names=None):
    """
    Looks for train.csv and test.csv in folder_path and returns summary dict.
    If the split files don't contain churn_label, tries to map by user_id using merged_df.
    id_col_names: list of possible id column names to look for in the split file (e.g. ['user_id', 'customerID'])
    """
    id_col_names = id_col_names or ['user_id', 'customerID', 'userId']

    result = {}
    for part in ["train.csv", "test.csv"]:
        p = os.path.join(folder_path, part)
        if not os.path.exists(p):
            result[part] = {"exists": False}
            continue

        df = safe_read_csv(p)
        if df is None:
            result[part] = {"exists": True, "error": "failed to read"}
            continue

        total_rows = len(df)
        result_dict = {"exists": True, "rows": total_rows, "counts": None}

        # If churn_label exists directly in file (e.g. merged LRFM split), use it
        if "churn_label" in df.columns:
            counts = df["churn_label"].value_counts().to_dict()
            result_dict["counts"] = counts
            result[part] = result_dict
            continue

        # If telco-style 'Churn' exists:
        if "Churn" in df.columns:
            counts = df["Churn"].value_counts().to_dict()
            result_dict["counts"] = counts
            result[part] = result_dict
            continue

        # Otherwise, try to map using merged_df by id column
        mapped = None
        if merged_df is not None:
            found_id = None
            for c in id_col_names:
                if c in df.columns:
                    found_id = c
                    break

            if found_id:
                # merged_df index might be user id index or may have column 'user_id'
                merged_lookup = merged_df.copy()
                # if merged index is not user_id column, try to reset index to a column called user_id
                if "user_id" in merged_lookup.columns:
                    merged_lookup = merged_lookup.set_index("user_id")
                else:
                    # if index is numeric or other string, ensure it is string to match
                    merged_lookup.index = merged_lookup.index.astype(str)

                # attempt mapping
                df_ids = df[found_id].astype(str)
                # select available ids which also exist in merged_lookup
                common = merged_lookup.loc[merged_lookup.index.intersection(df_ids)]
                if not common.empty:
                    # count churn_label in matched set
                    counts = common["churn_label"].value_counts().to_dict()
                    result_dict["counts"] = counts
                else:
                    result_dict["counts"] = {"mapped": 0, "note": "uploaded ids not found in merged dataset"}
            else:
                result_dict["counts"] = {"note": "no churn_label and no id column to map"}
        else:
            result_dict["counts"] = {"note": "merged dataset not available for mapping"}

        result[part] = result_dict

    return result

def print_summary(title, summary):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    for k, v in summary.items():
        print(f"\nFile: {k}")
        if not v.get("exists"):
            print("  -> Not found.")
            continue
        if v.get("error"):
            print("  -> Error:", v["error"])
            continue
        print("  Rows:", v.get("rows"))
        print("  Counts:")
        counts = v.get("counts")
        if counts is None:
            print("    No counts available")
        else:
            # pretty print counts dict
            for label, cnt in counts.items():
                print(f"    {label}: {cnt}")

def main():
    print("📊 Summary script — showing train/test sizes and churn counts\n")

    # load merged labeled if exists
    merged_df = None
    if os.path.exists(MERGED_LRFM_PATH):
        try:
            merged_df = pd.read_csv(MERGED_LRFM_PATH, index_col=0)
            print("✅ Loaded merged LRFM labeled dataset:", MERGED_LRFM_PATH, "rows:", len(merged_df))
            # ensure churn_label present
            if "churn_label" not in merged_df.columns:
                print("⚠️ merged file exists but has no 'churn_label' column.")
                merged_df = None
        except Exception as e:
            print("⚠️ Could not load merged LRFM file:", e)
            merged_df = None
    else:
        print("ℹ️ merged_LRFM_labeled.csv not found at processed/. Mapping by user_id will be skipped.")

    # 1) NOVEMBER
    if os.path.exists(NOV_SPLIT_DIR):
        nov_summary = summarize_split_folder(NOV_SPLIT_DIR, merged_df=merged_df, id_col_names=['user_id', 'userId'])
        print_summary("November dataset splits (nov_dataset_split)", nov_summary)
    else:
        print("\nNovember split folder not found:", NOV_SPLIT_DIR)

    # 2) OCTOBER
    if os.path.exists(OCT_SPLIT_DIR):
        oct_summary = summarize_split_folder(OCT_SPLIT_DIR, merged_df=merged_df, id_col_names=['user_id', 'userId'])
        print_summary("October dataset splits (oct_dataset_split)", oct_summary)
    else:
        print("\nOctober split folder not found:", OCT_SPLIT_DIR)

    # 3) TELCO
    if os.path.exists(TELCO_SPLIT_DIR):
        telco_summary = summarize_split_folder(TELCO_SPLIT_DIR, merged_df=None, id_col_names=['customerID'])
        print_summary("Telco dataset splits (telco_dataset_split)", telco_summary)
    else:
        print("\nTelco split folder not found:", TELCO_SPLIT_DIR)

    # 4) Overall merged churn counts (if merged exists)
    if merged_df is not None:
        print("\n" + "="*60)
        print("OVERALL (merged_LRFM_labeled.csv) churn distribution")
        print("="*60)
        print(merged_df["churn_label"].value_counts().to_string())
    else:
        print("\nNo merged_LRFM_labeled.csv available to show overall churn distribution.")

    print("\n✅ Done.\n")

if __name__ == "__main__":
    main()
