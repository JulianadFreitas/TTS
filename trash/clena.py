# import pandas as pd
# import glob
# import os

# def delete_column_from_raw(raw_path, column_to_delete):
#     """
#     Iterates through all CSV files in the RAW folder and removes
#     a specific column if it exists.
#     """
#     # Using absolute path to avoid the directory issues we saw earlier
#     abs_path = os.path.abspath(raw_path)
#     print(f"Searching for files in: {abs_path}")
    
#     files = glob.glob(os.path.join(abs_path, "*.csv"))
    
#     if not files:
#         print("❌ No CSV files found. Please check your path.")
#         return

#     print(f"Found {len(files)} files. Starting cleanup...")

#     for file in files:
#         try:
#             # Read the CSV
#             df = pd.read_csv(file)
            
#             if column_to_delete in df.columns:
#                 # Drop the column
#                 df.drop(columns=[column_to_delete], inplace=True)
#                 # Save the file back (overwriting)
#                 df.to_csv(file, index=False)
#                 print(f"✅ Removed '{column_to_delete}' from {os.path.basename(file)}")
#             else:
#                 print(f"ℹ️ Column '{column_to_delete}' not found in {os.path.basename(file)}, skipping.")
        
#         except Exception as e:
#             print(f"⚠️ Error processing {os.path.basename(file)}: {e}")

# if __name__ == "__main__":
#     # Adjust path relative to where you run the script (likely inside RF-NEW)
#     RAW_DATA_PATH = "../data/RAW/"
#     COLUMN_NAME = "dependency_changes"
    
#     delete_column_from_raw(RAW_DATA_PATH, COLUMN_NAME)
#     print("\nCleanup process finished.")

print("merged_at NaN:", df["merged_at"].isna().sum())
print("created_at NaN:", df["created_at"].isna().sum())

df["PR_Lifetime(hours)"] = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 3600
print("lifetime NaN:", df["PR_Lifetime(hours)"].isna().sum())
print("lifetime negative:", (df["PR_Lifetime(hours)"] < 0).sum())
