import pandas as pd
import glob
import os

def cleanup_preprocessed_folder(folder_path, column_to_delete):
    """
    Removes a specific column from all consolidated and binned 
    files in the PREPROCESSED directory.
    """
    abs_path = os.path.abspath(folder_path)
    print(f"Checking preprocessed tables in: {abs_path}")
    
    # Matches final_dataset.csv and all group files
    files = glob.glob(os.path.join(abs_path, "*.csv"))
    
    if not files:
        print("❌ No preprocessed CSV files found.")
        return

    for file in files:
        try:
            df = pd.read_csv(file)
            
            if column_to_delete in df.columns:
                df.drop(columns=[column_to_delete], inplace=True)
                df.to_csv(file, index=False)
                print(f"✨ Cleaned: {os.path.basename(file)}")
            else:
                print(f"idx: '{column_to_delete}' not in {os.path.basename(file)}")
        
        except Exception as e:
            print(f"⚠️ Error in {os.path.basename(file)}: {e}")

if __name__ == "__main__":
    # Path based on your VS Code structure
    PREPROCESSED_PATH = "../data/PREPROCESSED/"
    COLUMN_NAME = "dependency_changes"
    
    cleanup_preprocessed_folder(PREPROCESSED_PATH, COLUMN_NAME)
    print("\nPreprocessed data is now lean and clean.")