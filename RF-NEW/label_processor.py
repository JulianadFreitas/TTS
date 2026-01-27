import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from typing import Tuple
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def clean_and_parse_labels(label_str):
    """
    Standardizes label format by handling nested lists and strings.
    Converts to lowercase and strips whitespace to avoid duplicates.
    """
    if pd.isna(label_str) or label_str == "" or label_str == "[]":
        return []
    try:
        # Evaluates string to Python object
        data = ast.literal_eval(label_str)
        # Handles the specific nested format: ["['bug', 'ui']"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            if data[0].startswith("["):
                data = ast.literal_eval(data[0])
        return [str(label).lower().strip() for label in data]
    except (ValueError, SyntaxError):
        return []


def add_label_features_and_report(
    df: pd.DataFrame,
    output_dir: str,
    label_col: str = "labels",
    min_occurrences: int = 10,
    report_filename: str = "label_frequency_report.csv",
    feature_prefix: str = "CONT_label_",
) -> pd.DataFrame:
    """
    Adds one-hot encoded label features for frequent labels and writes a frequency report.
    Returns a new DataFrame with added features (does not drop the raw label column).
    """
    if label_col not in df.columns:
        return df

    df = df.copy()

    parsed_labels = df[label_col].apply(clean_and_parse_labels)
    all_labels_flat = [lbl for sublist in parsed_labels for lbl in sublist]
    label_counts = pd.Series(all_labels_flat).value_counts()

    report_df = label_counts.reset_index()
    report_df.columns = ["Label", "Frequency"]
    report_df["Kept_as_Feature"] = report_df["Frequency"] >= min_occurrences

    report_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_path, index=False)

    common_labels = report_df.loc[report_df["Kept_as_Feature"], "Label"].tolist()
    filtered_labels = parsed_labels.apply(lambda x: [lbl for lbl in x if lbl in common_labels])

    mlb = MultiLabelBinarizer(classes=sorted(common_labels))
    mat = mlb.fit_transform(filtered_labels)

    label_cols = [f"{feature_prefix}{c}" for c in mlb.classes_]
    label_features_df = pd.DataFrame(mat, columns=label_cols, index=df.index)

    return pd.concat([df, label_features_df], axis=1)


def run_experiment(input_path, min_occurrences=10):
    print(f"--- Starting Label Analysis (Threshold: {min_occurrences}) ---")
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Step 1: Parse labels into clean lists
    df['parsed_labels'] = df['labels'].apply(clean_and_parse_labels)
    
    # Step 2: Frequency Analysis
    all_labels_flat = [lbl for sublist in df['parsed_labels'] for lbl in sublist]
    label_counts = pd.Series(all_labels_flat).value_counts()
    
    # Step 3: Identify labels that meet the research criteria 
    common_labels = label_counts[label_counts >= min_occurrences].index.tolist()
    print(f"Total unique labels found: {len(label_counts)}")
    print(f"Labels passing the threshold: {len(common_labels)}")

    # Step 4: Generate Document - Frequency Report
    report_df = label_counts.reset_index()
    report_df.columns = ['Label', 'Frequency']
    report_df['Kept_as_Feature'] = report_df['Frequency'] >= min_occurrences
    report_df.to_csv('label_frequency_report.csv', index=False)
    print("✅ Document 'label_frequency_report.csv' generated.")

    # Step 5: Binarization (One-Hot Encoding)
    # Only includes labels that passed the threshold
    df['filtered_labels'] = df['parsed_labels'].apply(
        lambda x: [lbl for lbl in x if lbl in common_labels]
    )
    
    mlb = MultiLabelBinarizer(classes=sorted(common_labels))
    binarized_matrix = mlb.fit_transform(df['filtered_labels'])
    
    # Create columns with 'CONT_' prefix for context metadata 
    label_cols = [f"CONT_label_{c}" for c in mlb.classes_]
    label_features_df = pd.DataFrame(binarized_matrix, columns=label_cols, index=df.index)
    
    # Step 6: Save final feature set
    # Keeping 'repo' for context as per Table I [cite: 151]
    output_df = pd.concat([df[['CONT_repo']], label_features_df], axis=1)
    output_df.to_csv('final_label_features.csv', index=False)
    print("✅ Document 'final_label_features.csv' generated.")

    intermediate_cols = ['labels', 'parsed_labels', 'filtered_labels']

    final_dataset = pd.concat([
        df.drop(columns=intermediate_cols, errors='ignore'), 
        label_features_df
    ], axis=1)
    
    final_dataset.to_csv('final_consolidated_dataset.csv', index=False)
    print(f"✅ Success! Consolidated dataset saved with {final_dataset.shape[1]} columns.")

    # Step 7: Visual Documentation
    plt.figure(figsize=(10, 6))
    label_counts.head(15).plot(kind='bar', color='teal')
    plt.axhline(y=min_occurrences, color='red', linestyle='--', label=f'Threshold ({min_occurrences})')
    plt.title('Top 15 Labels Frequency')
    plt.ylabel('Occurrences')
    plt.legend()
    plt.tight_layout()
    plt.savefig('label_distribution_chart.png')
    print("✅ Chart 'label_distribution_chart.png' saved.")

if __name__ == "__main__":
    # Ensure you have a 'dataset.csv' in the same folder or update the path below
    # The CSV must have a 'labels' column and a 'repo' column
    input_file = "../data/PREPROCESSED/Final_TtST_Dataset.csv" 
    
    if os.path.exists(input_file):
        run_experiment(input_file)
    else:
        print(f"❌ Error: File '{input_file}' not found. Please provide your data.")