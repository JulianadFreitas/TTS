import pandas as pd
import numpy as np
import glob
import os
import re
import ast
from label_processor import add_label_features_and_report, clean_and_parse_labels
from sklearn.preprocessing import MultiLabelBinarizer
from plot_utils import save_class_distribution_plot

class DataPreparator:
    def __init__(self, raw_path, output_path):
        self.raw_path = os.path.abspath(raw_path)
        self.output_path = output_path
        self.leakage_cutoff = '2024-07-18'


    def run_pipeline(self):
        # Step 1: Consolidating files
        print(f"Step 1: Consolidating files from {self.raw_path}...")
        files = glob.glob(os.path.join(self.raw_path, "*.csv"))
        if not files:
            print("❌ Error: No CSV files found."); return

        all_data = []
        for i, file in enumerate(files, 1):
            temp_df = pd.read_csv(file)
            temp_df['repo'] = str(i)
            all_data.append(temp_df)
        df = pd.concat(all_data, ignore_index=True)

        df = df.rename(columns={'issue_comments_text': 'PR_comments_text'})
        df = df.rename(columns={'issue_text': 'PR_text'})
        
        # Step 2: Leakage Prevention
        text_and_leakage_cols_to_drop = [
            'review_duration',
            'review_comments',
            'comment_authors',
            'reviews_text',
            'PR_comments_text'
        ]

        df.drop(columns=[c for c in text_and_leakage_cols_to_drop if c in df.columns], inplace=True, errors='ignore')


        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["merged_at"] = pd.to_datetime(df["merged_at"], errors="coerce")

        df = df[df['created_at'] > self.leakage_cutoff].copy()

        # Step 3: Compute Time to Solve (TtST) and Bins
        df['PR_Lifetime(hours)'] = (df['merged_at'] - df['created_at']).dt.total_seconds() / 3600
        conditions = [
            (df['PR_Lifetime(hours)'] < 1),
            (df['PR_Lifetime(hours)'] < 24),
            (df['PR_Lifetime(hours)'] < 168),
            (df['PR_Lifetime(hours)'] < 5040),
            (df['PR_Lifetime(hours)'] >= 5040)
        ]
        df['PR_lifetime'] = np.select(conditions, [1, 2, 3, 4, 5], default=5)

        plot_path = save_class_distribution_plot(
            df=df,
            target_col="PR_lifetime",
            output_dir=os.path.dirname(self.output_path),
            filename="pr_lifetime_class_distribution.png",
            title="PR_lifetime Class Distribution (1-5)",
        )
        print(f"✅ Class distribution chart saved: {plot_path}")
        print(df["PR_lifetime"].value_counts().sort_index().to_string())

        df["label_count"] = df["labels"].apply(lambda x: len(clean_and_parse_labels(x)))

        # Add one-hot label features + save frequency report
        df = add_label_features_and_report(
            df=df,
            output_dir=os.path.dirname(self.output_path),
            label_col="labels",
            min_occurrences=10,
            report_filename="label_frequency_report.csv",
            feature_prefix="CONT_label_",
        )

        # Step 4: Compute Historical & Aditional Metrics
        df = df.sort_values(by=['repo', 'created_at']).reset_index(drop=True)
        
        df['PR_text_wordiness'] = df['PR_text'].apply(lambda x: 0 if pd.isnull(x) else len(x.split()))
        #determien teh day ofthe week teh PR was created
        df['day_of_week'] = df['created_at'].dt.dayofweek
        #0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday, 4 = Friday, 5 = Saturday, 6 = Sunday

        #for df['weekday'] set to 1 if day_of_week is between 0 and 4 else 0
        df['weekday'] = df['day_of_week'].apply(lambda x: 1 if x >= 0 and x <= 4 else 0)

        df['author_PRs_opened'] = df.groupby(['repo', 'author']).cumcount()
        df['PRs_opened_in_last_2_weeks'] = df.apply(
            lambda r: df[(df['created_at'] >= r['created_at'] - pd.Timedelta(days=14)) & 
                         (df['created_at'] < r['created_at']) & (df['repo'] == r['repo'])].shape[0], axis=1)
        df['PRs_closed_in_last_2_weeks'] = df.apply(
            lambda r: df[(df['merged_at'] >= r['merged_at'] - pd.Timedelta(days=14)) & 
                         (df['merged_at'] < r['merged_at']) & (df['repo'] == r['repo'])].shape[0], axis=1)
        df['open_PRs_at_open_date'] = df.apply(
            lambda r: df[(df['created_at'] < r['created_at']) & 
                         (df['merged_at'] > r['created_at']) & (df['repo'] == r['repo'])].shape[0], axis=1)

        # Step 5: Feature Grouping and Prefixing
        print("Step 5: Grouping features by category...")
        
        # Defining Groups
        code_content_cols = [
            "number_of_commits",
            "lines_of_code_changed",
            "number_of_files_changed",
            "number_of_revisions",
            "number_of_milestones"
            ]

        activity_collaboration_cols = [
            "number_of_comments",
            "number_of_review_comments",
            "number_of_reviewers",
            "number_of_approvals",
            "number_of_changes_requested",
            "number_of_reviews_requested",
            "number_of_linked_issues",
            "number_of_assignees",
            "time_to_first_response",
            "time_since_last_commit",
            ]
        context_metadata_cols = [
            "repo",
            "day_of_week",
            "weekday",
            "label_count",
            "PR_text_wordiness"
            ]
        org_author_history_cols = [
            "author_PRs_opened",
            "open_PRs_at_open_date",
            "PRs_closed_in_last_2_weeks",
            "PRs_opened_in_last_2_weeks"
            ]
        prefix_map = {
            "CODE": code_content_cols,
            "ACT": activity_collaboration_cols,
            "CONT": context_metadata_cols,
            "ORG": org_author_history_cols
            }
        
        rename_mapping = {}
        for prefix, columns in prefix_map.items():
            for col in columns:
                if col in df.columns:
                    rename_mapping[col] = f"{prefix}_{col}"

        df = df.rename(columns=rename_mapping)

        # Step 6: Final Cleanup - Dropping raw text and unused columns
        print("Step 6: Dropping raw text and unused columns...")
        cols_to_drop = [
            'created_at',
            'merged_at',
            'author',
            'PR_Lifetime(hours)',
            'labels',
            'PR_text'
        ]
     
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        # Save the dataset
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"✅ Success! Dataset organized with prefixes at: {self.output_path}")

if __name__ == "__main__":
    prep = DataPreparator(raw_path="../data/RAW/", output_path="../data/PREPROCESSED/Final_TtST_Dataset.csv")
    prep.run_pipeline()