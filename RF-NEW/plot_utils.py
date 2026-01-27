import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def save_class_distribution_plot(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    filename: str = "class_distribution.png",
    title: Optional[str] = None,
) -> str:
    """
    Saves a bar chart showing the class distribution of a target column.

    Args:
        df: Input dataframe.
        target_col: Name of the target column (e.g., 'PR_lifetime').
        output_dir: Directory where the plot will be saved.
        filename: Output filename for the plot.
        title: Custom title (optional).

    Returns:
        The full path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_counts = df[target_col].value_counts(dropna=False).sort_index()

    plt.figure(figsize=(8, 5))
    class_counts.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title(title or f"{target_col} Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return plot_path
