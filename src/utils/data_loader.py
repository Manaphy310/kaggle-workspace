"""Data loading utilities."""

import pandas as pd
from pathlib import Path


def load_train_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load training data.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Training dataframe
    """
    data_path = Path(data_dir) / "train.csv"
    return pd.read_csv(data_path)


def load_test_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load test data.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Test dataframe
    """
    data_path = Path(data_dir) / "test.csv"
    return pd.read_csv(data_path)


def save_submission(predictions: pd.DataFrame, filename: str = "submission.csv") -> None:
    """Save predictions in submission format.

    Args:
        predictions: Dataframe with predictions
        filename: Output filename
    """
    output_path = Path("submissions") / filename
    predictions.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
