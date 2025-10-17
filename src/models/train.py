"""Model training functions."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from typing import Any, Tuple


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any) -> Any:
    """Train a model.

    Args:
        X_train: Training features
        y_train: Training target
        model: Model instance

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Evaluate model using cross-validation.

    Args:
        model: Model instance
        X: Features
        y: Target
        cv: Number of cross-validation folds

    Returns:
        Mean cross-validation score
    """
    scores = cross_val_score(model, X, y, cv=cv)
    mean_score = scores.mean()
    print(f"CV Score: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
    return mean_score


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Make predictions.

    Args:
        model: Trained model
        X: Features

    Returns:
        Predictions
    """
    return model.predict(X)
