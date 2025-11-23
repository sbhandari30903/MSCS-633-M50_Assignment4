"""
train_autoencoder.py

Fraud detection using an AutoEncoder from PyOD on the Kaggle
"Fraud Detection" dataset (whenamancodes/fraud-detection),
which provides the anonymized credit card transactions file
`creditcard.csv`.

Data is loaded directly from Kaggle using kagglehub, so you do
not need to store creditcard.csv in your repository.

Author: Your Name
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from pyod.models.auto_encoder import AutoEncoder


# -------------------------------------------------------------------
# Data loading with kagglehub
# -------------------------------------------------------------------
def load_data_from_kaggle() -> pd.DataFrame:
    """
    Load the anonymized credit card transactions dataset from Kaggle
    using kagglehub.

    Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
    File: creditcard.csv

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing all transactions with features and 'Class' label.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as e:
        raise ImportError(
            "kagglehub is not installed. "
            "Install it with: pip install kagglehub[pandas-datasets]"
        ) from e

    dataset_id = "whenamancodes/fraud-detection"
    file_path = "creditcard.csv"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset_id,
        file_path,
    )

    return df


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame):
    """
    Split the dataset into features and labels, then perform
    train/validation/test split and feature scaling.

    We treat this as an outlier-detection problem and primarily
    train on non-fraud (Class = 0) transactions.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe loaded from Kaggle.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices for train, validation, test.
    y_train, y_val, y_test : np.ndarray
        Corresponding labels (0/1).
    scaler : StandardScaler
        Fitted scaler object.
    """
    # Separate features and labels
    X = df.drop(columns=["Class"])
    y = df["Class"].values

    # First split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    # Now: 60% train, 20% val, 20% test

    # Train a scaler only on training data (good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )


# -------------------------------------------------------------------
# Model training
# -------------------------------------------------------------------
def train_autoencoder(X_train: np.ndarray) -> AutoEncoder:
    """
    Train the PyOD AutoEncoder model.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training features (mostly normal transactions).

    Returns
    -------
    clf : AutoEncoder
        Fitted AutoEncoder model.
    """
    clf = AutoEncoder(
        hidden_neuron_list=[32, 16, 16, 32],  # encoder/decoder layers
        epoch_num=30,                         # number of epochs
        batch_size=256,
        dropout_rate=0.2,
        contamination=0.002,                 # approx. fraud rate (~0.17%)
        lr=1e-3,
        verbose=1,
        random_state=42,
        batch_norm=True,
    )

    clf.fit(X_train)
    return clf


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_model(
    clf: AutoEncoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Evaluate the AutoEncoder model using validation data to select
    a decision threshold, then test on a hold-out test set.

    Parameters
    ----------
    clf : AutoEncoder
        Trained AutoEncoder model.
    X_val, X_test : np.ndarray
        Feature matrices for validation and test sets.
    y_val, y_test : np.ndarray
        Labels (0/1) for validation and test sets.
    """
    # PyOD: decision_function returns anomaly scores (higher = more anomalous)
    val_scores = clf.decision_function(X_val)
    test_scores = clf.decision_function(X_test)

    # Default threshold learned by PyOD
    default_threshold = clf.threshold_

    # Custom threshold using quantile on validation scores
    custom_threshold = np.quantile(val_scores, 0.995)

    print(f"\nModel default threshold: {default_threshold:.6f}")
    print(f"Custom threshold (99.5% quantile on val): {custom_threshold:.6f}")

    # Use custom threshold to binarize anomaly scores
    val_pred = (val_scores > custom_threshold).astype(int)
    test_pred = (test_scores > custom_threshold).astype(int)

    print("\n=== Validation set evaluation ===")
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, val_pred))

    print("\n=== Test set evaluation ===")
    print(classification_report(y_test, test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, test_pred))

    # ROC-AUC on test set using scores directly
    auc_test = roc_auc_score(y_test, test_scores)
    print(f"\nROC-AUC (test, using anomaly scores): {auc_test:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AutoEncoder (AUC = {auc_test:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AutoEncoder Fraud Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("autoencoder_roc_curve.png")
    plt.close()

    # Plot anomaly score distribution
    plt.figure()
    plt.hist(test_scores[y_test == 0], bins=50, alpha=0.7, label="Normal")
    plt.hist(test_scores[y_test == 1], bins=50, alpha=0.7, label="Fraud")
    plt.axvline(custom_threshold, color="k", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.title("Decision score distribution on test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("autoencoder_score_distribution.png")
    plt.close()

    print("\nSaved plots:")
    print(" - autoencoder_roc_curve.png")
    print(" - autoencoder_score_distribution.png")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main():
    # 1. Load data from Kaggle via kagglehub
    print("Loading dataset from Kaggle (whenamancodes/fraud-detection)...")
    df = load_data_from_kaggle()

    print("Dataset shape:", df.shape)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in dataset but did not find it.")

    print("Class distribution:")
    print(df["Class"].value_counts())

    # 2. Preprocess data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = preprocess_data(df)

    # Train AutoEncoder only on normal (non-fraud) transactions
    X_train_normal = X_train[y_train == 0]

    print("Training samples (normal only):", X_train_normal.shape[0])

    # 3. Train model
    clf = train_autoencoder(X_train_normal)

    # 4. Evaluate model
    evaluate_model(clf, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()

