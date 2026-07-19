"""
Project 2: Data Classification Using AI
DecodeLabs AI Industrial Training — Batch 2026

Goal: Build a basic classification model using a small dataset (the Iris benchmark).
Pipeline follows the brief's IPO Framework:
    INPUT   -> Iris domain, feature scaling
    PROCESS -> Train-test split, K-Nearest Neighbors algorithm
    OUTPUT  -> Confusion matrix, F1 score, accuracy validation

Author: Jerocin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # save plots next to this script
sns.set_theme(style="whitegrid")


# =========================================================
# PHASE 1 — INPUT: Load and understand the dataset
# =========================================================
def load_dataset():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    class_names = iris.target_names

    print("=" * 60)
    print("RAW MATERIAL: THE IRIS BENCHMARK")
    print("=" * 60)
    print(f"Samples     : {len(X)} (balanced)")
    print(f"Classes     : {len(class_names)} -> {list(class_names)}")
    print(f"Dimensions  : {X.shape[1]} -> {list(X.columns)}")
    print(X.describe().round(2))
    print()
    return X, y, class_names


# =========================================================
# PHASE 2 — STRUCTURAL INTEGRITY: The Train/Test Split
# =========================================================
def split_data(X, y):
    # Shuffle before splitting to remove order bias (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,          # 80/20 split as per the IPO diagram
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    print("=" * 60)
    print("STRUCTURAL INTEGRITY: THE SPLIT")
    print("=" * 60)
    print(f"Training set : {len(X_train)} samples")
    print(f"Test set     : {len(X_test)} samples")
    print()
    return X_train, X_test, y_train, y_test


# =========================================================
# PHASE 3 — THE GATEKEEPER RULE: Feature Scaling
# =========================================================
def scale_features(X_train, X_test):
    scaler = StandardScaler()  # mean = 0, variance = 1
    X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
    X_test_scaled = scaler.transform(X_test)          # transform test data using train stats
    print("=" * 60)
    print("THE GATEKEEPER RULE: SCALING")
    print("=" * 60)
    print("StandardScaler applied. Mean = 0, Variance = 1 on training features.")
    print()
    return X_train_scaled, X_test_scaled, scaler


# =========================================================
# PHASE 4 — TUNING THE ENGINE: Choosing "K" (the elbow method)
# =========================================================
def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    # Cross-validate on the TRAINING set only — the test set stays untouched
    # until final evaluation. This also avoids blindly picking K=1, which
    # memorizes noise (see: "Tuning the Engine" — K=1 is overfitting).
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        error_rates.append(1 - scores.mean())

    best_k = k_range[int(np.argmin(error_rates))]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o", color="#1c3f60")
    plt.scatter([best_k], [error_rates[best_k - 1]], color="#d9531e", s=140,
                zorder=5, label=f"Optimal K = {best_k}")
    plt.title("Tuning the Engine: Choosing K", fontsize=14, fontweight="bold")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_curve.png"), dpi=150)
    plt.close()

    print("=" * 60)
    print("TUNING THE ENGINE: CHOOSING K")
    print("=" * 60)
    print(f"Optimal K found at the elbow: K = {best_k}")
    print()
    return best_k


# =========================================================
# PHASE 5 — THE WORKFLOW: scikit-learn (Instantiate, Fit, Predict)
# =========================================================
def train_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                    # FIT (memorize the map)
    return model


# =========================================================
# PHASE 6 — OUTPUT VALIDATION: Confusion Matrix + F1 Score
# =========================================================
def evaluate_model(model, X_test, y_test, class_names):
    predictions = model.predict(X_test)             # PREDICT (apply logic)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print("OUTPUT VALIDATION")
    print("=" * 60)
    print(f"Accuracy      : {acc:.2%}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, linewidths=1, linecolor="white"
    )
    plt.title("The Diagnostic Tool: Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    return acc, f1, cm


# =========================================================
# MAIN — THE FULL ARCHITECTURE (IPO Pipeline)
# =========================================================
def main():
    X, y, class_names = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    best_k = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)
    model = train_model(X_train_scaled, y_train, best_k)
    acc, f1, cm = evaluate_model(model, X_test_scaled, y_test, class_names)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final model: KNeighborsClassifier(n_neighbors={best_k})")
    print(f"Test Accuracy: {acc:.2%} | Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()