import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# VITALAGENT — WESAD STRESS MODEL COMPARISON
# ============================================================

BASE_DIR = Path("data/processed/stress/embeddings")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VITALAGENT — WESAD STRESS MODEL COMPARISON")
print("=" * 60)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X_train = np.load(BASE_DIR / "X_train_embeddings.npy")
X_val = np.load(BASE_DIR / "X_val_embeddings.npy")
X_test = np.load(BASE_DIR / "X_test_embeddings.npy")

y_train = np.load(BASE_DIR / "y_train.npy")
y_val = np.load(BASE_DIR / "y_val.npy")
y_test = np.load(BASE_DIR / "y_test.npy")


print("\nDataset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)


# ------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------

def evaluate_model(model, X, y, name):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    roc_auc = roc_auc_score(y, probabilities)

    print(f"\n{name}")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining...")
logistic.fit(X_train, y_train)

logistic_val = evaluate_model(
    logistic,
    X_val,
    y_val,
    "Logistic Regression — Validation"
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")
random_forest.fit(X_train, y_train)

rf_val = evaluate_model(
    random_forest,
    X_val,
    y_val,
    "Random Forest — Validation"
)


# ------------------------------------------------------------
# Select best model
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"\nLogistic Regression F1: "
    f"{logistic_val['f1']:.4f}"
)

print(
    f"Random Forest F1:      "
    f"{rf_val['f1']:.4f}"
)


if rf_val["f1"] > logistic_val["f1"]:

    best_model = random_forest
    best_name = "Random Forest"

else:

    best_model = logistic
    best_name = "Logistic Regression"


print(f"\n🏆 Best validation model: {best_name}")


# ------------------------------------------------------------
# Final test evaluation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = evaluate_model(
    best_model,
    X_test,
    y_test,
    f"{best_name} — TEST"
)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

model_path = MODEL_DIR / "wesad_stress_classifier.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nSaved model:")
print(model_path)

print("\n" + "=" * 60)
print("✅ WESAD STRESS CLASSIFICATION COMPLETED")
print("=" * 60)