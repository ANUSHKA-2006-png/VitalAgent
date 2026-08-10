import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data" / "processed" / "moment_embeddings" / "fall"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("VITALAGENT — FALL MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train = np.load(BASE / "train_embeddings.npy")
    y_train = np.load(BASE / "train_y.npy")

    X_val = np.load(BASE / "val_embeddings.npy")
    y_val = np.load(BASE / "val_y.npy")

    X_test = np.load(BASE / "test_embeddings.npy")
    y_test = np.load(BASE / "test_y.npy")

    print("\nDataset shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_pred_rf = rf.predict(X_val)
    f1_rf = f1_score(y_val, val_pred_rf, zero_division=0)
    print(f"\nRandom Forest Validation F1: {f1_rf:.4f}")

    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    val_pred_lr = lr.predict(X_val)
    f1_lr = f1_score(y_val, val_pred_lr, zero_division=0)
    print(f"Logistic Regression Validation F1: {f1_lr:.4f}")

    if f1_rf >= f1_lr:
        best_model = rf
        best_name = "Random Forest"
        best_f1 = f1_rf
    else:
        best_model = lr
        best_name = "Logistic Regression"
        best_f1 = f1_lr

    print(f"\nBest Validation Model: {best_name} (F1: {best_f1:.4f})")

    # Final Test Evaluation
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    cm = confusion_matrix(y_test, test_pred)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (Fall Detection)")
    print("=" * 60)
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1 Score : {test_f1:.4f} (Target: > 0.85)")
    print("Confusion Matrix:")
    print(cm)

    # Save model
    save_path = MODELS_DIR / "fall_classifier.pkl"
    joblib.dump(best_model, save_path)
    print(f"\n[OK] Saved Fall model to {save_path}")

if __name__ == "__main__":
    main()
