import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "data" / "processed" / "moment_embeddings" / "spo2"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("VITALAGENT — SpO2 MODEL COMPARISON & TRAINING")
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

    # Candidate models evaluated on Validation MAE
    candidates = []

    # 1. Baseline Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    val_mae_ridge = mean_absolute_error(y_val, ridge.predict(X_val))
    candidates.append(("Unscaled Ridge (alpha=10.0)", ridge, val_mae_ridge))

    # 2. Scaled Ridge Grid Search
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha))
        ])
        pipe.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, pipe.predict(X_val))
        candidates.append((f"Scaled Ridge (alpha={alpha})", pipe, val_mae))

    # Select best model exclusively on Validation MAE
    candidates.sort(key=lambda x: x[2])
    best_name, best_model, best_val_mae = candidates[0]

    print(f"\nBest Validation Model: {best_name} (Val MAE: {best_val_mae:.3f}%)")

    # Final Test Evaluation
    test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (SpO2)")
    print("=" * 60)
    print(f"Test MAE :  {test_mae:.3f}% (Target: < 3%)")
    print(f"Test RMSE:  {test_rmse:.3f}%")
    print(f"Test R2  :  {test_r2:.4f}")

    print("\nSample predictions:")
    for i in range(min(10, len(y_test))):
        print(f"{i+1}. Actual = {y_test[i]:.2f}% | Predicted = {test_pred[i]:.2f}%")

    # Save model
    save_path = MODELS_DIR / "spo2_regressor.pkl"
    joblib.dump(best_model, save_path)
    print(f"\n[OK] Saved SpO2 model to {save_path}")

if __name__ == "__main__":
    main()
