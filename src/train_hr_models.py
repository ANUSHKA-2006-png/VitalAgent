import os
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 60)
print("VITALAGENT — HR MODEL COMPARISON")
print("=" * 60)


BASE = "data/processed/moment_embeddings/heart_rate"


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------

X_train = np.load(
    os.path.join(BASE, "train_embeddings.npy")
)

y_train = np.load(
    os.path.join(BASE, "train_y.npy")
)

X_val = np.load(
    os.path.join(BASE, "val_embeddings.npy")
)

y_val = np.load(
    os.path.join(BASE, "val_y.npy")
)

X_test = np.load(
    os.path.join(BASE, "test_embeddings.npy")
)

y_test = np.load(
    os.path.join(BASE, "test_y.npy")
)


print("\nDataset shapes:")

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate_model(name, model):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    r2 = r2_score(
        y_val,
        predictions
    )

    print(f"MAE :  {mae:.2f} BPM")
    print(f"RMSE:  {rmse:.2f} BPM")
    print(f"R²  :  {r2:.4f}")

    return model, mae


# ---------------------------------------------------------
# Model 1 — Ridge
# ---------------------------------------------------------

ridge = Ridge(
    alpha=10.0
)

ridge, ridge_mae = evaluate_model(
    "RIDGE REGRESSION",
    ridge
)


# ---------------------------------------------------------
# Model 2 — Random Forest
# ---------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

random_forest, rf_mae = evaluate_model(
    "RANDOM FOREST",
    random_forest
)


# ---------------------------------------------------------
# Choose best model
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"Ridge MAE:         {ridge_mae:.2f} BPM"
)

print(
    f"Random Forest MAE: {rf_mae:.2f} BPM"
)


if ridge_mae <= rf_mae:

    best_model = ridge
    best_name = "Ridge"

else:

    best_model = random_forest
    best_name = "Random Forest"


print(
    f"\n[BEST] Best validation model: {best_name}"
)


# ---------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


test_predictions = best_model.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print(f"Test MAE :  {test_mae:.2f} BPM")
print(f"Test RMSE:  {test_rmse:.2f} BPM")
print(f"Test R²  :  {test_r2:.4f}")


# ---------------------------------------------------------
# Example predictions
# ---------------------------------------------------------

print("\nSample predictions:")

for i in range(10):

    print(
        f"{i+1}. "
        f"Actual = {y_test[i]:.2f} BPM | "
        f"Predicted = {test_predictions[i]:.2f} BPM"
    )

import joblib
from pathlib import Path
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, models_dir / "ppg_dalia_hr_regressor.pkl")
joblib.dump(best_model, models_dir / "heart_rate_regressor.pkl")
print(f"\n[OK] Saved best model ({best_name}) to models/ppg_dalia_hr_regressor.pkl")


print("\n" + "=" * 60)
print("[OK] HR MODEL EVALUATION COMPLETED")
print("=" * 60)
