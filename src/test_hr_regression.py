import numpy as np
import torch
from momentfm import MOMENTPipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 60)
print("VITALAGENT — MOMENT + HEART RATE REGRESSION TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------

X_train = np.load(
    "data/processed/heart_rate/splits/X_train.npy"
)

y_train = np.load(
    "data/processed/heart_rate/splits/y_train.npy"
)

print("\nTraining data:")
print("X:", X_train.shape)
print("y:", y_train.shape)


# ---------------------------------------------------------
# 2. Load validation data
# ---------------------------------------------------------

X_val = np.load(
    "data/processed/heart_rate/splits/X_val.npy"
)

y_val = np.load(
    "data/processed/heart_rate/splits/y_val.npy"
)

print("\nValidation data:")
print("X:", X_val.shape)
print("y:", y_val.shape)


# ---------------------------------------------------------
# 3. Use small subset first
# ---------------------------------------------------------

N_TRAIN = 50
N_VAL = 20

X_train_small = X_train[:N_TRAIN]
y_train_small = y_train[:N_TRAIN]

X_val_small = X_val[:N_VAL]
y_val_small = y_val[:N_VAL]

print("\nSmall experiment:")
print("Training samples:", len(X_train_small))
print("Validation samples:", len(X_val_small))


# ---------------------------------------------------------
# 4. Convert to PyTorch
# ---------------------------------------------------------

X_train_tensor = torch.tensor(
    X_train_small,
    dtype=torch.float32
).unsqueeze(1)

X_val_tensor = torch.tensor(
    X_val_small,
    dtype=torch.float32
).unsqueeze(1)


print("\nTensor shapes:")
print("Train:", X_train_tensor.shape)
print("Val:", X_val_tensor.shape)


# ---------------------------------------------------------
# 5. Load MOMENT
# ---------------------------------------------------------

print("\nLoading MOMENT...")

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding"
    }
)

model.init()

model.eval()

print("MOMENT loaded successfully")


# ---------------------------------------------------------
# 6. Generate training embeddings
# ---------------------------------------------------------

print("\nGenerating training embeddings...")

with torch.no_grad():

    train_output = model(
        x_enc=X_train_tensor
    )

    train_embeddings = train_output.embeddings


print("Training embeddings:")
print(train_embeddings.shape)


# ---------------------------------------------------------
# 7. Generate validation embeddings
# ---------------------------------------------------------

print("\nGenerating validation embeddings...")

with torch.no_grad():

    val_output = model(
        x_enc=X_val_tensor
    )

    val_embeddings = val_output.embeddings


print("Validation embeddings:")
print(val_embeddings.shape)


# ---------------------------------------------------------
# 8. Convert to NumPy
# ---------------------------------------------------------

train_embeddings = train_embeddings.cpu().numpy()
val_embeddings = val_embeddings.cpu().numpy()


# ---------------------------------------------------------
# 9. Train simple regression model
# ---------------------------------------------------------

print("\nTraining regression head...")

regressor = Ridge(alpha=1.0)

regressor.fit(
    train_embeddings,
    y_train_small
)

print("Regression model trained")


# ---------------------------------------------------------
# 10. Predict HR
# ---------------------------------------------------------

predicted_hr = regressor.predict(
    val_embeddings
)


# ---------------------------------------------------------
# 11. Evaluate
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_val_small,
    predicted_hr
)

rmse = np.sqrt(
    mean_squared_error(
        y_val_small,
        predicted_hr
    )
)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.2f} BPM")
print(f"RMSE: {rmse:.2f} BPM")


# ---------------------------------------------------------
# 12. Show predictions
# ---------------------------------------------------------

print("\nSample predictions:")

for i in range(min(10, len(y_val_small))):

    print(
        f"Sample {i+1}: "
        f"True HR = {y_val_small[i]:.2f} BPM | "
        f"Predicted HR = {predicted_hr[i]:.2f} BPM"
    )


print("\n✅ MOMENT + HR regression pipeline completed!")