import numpy as np
import torch
import joblib
from momentfm import MOMENTPipeline


# ============================================================
# VITALAGENT — WESAD STRESS PREDICTION
# MOMENT Embedding + Random Forest Classifier
# ============================================================

MODEL_PATH = "models/wesad_stress_classifier.pkl"
TEST_X_PATH = "data/processed/stress/splits/X_test.npy"
TEST_Y_PATH = "data/processed/stress/splits/y_test.npy"


print("\n" + "=" * 60)
print("VITALAGENT — STRESS PREDICTION")
print("=" * 60)


# ============================================================
# 1. Load trained Random Forest model
# ============================================================

print("\nLoading trained classifier...")

classifier = joblib.load(MODEL_PATH)

print("✅ Random Forest classifier loaded")


# ============================================================
# 2. Load MOMENT
# ============================================================

print("\nLoading MOMENT...")

moment = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        "task_name": "embedding"
    }
)

moment.init()
moment.eval()

print("✅ MOMENT loaded")


# ============================================================
# 3. Load WESAD test data
# ============================================================

print("\nLoading WESAD test data...")

X_test = np.load(TEST_X_PATH)
y_test = np.load(TEST_Y_PATH)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 4. Select one test sample
# ============================================================

sample_index = 0

sample = X_test[sample_index]
true_label = y_test[sample_index]

print("\nSelected sample:", sample_index)
print("Sample shape:", sample.shape)

print(
    "True label:",
    "STRESS" if true_label == 1 else "NON-STRESS"
)


# ============================================================
# 5. Convert BVP → PyTorch tensor
# ============================================================

input_tensor = torch.tensor(
    sample,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print("\nInput tensor shape:", input_tensor.shape)


# Expected:
# [batch, channels, sequence]
# [1, 1, 512]


# ============================================================
# 6. Generate MOMENT embedding
# ============================================================

print("\nGenerating MOMENT embedding...")

with torch.no_grad():

    output = moment(
        x_enc=input_tensor
    )

embedding = output.embeddings

print("Embedding shape:", embedding.shape)


# Expected:
# [1, 1024]


# ============================================================
# 7. Convert embedding to NumPy
# ============================================================

embedding_np = embedding.cpu().numpy()

print("Embedding dtype:", embedding_np.dtype)


# ============================================================
# 8. Predict stress
# ============================================================

prediction = classifier.predict(
    embedding_np
)[0]

probabilities = classifier.predict_proba(
    embedding_np
)[0]


# ============================================================
# 9. Display result
# ============================================================

predicted_text = (
    "STRESS"
    if prediction == 1
    else "NON-STRESS"
)

true_text = (
    "STRESS"
    if true_label == 1
    else "NON-STRESS"
)


print("\n" + "=" * 60)
print("STRESS PREDICTION RESULT")
print("=" * 60)

print("True label       :", true_text)
print("Predicted label  :", predicted_text)

print(
    "Non-Stress probability:",
    f"{probabilities[0]:.4f}"
)

print(
    "Stress probability    :",
    f"{probabilities[1]:.4f}"
)

print("=" * 60)

if prediction == true_label:
    print("✅ Prediction is CORRECT")
else:
    print("❌ Prediction is INCORRECT")

print("=" * 60)