import numpy as np
import torch
import joblib

from momentfm import MOMENTPipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# VITALAGENT — WESAD FULL TEST EVALUATION
# MOMENT + RANDOM FOREST
# ============================================================

MODEL_PATH = "models/wesad_stress_classifier.pkl"

X_TEST_PATH = "data/processed/stress/splits/X_test.npy"
Y_TEST_PATH = "data/processed/stress/splits/y_test.npy"


print("\n" + "=" * 60)
print("VITALAGENT — WESAD STRESS MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. Load Random Forest
# ============================================================

print("\nLoading Random Forest...")

classifier = joblib.load(MODEL_PATH)

print("✅ Classifier loaded")


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
# 3. Load test data
# ============================================================

print("\nLoading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 4. Generate embeddings
# ============================================================

print("\nGenerating test embeddings...")

embeddings = []

batch_size = 32

for start in range(0, len(X_test), batch_size):

    end = min(start + batch_size, len(X_test))

    batch = torch.tensor(
        X_test[start:end],
        dtype=torch.float32
    ).unsqueeze(1)

    with torch.no_grad():

        output = moment(
            x_enc=batch
        )

    batch_embeddings = output.embeddings.cpu().numpy()

    embeddings.append(batch_embeddings)

    print(f"Processed {end}/{len(X_test)}")


X_embeddings = np.concatenate(
    embeddings,
    axis=0
)

print("\nEmbeddings shape:", X_embeddings.shape)


# ============================================================
# 5. Prediction
# ============================================================

print("\nGenerating predictions...")

y_pred = classifier.predict(X_embeddings)

y_probability = classifier.predict_proba(
    X_embeddings
)[:, 1]


# ============================================================
# 6. Metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 7. Display results
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NON-STRESS",
            "STRESS"
        ]
    )
)


# ============================================================
# 8. Sample predictions
# ============================================================

print("\nSample Predictions:")

for i in range(min(10, len(y_test))):

    true_text = (
        "STRESS"
        if y_test[i] == 1
        else "NON-STRESS"
    )

    pred_text = (
        "STRESS"
        if y_pred[i] == 1
        else "NON-STRESS"
    )

    print(
        f"{i + 1}. "
        f"Actual = {true_text} | "
        f"Predicted = {pred_text} | "
        f"Stress Probability = {y_probability[i]:.4f}"
    )


print("\n" + "=" * 60)
print("✅ WESAD TEST EVALUATION COMPLETED")
print("=" * 60)