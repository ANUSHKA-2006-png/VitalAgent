from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from momentfm import MOMENTPipeline

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    PROJECT_ROOT,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_one_window,
)
from wesad_modality_common import (
    MODALITY_MODEL_ROOT,
    load_model_artifact,
    predict_with_artifact,
)


DEFAULT_HR_CANDIDATES = [
    PROJECT_ROOT / "models" / "ppg_dalia_hr_regressor.pkl",
    PROJECT_ROOT / "models" / "heart_rate_regressor.pkl",
    PROJECT_ROOT / "models" / "hr_random_forest_regressor.pkl",
    PROJECT_ROOT / "models" / "ppg_dalia_hr_random_forest.pkl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VitalAgent multimodal stress prediction for one 8-second window."
    )
    parser.add_argument("--input-npy", type=Path, default=None)
    parser.add_argument("--wesad-test-index", type=int, default=0)
    parser.add_argument("--hr-model", type=Path, default=None)
    parser.add_argument("--stress-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--stress-threshold", type=float, default=None)
    parser.add_argument("--bvp-window", action="store_true", help="Interpret --input-npy as a BVP window only")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def find_hr_model(path: Path | None) -> Path:
    if path is not None:
        if path.exists():
            return path
        raise FileNotFoundError(f"HR model not found: {path}")

    for candidate in DEFAULT_HR_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"- {candidate}" for candidate in DEFAULT_HR_CANDIDATES)
    raise FileNotFoundError(
        "No serialized HR regressor was found. The completed HR report exists, "
        "but the trained regressor is not present under models/. To avoid "
        "retraining the completed HR pipeline inside this integration script, "
        "provide --hr-model or save the existing HR Random Forest model to one "
        f"of these paths:\n{candidates}"
    )


def load_window(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.input_npy is not None:
        window = np.load(args.input_npy)
        if window.ndim == 2 and window.shape[0] == 1:
            window = window[0]
        return np.asarray(window, dtype=np.float32), str(args.input_npy)

    X_test, _, _ = load_wesad_split("test")
    if args.wesad_test_index < 0 or args.wesad_test_index >= len(X_test):
        raise IndexError(
            f"wesad-test-index must be between 0 and {len(X_test) - 1}."
        )
    return X_test[args.wesad_test_index], f"WESAD test sample {args.wesad_test_index}"


def load_multimodal_windows(args: argparse.Namespace) -> dict[str, tuple[np.ndarray, str]]:
    if args.input_npy is None:
        raise ValueError("--input-npy must be provided for multimodal inference")

    payload = np.load(args.input_npy, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.ndim == 0:
        payload = payload.item()
    if isinstance(payload, dict):
        windows = payload
    else:
        raise ValueError("Expected a .npy file containing a dict of modality windows")

    result: dict[str, tuple[np.ndarray, str]] = {}
    for modality in ("bvp", "eda", "temp", "acc"):
        if modality not in windows:
            raise KeyError(f"Missing {modality} window in {args.input_npy}")
        window = np.asarray(windows[modality], dtype=np.float32)
        result[modality] = (window, f"{modality} from {args.input_npy}")
    return result


def predict_heart_rate(
    window: np.ndarray,
    *,
    hr_model_path: Path,
    device: torch.device,
    local_files_only: bool,
) -> float:
    regressor = joblib.load(hr_model_path)
    moment = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        local_files_only=local_files_only,
        model_kwargs={"task_name": "embedding"},
    )
    moment.init()
    moment.to(device)
    moment.eval()

    signal = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = moment(x_enc=signal).embeddings.cpu().numpy()

    return float(regressor.predict(embedding)[0])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only = not args.allow_download

    if args.bvp_window:
        window, source = load_window(args)
        if window.shape != (512,):
            raise ValueError(f"Expected a 512-sample BVP window, got {window.shape}.")
        if not np.isfinite(window).all():
            raise ValueError("Input BVP window contains NaN or Inf values.")

        hr_model_path = find_hr_model(args.hr_model)
        stress_model, stress_checkpoint = load_finetuned_checkpoint(
            args.stress_checkpoint,
            device=device,
            local_files_only=local_files_only,
        )
        stress_threshold = (
            float(args.stress_threshold)
            if args.stress_threshold is not None
            else float(stress_checkpoint.get("threshold", 0.5))
        )
        heart_rate = predict_heart_rate(
            window,
            hr_model_path=hr_model_path,
            device=device,
            local_files_only=local_files_only,
        )
        stress_result = predict_one_window(
            stress_model,
            window,
            device=device,
            threshold=stress_threshold,
        )

        print("\n" + "=" * 70)
        print("VITALAGENT HEALTH STATE")
        print("=" * 70)
        print(f"Input: {source}")
        print(f"Heart Rate: {heart_rate:.1f} BPM")
        print(f"Stress: {stress_result['prediction_text']}")
        print(f"Stress Probability: {stress_result['stress_probability'] * 100:.2f}%")
        print(f"Non-stress Probability: {stress_result['non_stress_probability'] * 100:.2f}%")
        return

    windows = load_multimodal_windows(args)
    bvp_window = windows["bvp"]
    eda_window = windows["eda"]
    temp_window = windows["temp"]
    acc_window = windows["acc"]

    for modality, (window, source) in {"bvp": bvp_window, "eda": eda_window, "temp": temp_window, "acc": acc_window}.items():
        if not np.isfinite(window).all():
            raise ValueError(f"{modality} input contains NaN or Inf values.")

    if acc_window[0].ndim == 1:
        acc_window = (acc_window[0].reshape(-1, 3), acc_window[1])
    if bvp_window[0].ndim != 1:
        raise ValueError(f"Expected BVP window to be a 1D vector, got {bvp_window[0].shape}")
    if eda_window[0].ndim != 1:
        raise ValueError(f"Expected EDA window to be a 1D vector, got {eda_window[0].shape}")
    if temp_window[0].ndim != 1:
        raise ValueError(f"Expected TEMP window to be a 1D vector, got {temp_window[0].shape}")

    eda_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "eda_model.joblib")
    temp_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "temp_model.joblib")
    acc_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "acc_model.joblib")
    fusion_artifact = load_model_artifact(MODALITY_MODEL_ROOT / "fusion_model.joblib")

    bvp_model, _ = load_finetuned_checkpoint(
        args.stress_checkpoint,
        device=device,
        local_files_only=local_files_only,
    )
    bvp_probs = np.asarray(
        [
            predict_one_window(
                bvp_model,
                bvp_window[0],
                device=device,
                threshold=0.66,
            )["stress_probability"]
        ],
        dtype=np.float32,
    )
    eda_probs = predict_with_artifact(eda_artifact, eda_window[0].reshape(1, -1))
    temp_probs = predict_with_artifact(temp_artifact, temp_window[0].reshape(1, -1))
    acc_window_for_model = acc_window[0].reshape(1, -1, 3)
    acc_probs = predict_with_artifact(acc_artifact, acc_window_for_model)

    fusion_features = np.column_stack([bvp_probs[0], eda_probs[0], temp_probs[0], acc_probs[0]])
    fused_probability = float(fusion_artifact["model"].predict_proba(fusion_features.reshape(1, -1))[0, 1])
    fused_prediction = int(fused_probability >= float(fusion_artifact["threshold"]))

    print("\n" + "=" * 70)
    print("VITALAGENT MULTIMODAL STRESS PREDICTION")
    print("=" * 70)
    print(f"Input: {args.input_npy}")
    print(f"BVP Stress Probability: {bvp_probs[0] * 100:.2f}%")
    print(f"EDA Stress Probability: {eda_probs[0] * 100:.2f}%")
    print(f"TEMP Stress Probability: {temp_probs[0] * 100:.2f}%")
    print(f"ACC Stress Probability: {acc_probs[0] * 100:.2f}%")
    print(f"Fused Stress Probability: {fused_probability * 100:.2f}%")
    print(f"Final Prediction: {'STRESS' if fused_prediction else 'NON-STRESS'}")


if __name__ == "__main__":
    main()
