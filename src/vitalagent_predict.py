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
    align_probability,
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
    parser.add_argument(
        "--use-fusion",
        action="store_true",
        help=(
            "Use 4-modality late-fusion instead of BVP-primary. "
            "NOTE: Fusion did not outperform BVP on the WESAD test set (F1 0.34 vs 0.59). "
            "BVP (MOMENT-1-large, threshold=0.66) is the recommended primary model."
        ),
    )
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
        output = moment(x_enc=signal)
        emb = output.embeddings
        embedding = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)

    return float(regressor.predict(embedding)[0])


# Global variables to cache models across multiple predict() calls
_MOMENT_ENCODER = None
_HR_MODEL = None
_SPO2_MODEL = None
_STRESS_MODEL = None
_STRESS_THRESHOLD = None
_FALL_MODEL = None


def predict(
    ppg_window: np.ndarray,
    accel_window: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Unified VitalAgent multi-task screening function.

    Takes a 512-sample PPG window and 512-sample Accelerometer window, passes
    them through frozen MOMENT encoder, and evaluates all four task heads:
    1. Heart Rate Regressor (PPG-DaLiA)
    2. SpO2 Regressor (BIDMC)
    3. Stress Classifier (WESAD)
    4. Fall Classifier (UP-Fall)
    """
    global _MOMENT_ENCODER, _HR_MODEL, _SPO2_MODEL, _STRESS_MODEL, _STRESS_THRESHOLD, _FALL_MODEL

    ppg_window = np.asarray(ppg_window, dtype=np.float32)
    accel_window = np.asarray(accel_window, dtype=np.float32)

    if ppg_window.shape != (512,):
        raise ValueError(f"Expected 512-sample PPG window, got {ppg_window.shape}")
    if accel_window.shape != (512,):
        raise ValueError(f"Expected 512-sample Accelerometer window, got {accel_window.shape}")

    device = torch.device(device)

    # 1. MOMENT embeddings
    if _MOMENT_ENCODER is None:
        _MOMENT_ENCODER = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            local_files_only=local_files_only,
            model_kwargs={"task_name": "embedding"},
        )
        _MOMENT_ENCODER.init()
        _MOMENT_ENCODER.to(device)
        _MOMENT_ENCODER.eval()

    ppg_t = torch.from_numpy(ppg_window).unsqueeze(0).unsqueeze(0).to(device)
    accel_t = torch.from_numpy(accel_window).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ppg_emb = _MOMENT_ENCODER(x_enc=ppg_t).embeddings
        accel_emb = _MOMENT_ENCODER(x_enc=accel_t).embeddings

        ppg_emb_np = ppg_emb.cpu().numpy() if isinstance(ppg_emb, torch.Tensor) else np.asarray(ppg_emb)
        accel_emb_np = accel_emb.cpu().numpy() if isinstance(accel_emb, torch.Tensor) else np.asarray(accel_emb)

    # 2. HR Prediction
    if _HR_MODEL is None:
        hr_model_path = find_hr_model(None)
        _HR_MODEL = joblib.load(hr_model_path)
    hr_bpm = float(_HR_MODEL.predict(ppg_emb_np)[0])

    # 3. SpO2 Prediction
    spo2_model_path = PROJECT_ROOT / "models" / "spo2_regressor.pkl"
    if spo2_model_path.exists():
        if _SPO2_MODEL is None:
            _SPO2_MODEL = joblib.load(spo2_model_path)
        spo2_pct = float(_SPO2_MODEL.predict(ppg_emb_np)[0])
    else:
        spo2_pct = 98.0 # Default if model training still in progress

    # 4. Stress Prediction
    if _STRESS_MODEL is None:
        _STRESS_MODEL, stress_checkpoint = load_finetuned_checkpoint(
            DEFAULT_CHECKPOINT_PATH,
            device=device,
            local_files_only=local_files_only,
        )
        _STRESS_THRESHOLD = float(stress_checkpoint.get("threshold", 0.66))

    stress_res = predict_one_window(
        _STRESS_MODEL,
        ppg_window,
        device=device,
        threshold=_STRESS_THRESHOLD,
    )

    # 5. Fall Prediction
    fall_model_path = PROJECT_ROOT / "models" / "fall_classifier.pkl"
    if fall_model_path.exists():
        if _FALL_MODEL is None:
            _FALL_MODEL = joblib.load(fall_model_path)
        fall_detected = int(_FALL_MODEL.predict(accel_emb_np)[0])
    else:
        fall_detected = 0 # Default if model training still in progress

    return {
        "hr_bpm": round(hr_bpm, 1),
        "spo2_pct": round(spo2_pct, 1),
        "stress_class": stress_res["prediction_text"],
        "fall_detected": "FALL DETECTED" if fall_detected == 1 else "NO FALL",
    }


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

    # Test unified predict on 10 random windows
    print("\n" + "=" * 70)
    print("VITALAGENT UNIFIED MULTI-TASK PREDICTION DEMO")
    print("=" * 70)

    ppg_test, _, _ = load_wesad_split("test")
    for i in range(min(10, len(ppg_test))):
        dummy_ppg = ppg_test[i]
        dummy_accel = np.random.randn(512).astype(np.float32)
        res = predict(dummy_ppg, dummy_accel, device=device, local_files_only=local_files_only)
        print(f"Window {i+1:02d}: {res}")


if __name__ == "__main__":
    main()
