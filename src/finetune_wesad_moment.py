from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wesad_finetuned_common import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METRICS_PATH,
    FINETUNED_DIR,
    MOMENT_MODEL_ID,
    WesadBvpDataset,
    class_distribution,
    compute_binary_metrics,
    compute_class_weights,
    count_parameters,
    load_fresh_moment_classifier,
    load_wesad_split,
    predict_probabilities,
    print_metrics,
    save_json,
    stratified_subset,
    verify_no_subject_overlap,
    verify_wesad_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MOMENT for WESAD stress classification."
    )
    parser.add_argument("--stage", choices=["head", "encoder", "full"], default="head")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def effective_learning_rate(args: argparse.Namespace) -> float:
    if args.lr is not None:
        return args.lr
    if args.stage == "head":
        return 1e-3
    return 1e-5


def print_split_report(split: str, checks: dict[str, Any]) -> None:
    print(f"\n{split.upper()} split")
    print(f"X shape: {tuple(checks['X_shape'])} ({checks['X_dtype']})")
    print(f"y shape: {tuple(checks['y_shape'])} ({checks['y_dtype']})")
    print(f"Subjects: {checks['subjects']}")
    print(f"Class distribution: {checks['class_distribution']}")
    print(f"Finite: {checks['is_finite']}")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        WesadBvpDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def trainable_parameters(model: torch.nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def set_training_mode(model: torch.nn.Module, stage: str) -> None:
    model.train()
    if stage == "head":
        # Keep frozen MOMENT features deterministic while training the random head.
        model.patch_embedding.eval()
        model.encoder.eval()
        model.head.train()


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    probabilities, labels = predict_probabilities(model, loader, device=device)
    if labels is None:
        raise ValueError("Evaluation requires labels.")
    return compute_binary_metrics(labels, probabilities, threshold=threshold)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    checkpoint_type: str,
    model_kwargs: dict[str, Any],
    args: argparse.Namespace,
    best_epoch: int,
    best_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    split_checks: dict[str, Any],
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_type": checkpoint_type,
        "moment_model_id": MOMENT_MODEL_ID,
        "model_kwargs": model_kwargs,
        "class_names": list(CLASS_NAMES),
        "threshold": float(args.threshold),
        "stage": args.stage,
        "best_epoch": int(best_epoch),
        "best_validation_metrics": best_metrics,
        "history": history,
        "split_checks": split_checks,
        "parameter_counts": parameter_counts,
        "training_args": vars(args),
        "input": {
            "sampling_rate_hz": 64,
            "window_samples": 512,
            "window_seconds": 8,
            "shape": ["batch", 1, 512],
        },
    }
    if checkpoint_type == "head_only":
        payload["head_state_dict"] = model.head.state_dict()
    else:
        payload["model_state_dict"] = model.state_dict()
    return payload


def main() -> None:
    args = parse_args()

    if args.sanity_check:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.max_train_samples = args.max_train_samples or 32
        args.max_val_samples = args.max_val_samples or 32
        if args.checkpoint_path == DEFAULT_CHECKPOINT_PATH:
            args.checkpoint_path = FINETUNED_DIR / "moment_stress_classifier_sanity.pt"
        if args.metrics_path == DEFAULT_METRICS_PATH:
            args.metrics_path = FINETUNED_DIR / "metrics_sanity.json"

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only = not args.allow_download
    lr = effective_learning_rate(args)
    checkpoint_type = "head_only" if args.stage == "head" else "full_state_dict"

    print("\n" + "=" * 70)
    print("VITALAGENT - WESAD MOMENT FINE-TUNING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"MOMENT: {MOMENT_MODEL_ID}")
    print(f"Stage: {args.stage}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Metrics: {args.metrics_path}")
    print(f"Local files only: {local_files_only}")

    print("\nLoading and verifying WESAD subject-wise splits...")
    split_data = {}
    split_checks = {}
    metadata_by_split = {}

    for split in ("train", "val", "test"):
        X, y, metadata = load_wesad_split(split)
        checks = verify_wesad_split(split, X, y, metadata)
        split_data[split] = (X, y, metadata)
        split_checks[split] = checks
        metadata_by_split[split] = metadata
        print_split_report(split, checks)

    overlaps = verify_no_subject_overlap(metadata_by_split)
    split_checks["subject_overlap"] = overlaps
    print(f"\nSubject overlap checks: {overlaps}")

    X_train, y_train, train_metadata = split_data["train"]
    X_val, y_val, val_metadata = split_data["val"]

    X_train, y_train, train_metadata = stratified_subset(
        X_train, y_train, train_metadata, args.max_train_samples, args.seed
    )
    X_val, y_val, val_metadata = stratified_subset(
        X_val, y_val, val_metadata, args.max_val_samples, args.seed
    )

    print("\nTraining data used:")
    print(f"Train: {X_train.shape}, classes {class_distribution(y_train)}")
    print(f"Val:   {X_val.shape}, classes {class_distribution(y_val)}")

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\nLoading MOMENT classifier...")
    model, model_kwargs = load_fresh_moment_classifier(
        stage=args.stage,
        head_dropout=args.head_dropout,
        local_files_only=local_files_only,
        device=device,
    )
    parameter_counts = count_parameters(model)
    print(f"Parameter counts: {parameter_counts}")
    print(f"Model kwargs: {model_kwargs}")
    print("Classification head is randomly initialized and will be fine-tuned.")

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    class_weights = compute_class_weights(y_train)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    print("\nTraining setup:")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Class weights [non-stress, stress]: {class_weights.tolist()}")
    print(f"Selection metric: validation F1 at threshold {args.threshold}")

    best_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        set_training_mode(model, args.stage)
        total_loss = 0.0
        total_examples = 0

        for batch_index, (signals, input_mask, labels) in enumerate(train_loader, start=1):
            signals = signals.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(x_enc=signals, input_mask=input_mask)
            loss = criterion(output.logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {batch_index}/{len(train_loader)} | "
                    f"loss {loss.item():.4f}"
                )

        train_loss = total_loss / max(total_examples, 1)
        val_metrics = evaluate_loader(
            model,
            val_loader,
            device=device,
            threshold=args.threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation": val_metrics,
        }
        history.append(epoch_record)

        print(f"\nEpoch {epoch} summary")
        print(f"Train loss: {train_loss:.4f}")
        print_metrics("Validation", val_metrics)

        val_f1 = float(val_metrics["f1"])
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    checkpoint_type=checkpoint_type,
                    model_kwargs=model_kwargs,
                    args=args,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    history=history,
                    split_checks=split_checks,
                    parameter_counts=parameter_counts,
                ),
                args.checkpoint_path,
            )
            print(f"Saved new best checkpoint: {args.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No validation F1 improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    elapsed_seconds = time.time() - start_time
    if best_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics.")

    metrics_payload = {
        "moment_model_id": MOMENT_MODEL_ID,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_type": checkpoint_type,
        "stage": args.stage,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "history": history,
        "split_checks": split_checks,
        "class_weights": class_weights.tolist(),
        "parameter_counts": parameter_counts,
        "elapsed_seconds": elapsed_seconds,
        "training_args": vars(args),
    }

    if args.evaluate_test:
        print("\nEvaluating held-out test split with the in-memory best checkpoint...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if checkpoint_type == "head_only":
            model.head.load_state_dict(checkpoint["head_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        X_test, y_test, _ = split_data["test"]
        test_loader = make_loader(
            X_test,
            y_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_metrics = evaluate_loader(
            model,
            test_loader,
            device=device,
            threshold=args.threshold,
        )
        metrics_payload["test_metrics"] = test_metrics
        print_metrics("Held-out Test", test_metrics)

    save_json(args.metrics_path, metrics_payload)

    print("\n" + "=" * 70)
    print("WESAD MOMENT FINE-TUNING FINISHED")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_metrics['f1']:.4f}")
    print(f"Checkpoint saved: {args.checkpoint_path}")
    print(f"Metrics saved: {args.metrics_path}")


if __name__ == "__main__":
    main()
