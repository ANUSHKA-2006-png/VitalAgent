from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wesad_finetuned_common import (
    DEFAULT_CHECKPOINT_PATH,
    WesadBvpDataset,
    load_finetuned_checkpoint,
    load_wesad_split,
    predict_probabilities,
)
from wesad_modality_common import MODALITY_MODEL_ROOT, ensure_directory, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BVP probability vectors for the WESAD splits")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODALITY_MODEL_ROOT / "bvp_probabilities")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_finetuned_checkpoint(
        args.checkpoint,
        device=device,
        local_files_only=not args.allow_download,
    )

    payload = {}
    for split in ("train", "val", "test"):
        X, y, _ = load_wesad_split(split)
        loader = torch.utils.data.DataLoader(
            WesadBvpDataset(X, y),
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        probabilities, labels = predict_probabilities(model, loader, device=device)
        np.save(output_dir / f"{split}_probabilities.npy", probabilities.astype(np.float32))
        np.save(output_dir / f"{split}_labels.npy", labels.astype(np.int64))
        payload[split] = {"shape": list(probabilities.shape), "label_shape": list(labels.shape)}

    save_json(output_dir / "manifest.json", payload)
    print(f"Saved BVP probabilities to {output_dir}")


if __name__ == "__main__":
    main()
