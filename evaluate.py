"""
evaluate.py - Evaluate the pneumonia model on YOUR labelled, held-out images.

Expected folder layout (same as the Kaggle/Kermany "chest_xray/test" folder):
    <data-dir>/NORMAL/*.jpeg
    <data-dir>/PNEUMONIA/*.jpeg

Usage (Windows):
    python evaluate.py --data-dir chest_xray\\test
    python evaluate.py --data-dir chest_xray\\test --weights densenet121-res224-rsna --threshold 0.5

Prints accuracy, precision, recall, F1, ROC-AUC, confusion matrix and saves
eval_results/metrics.json. Nothing is hard-coded: the numbers come from the run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pneumonia_model import (CLASS_NAMES, DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, InvalidImageError,
                   ModelLoadError, PneumoniaClassifier, label_from_score, load_image)

IMG_EXT = {".jpg", ".jpeg", ".png"}


def collect_images(data_dir: Path):
    items = []
    for label_idx, name in enumerate(CLASS_NAMES):  # 0 = NORMAL, 1 = PNEUMONIA
        folder = data_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, label_idx))
    if not items:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return items


def compute_metrics(y_true, scores, threshold=DEFAULT_THRESHOLD) -> dict:
    """Pure function: metrics from true labels (0/1) and pneumonia scores."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "n_images": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"labels": list(CLASS_NAMES), "matrix": cm.tolist()},
    }
    result["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", type=Path, default=Path("eval_results"))
    args = ap.parse_args(argv)

    try:
        items = collect_images(args.data_dir)
        clf = PneumoniaClassifier(weights=args.weights)
    except (FileNotFoundError, ModelLoadError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    y_true, scores, skipped = [], [], 0
    for i, (path, y) in enumerate(items, 1):
        try:
            scores.append(clf.pneumonia_score(load_image(path)))
            y_true.append(y)
        except InvalidImageError:
            skipped += 1
        if i % 50 == 0:
            print(f"  processed {i}/{len(items)}")
    m = compute_metrics(y_true, scores, args.threshold)
    m.update({"weights": args.weights, "data_dir": str(args.data_dir), "skipped_invalid": skipped})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))
    cm = m["confusion_matrix"]["matrix"]
    print("\n=== Results (computed by THIS run) ===")
    for k in ("n_images", "accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:10s}: {m[k]}")
    print("Confusion matrix (rows=true, cols=pred) [NORMAL, PNEUMONIA]:")
    print(f"  NORMAL    : {cm[0]}\n  PNEUMONIA : {cm[1]}")
    print(f"Saved to {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
