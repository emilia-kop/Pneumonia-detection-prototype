"""
pneumonia_model.py - Pneumonia model loading and prediction.

(This file was called model.py originally. It was renamed because TorchXRayVision
imports an internal package called `model`, and our model.py shadowed it, which
broke `import torchxrayvision`.)

MODEL SOURCE:
  Library : TorchXRayVision (Cohen et al.), https://github.com/mlmed/torchxrayvision
  Weights : "densenet121-res224-all" (default) or "densenet121-res224-rsna".
            The installed library (checked: v1.5.7) downloads them on first use from
            the official GitHub release assets of mlmed/torchxrayvision
            (https://github.com/mlmed/torchxrayvision/releases/tag/v1)
            into ~/.torchxrayvision/models_data/ and caches them.
  Arch    : DenseNet121, 224x224 input, 1 channel, 18 pathology outputs

IMPORTANT - what this model is and is NOT:
  * These weights are trained on chest X-ray datasets (NOT just ImageNet).
    "all" = NIH, PadChest, CheXpert, MIMIC-CXR, Google-NIH, OpenI, RSNA.
    "rsna" = RSNA Pneumonia Detection Challenge only.
  * It is a MULTI-LABEL model with 18 outputs; one of them is "Pneumonia".
    It is NOT a native NORMAL/PNEUMONIA binary classifier.
  * The model output is sigmoid(logit) rescaled by the library so that 0.5 sits at the
    model's own per-pathology operating point (op_norm in torchxrayvision). It is NOT a
    calibrated probability, so we call it a "pneumonia score", never a "confidence".
  * We map it to a binary label with a threshold on the "Pneumonia" output.
    NORMAL here means "pneumonia not flagged by this model", not "healthy".

Educational prototype only - not a clinical diagnostic tool.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_WEIGHTS = os.environ.get("XRV_WEIGHTS", "densenet121-res224-all")
SUPPORTED_WEIGHTS = ("densenet121-res224-all", "densenet121-res224-rsna")
INPUT_SIZE = 224
DEFAULT_THRESHOLD = 0.5
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


class ModelLoadError(RuntimeError):
    """Raised when the model or its weights cannot be loaded."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file is not a usable image."""


@dataclass
class Prediction:
    label: str            # "NORMAL" or "PNEUMONIA"
    pneumonia_score: float  # model output for the "Pneumonia" class (op-point normalised)
    threshold: float
    weights: str
    all_scores: Optional[dict] = None  # every named model output, e.g. {"Lung Opacity": 0.41}


def _import_backends():
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as exc:  # pragma: no cover
        raise ModelLoadError(
            "Required packages are missing. Run:  pip install -r requirements.txt  "
            f"(details: {exc})"
        ) from exc
    return torch, xrv


def load_image(file_or_path) -> Image.Image:
    """Open and validate an image. Raises InvalidImageError on failure."""
    try:
        img = Image.open(file_or_path)
        img.load()  # force decode so truncated/corrupt files fail here
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("This file is not a valid JPG/PNG image.") from exc
    if min(img.size) < 64:
        raise InvalidImageError(f"Image is too small ({img.size[0]}x{img.size[1]} px).")
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Documented TorchXRayVision preprocessing:
      1. 8-bit image -> [-1024, 1024] range  (xrv.datasets.normalize)
      2. single channel
      3. center crop to square, resize to 224x224
    Returns array of shape (1, 224, 224), float32.
    """
    _, xrv = _import_backends()
    import torchvision.transforms as T

    arr = np.asarray(img.convert("L")).astype(np.float32)  # grayscale 2D
    arr = xrv.datasets.normalize(arr, 255)                 # [-1024, 1024]
    arr = arr[None, :, :]                                  # add channel dim
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(INPUT_SIZE)])
    return transform(arr).astype(np.float32)


class PneumoniaClassifier:
    """Loads the model ONCE; call predict() many times."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: Optional[str] = None):
        if weights not in SUPPORTED_WEIGHTS:
            raise ModelLoadError(f"Unsupported weights '{weights}'. Use one of {SUPPORTED_WEIGHTS}.")
        torch, xrv = _import_backends()
        self.torch = torch
        self.weights = weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Downloads weights from the official GitHub release on first use, then caches
            # them in ~/.torchxrayvision/models_data/ (no `from_hf_hub` option exists in
            # the installed library version).
            self.model = xrv.models.get_model(weights)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load weights '{weights}'. If this is the first run you need "
                "internet access (github.com) to download them once. You can also download "
                "the file manually into ~/.torchxrayvision/models_data/ (see README). "
                f"Details: {exc}"
            ) from exc
        self.model = self.model.to(self.device).eval()
        if "Pneumonia" not in self.model.pathologies:
            raise ModelLoadError("Loaded model has no 'Pneumonia' output - incompatible weights.")
        self.pneumonia_idx = list(self.model.pathologies).index("Pneumonia")

    def _forward(self, img: Image.Image) -> np.ndarray:
        x = preprocess(img)
        t = self.torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return self.model(t).cpu().numpy()[0]

    def pneumonia_score(self, img: Image.Image) -> float:
        return float(self._forward(img)[self.pneumonia_idx])

    def predict(self, img: Image.Image, threshold: float = DEFAULT_THRESHOLD) -> Prediction:
        out = self._forward(img)
        score = float(out[self.pneumonia_idx])
        # Only outputs the loaded weights actually define have a name (RSNA weights name just two).
        named = {name: float(v) for name, v in zip(self.model.pathologies, out) if name}
        return Prediction(label=label_from_score(score, threshold), pneumonia_score=score,
                          threshold=threshold, weights=self.weights, all_scores=named)


def label_from_score(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Pure function (no torch needed) - used by tests and the evaluation script."""
    return CLASS_NAMES[1] if score >= threshold else CLASS_NAMES[0]
