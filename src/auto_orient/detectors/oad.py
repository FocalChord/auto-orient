"""Deep-OAD orientation angle detection using a ViT model."""

from __future__ import annotations

import os

# Force legacy Keras (Keras 2) for transformers TF model compat
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from . import DetectionResult, Orientation

_IMAGE_SIZE = 224
_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "weights" / "model-vit-ang-loss.h5"

# Module-level model cache
_model = None


def _get_model():
    """Build ViT architecture and load fine-tuned OAD weights. Cached after first call."""
    global _model
    if _model is not None:
        return _model

    from transformers import AutoConfig, TFAutoModel
    from tf_keras.models import Model
    from tf_keras import layers as L

    config = AutoConfig.from_pretrained("google/vit-base-patch16-224")
    vit_base = TFAutoModel.from_config(config)

    img_input = L.Input(shape=(3, _IMAGE_SIZE, _IMAGE_SIZE))
    x = vit_base(img_input)
    y = L.Dense(1, activation="linear")(x[-1])
    model = Model(img_input, y)
    model.load_weights(str(_WEIGHTS_PATH))

    _model = model
    return _model


def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """PIL Image → normalized CHW array."""
    img = img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)  # HWC → CHW


def _preprocess(image_path: str) -> np.ndarray:
    """Load image, resize to 224x224, normalize to [-1, 1], return CHW array."""
    return _preprocess_pil(Image.open(image_path))


def _angle_to_orientation(angle: float) -> tuple[Orientation, float]:
    """Convert predicted angle to nearest Orientation and confidence.

    The model predicts the rotation applied to the image.
    We round to the nearest 90° and map to the correction needed.

    Returns (orientation, confidence) where confidence is 0-1 based on
    proximity to the nearest 90° boundary.
    """
    # Normalize to 0-360
    angle = angle % 360

    nearest = round(angle / 90) * 90 % 360

    # Confidence: how close is the prediction to a clean 90° multiple
    dist = min(angle % 90, 90 - angle % 90)
    confidence = 1.0 - (dist / 45.0)

    # Model predicts rotation angle → map to correction
    # predicted ~0°   → image is correct → CORRECT
    # predicted ~90°  → fix by rotating 90° CW  → CW_90
    # predicted ~180° → fix by rotating 180°     → CW_180
    # predicted ~270° → fix by rotating 90° CCW  → CCW_90
    orientation_map = {
        0: Orientation.CORRECT,
        90: Orientation.CW_90,
        180: Orientation.CW_180,
        270: Orientation.CCW_90,
    }

    return orientation_map[nearest], confidence


def _verify_direction(model, image_path: str) -> tuple[Orientation, str]:
    """For 90°/270° predictions, rotate both ways and pick the direction
    that makes the image look most upright (closest to 0°).
    """
    img = Image.open(image_path)
    img_cw = img.rotate(-90, expand=True)
    img_ccw = img.rotate(90, expand=True)

    batch = np.stack([_preprocess_pil(img_cw), _preprocess_pil(img_ccw)])
    preds = model.predict(batch, verbose=0).flatten()

    dist_cw = min(abs(preds[0] % 360), abs(360 - preds[0] % 360))
    dist_ccw = min(abs(preds[1] % 360), abs(360 - preds[1] % 360))

    if dist_cw < dist_ccw:
        return Orientation.CW_90, f"verified CW ({preds[0]:.1f}° vs {preds[1]:.1f}°)"
    else:
        return Orientation.CCW_90, f"verified CCW ({preds[1]:.1f}° vs {preds[0]:.1f}°)"


def detect_oad(image_path: str) -> DetectionResult:
    """Run OAD model on a single image with direction verification."""
    model = _get_model()
    arr = _preprocess(image_path)
    batch = np.expand_dims(arr, 0)
    pred = model.predict(batch, verbose=0)[0][0]

    orientation, confidence = _angle_to_orientation(pred)
    details = f"predicted {pred:.1f}°"

    if orientation in (Orientation.CW_90, Orientation.CCW_90):
        orientation, verify_detail = _verify_direction(model, image_path)
        details = f"predicted {pred:.1f}°, {verify_detail}"

    return DetectionResult(
        orientation=orientation,
        confidence=confidence,
        method="oad",
        details=details,
    )


def detect_oad_batch(image_paths: List[str]) -> List[DetectionResult]:
    """Run OAD model on a batch of images with direction verification."""
    model = _get_model()
    batch = np.stack([_preprocess(p) for p in image_paths])
    preds = model.predict(batch, verbose=0).flatten()

    results = []
    for image_path, pred in zip(image_paths, preds):
        orientation, confidence = _angle_to_orientation(pred)
        details = f"predicted {pred:.1f}°"

        if orientation in (Orientation.CW_90, Orientation.CCW_90):
            orientation, verify_detail = _verify_direction(model, image_path)
            details = f"predicted {pred:.1f}°, {verify_detail}"

        results.append(DetectionResult(
            orientation=orientation,
            confidence=confidence,
            method="oad",
            details=details,
        ))

    return results
