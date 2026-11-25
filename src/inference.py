"""CLI tool for running fast-food image classification with TFLite and OOD gating."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tflite" / "model.tflite"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "models" / "tflite" / "label.txt"
DEFAULT_META_PATH = PROJECT_ROOT / "models" / "model_meta.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_interpreter(model_path: Path):
    """Instantiate a TFLite interpreter using LiteRT, tflite-runtime, or TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    import tensorflow as tf

    # tf.lite.Interpreter emits a deprecation warning on TF >= 2.20 when LiteRT is absent.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return tf.lite.Interpreter(model_path=str(model_path))


def load_labels(label_path: Path) -> list[str]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Label file is empty: {label_path}")
    return labels


def load_model_meta(meta_path: Path | None) -> dict[str, object]:
    if meta_path is not None and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "version": "v1",
        "preprocessing": "rescale_0_1",
        "crop": "center_square",
        "other_class": None,
        "ood": {"enabled": False},
    }


def _center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width == height:
        return img
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def preprocess_image(
    image_path: Path,
    target_size: tuple[int, int],
    preprocessing: str = "rescale_0_1",
    crop_mode: str = "center_square",
) -> np.ndarray:
    """Load RGB image, apply optional center-square crop, resize, and scale pixels."""
    height, width = target_size
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        if crop_mode == "center_square":
            img = _center_crop_square(img)
        resized = img.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)

    if preprocessing == "mobilenet_v2":
        array = (array / 127.5) - 1.0
    else:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def _shannon_entropy(probabilities: np.ndarray) -> float:
    probs = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


class FastFoodClassifier:
    """Wrapper around the exported TFLite model with metadata-driven preprocessing and OOD gating."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        label_path: Path = DEFAULT_LABEL_PATH,
        meta_path: Path | None = DEFAULT_META_PATH,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.meta = load_model_meta(meta_path)
        self.labels = load_labels(label_path)
        self.preprocessing = str(self.meta.get("preprocessing", "rescale_0_1"))
        self.crop_mode = str(self.meta.get("crop", "center_square"))
        self.other_class = self.meta.get("other_class")
        self.ood_cfg = dict(self.meta.get("ood") or {})

        self.interpreter = _load_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        self.target_size = (int(input_shape[1]), int(input_shape[2]))

    def _evaluate_ood(
        self,
        top_label: str,
        top_confidence: float,
        entropy: float,
    ) -> tuple[bool, str | None]:
        if self.other_class and top_label == self.other_class:
            return True, f"Predicted explicit out-of-domain class '{self.other_class}'."

        if not self.ood_cfg.get("enabled", False):
            return False, None

        min_conf = self.ood_cfg.get("min_confidence")
        max_ent = self.ood_cfg.get("max_entropy")
        if min_conf is not None and top_confidence < float(min_conf):
            return True, f"Top confidence ({top_confidence * 100:.1f}%) is below threshold ({float(min_conf) * 100:.0f}%)."
        if max_ent is not None and entropy > float(max_ent):
            return True, f"Prediction entropy ({entropy:.2f}) exceeds threshold ({float(max_ent):.2f})."
        return False, None

    def predict(self, image_path: Path, top_k: int = 3) -> dict[str, object]:
        """Run inference on image_path and return ranked predictions plus OOD assessment."""
        input_tensor = preprocess_image(
            image_path=image_path,
            target_size=self.target_size,
            preprocessing=self.preprocessing,
            crop_mode=self.crop_mode,
        )
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        probabilities = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        k = max(1, min(top_k, len(self.labels)))
        top_indices = np.argsort(probabilities)[::-1][:k]

        predictions = [
            {
                "label": self.labels[int(idx)],
                "confidence": round(float(probabilities[int(idx)]), 6),
            }
            for idx in top_indices
        ]
        entropy = _shannon_entropy(probabilities)
        is_ood, ood_reason = self._evaluate_ood(
            top_label=predictions[0]["label"],
            top_confidence=predictions[0]["confidence"],
            entropy=entropy,
        )
        return {
            "top_label": "Out-of-Distribution" if is_ood else predictions[0]["label"],
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "entropy": round(entropy, 4),
            "predictions": predictions,
        }


def _collect_image_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Image path not found: {target}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict fast food category from an image or folder using TFLite."
    )
    parser.add_argument("image", type=Path, help="Path to an image file or a directory of images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    image_paths = _collect_image_paths(args.image)
    if not image_paths:
        print(f"No supported images found in: {args.image}", file=sys.stderr)
        return 1

    classifier = FastFoodClassifier(
        model_path=args.model,
        label_path=args.labels,
        meta_path=args.meta,
    )
    results = []

    for image_path in image_paths:
        outcome = classifier.predict(image_path, top_k=args.top_k)
        results.append({"image": str(image_path), **outcome})

    if args.as_json:
        print(json.dumps(results, indent=2))
        return 0

    for entry in results:
        print(f"\nImage: {entry['image']}")
        if entry["is_ood"]:
            print(f"  [OOD] Outside 6 fast food categories ({entry['ood_reason']})")
        for rank, pred in enumerate(entry["predictions"], start=1):
            print(f"  {rank}. {pred['label']:<16} {pred['confidence'] * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
