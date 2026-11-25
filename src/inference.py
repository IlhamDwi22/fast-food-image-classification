"""CLI tool for running fast-food image classification with the TFLite model."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tflite" / "model.tflite"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "models" / "tflite" / "label.txt"
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


def preprocess_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load an RGB image, resize to (height, width), and normalize to [0, 1]."""
    height, width = target_size
    with Image.open(image_path) as img:
        resized = img.convert("RGB").resize((width, height))
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


class FastFoodClassifier:
    """Wrapper around the exported TFLite model for single or batch inference."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        label_path: Path = DEFAULT_LABEL_PATH,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        self.labels = load_labels(label_path)
        self.interpreter = _load_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        self.target_size = (int(input_shape[1]), int(input_shape[2]))

    def predict(self, image_path: Path, top_k: int = 3) -> list[dict[str, object]]:
        """Run inference on image_path and return top_k predictions sorted by confidence."""
        input_tensor = preprocess_image(image_path, self.target_size)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        probabilities = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        k = max(1, min(top_k, len(self.labels)))
        top_indices = np.argsort(probabilities)[::-1][:k]

        return [
            {
                "label": self.labels[int(idx)],
                "confidence": round(float(probabilities[int(idx)]), 6),
            }
            for idx in top_indices
        ]


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
    parser.add_argument(
        "image",
        type=Path,
        help="Path to an image file or a directory of images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to .tflite model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help=f"Path to label.txt (default: {DEFAULT_LABEL_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output predictions in JSON format.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    image_paths = _collect_image_paths(args.image)
    if not image_paths:
        print(f"No supported images found in: {args.image}", file=sys.stderr)
        return 1

    classifier = FastFoodClassifier(model_path=args.model, label_path=args.labels)
    results = []

    for image_path in image_paths:
        preds = classifier.predict(image_path, top_k=args.top_k)
        results.append({"image": str(image_path), "predictions": preds})

    if args.as_json:
        print(json.dumps(results, indent=2))
        return 0

    for entry in results:
        print(f"\nImage: {entry['image']}")
        for rank, pred in enumerate(entry["predictions"], start=1):
            print(f"  {rank}. {pred['label']:<16} {pred['confidence'] * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
