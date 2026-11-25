"""Train, evaluate, and export the MobileNetV2 Fast Food image classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_SIZE = (224, 224)


class ThresholdCallback(Callback):
    """Stop training once both training and validation accuracy reach target_acc."""

    def __init__(self, target_acc: float = 0.95) -> None:
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        train_acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)

        if train_acc >= self.target_acc and val_acc >= self.target_acc:
            print(
                f"\nTarget accuracy {self.target_acc * 100:.1f}% reached at epoch {epoch + 1} "
                f"(train: {train_acc * 100:.2f}%, val: {val_acc * 100:.2f}%). Stopping training."
            )
            self.model.stop_training = True


def create_data_generators(
    split_dir: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
):
    """Build train, validation, and test generators with augmentation on train only."""
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"

    for directory in (train_dir, val_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected split directory not found: {directory}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(val_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build the Sequential MobileNetV2 transfer learning architecture."""
    base_model = MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plots(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation accuracy/loss curves to output_path."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    axes[1].plot(epochs_range, loss, label="Training Loss", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _export_tfjs_manual(model: tf.keras.Model, tfjs_dir: Path) -> None:
    """Fallback TFJS layers-model serializer when tensorflowjs converter fails on TF 2.16+."""
    weight_data = bytearray()
    weight_manifest_entries = []

    for weight_tensor, var in zip(model.get_weights(), model.weights):
        weight_manifest_entries.append(
            {
                "name": var.name.split(":")[0],
                "shape": list(weight_tensor.shape),
                "dtype": "float32",
            }
        )
        weight_data.extend(weight_tensor.astype(np.float32).tobytes())

    bin_filename = "group1-shard1of1.bin"
    (tfjs_dir / bin_filename).write_bytes(bytes(weight_data))

    tfjs_manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.__version__}",
        "convertedBy": "TensorFlow.js Converter Fallback",
        "modelTopology": json.loads(model.to_json()),
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": weight_manifest_entries,
            }
        ],
    }
    (tfjs_dir / "model.json").write_text(json.dumps(tfjs_manifest, indent=2), encoding="utf-8")


def export_models(
    model: tf.keras.Model,
    class_indices: dict[str, int],
    models_dir: Path,
) -> None:
    """Export trained model to SavedModel, TFLite (with label.txt), and TFJS formats."""
    saved_model_dir = models_dir / "saved_model"
    tflite_dir = models_dir / "tflite"
    tfjs_dir = models_dir / "tfjs_model"

    for directory in (saved_model_dir, tflite_dir, tfjs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:
        tf.saved_model.save(model, str(saved_model_dir))
    print(f"SavedModel exported to: {saved_model_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path = tflite_dir / "model.tflite"
    tflite_path.write_bytes(tflite_bytes)

    ordered_labels = sorted(class_indices.keys(), key=lambda label: class_indices[label])
    label_path = tflite_dir / "label.txt"
    label_path.write_text("\n".join(ordered_labels) + "\n", encoding="utf-8")
    print(f"TFLite model exported to: {tflite_path}")
    print(f"Labels saved to: {label_path}")

    try:
        import tensorflowjs as tfjs

        tfjs.converters.save_keras_model(model, str(tfjs_dir))
        print(f"TFJS model exported via tensorflowjs to: {tfjs_dir}")
    except Exception as exc:
        print(f"tensorflowjs converter unavailable ({exc}); using fallback serializer.")
        _export_tfjs_manual(model, tfjs_dir)
        print(f"TFJS model exported to: {tfjs_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and export the Fast Food image classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dataset_split"),
        help="Directory containing train/, val/, and test/ folders (default: data/dataset_split).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for saved_model, tflite, and tfjs_model (default: models).",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("training_history.png"),
        help="Output path for the accuracy/loss curve image (default: training_history.png).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    train_gen, val_gen, test_gen = create_data_generators(
        split_dir=args.data_dir,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
    )

    model = build_model(
        num_classes=len(train_gen.class_indices),
        image_size=IMAGE_SIZE,
        learning_rate=args.lr,
    )
    model.summary()

    callbacks = [
        ThresholdCallback(target_acc=args.target_acc),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    save_training_plots(history, args.plot_path)
    print(f"Training plot saved to: {args.plot_path}")

    export_models(model, train_gen.class_indices, args.models_dir)


if __name__ == "__main__":
    main()
