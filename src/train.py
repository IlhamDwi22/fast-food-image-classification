"""2-stage MobileNetV2 fine-tuning with fast tf.data pipeline, OOD calibration, and multi-format export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
EMBEDDING_LAYER_NAME = "embedding_gap"
OTHER_CLASS_NAME = "Other"


class ThresholdCallback(Callback):
    """Stop training once both training and validation accuracy reach target_acc."""

    def __init__(self, target_acc: float = 0.96) -> None:
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


def _cast_to_uint8(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Cache resized images as uint8 (~4x less RAM than float32) before epoch-level augmentation.
    return tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8), label


def _preprocess_mobilenet(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # MobileNetV2 ImageNet weights expect [-1, 1] via (x / 127.5) - 1.0.
    scaled = (tf.cast(image, tf.float32) / 127.5) - 1.0
    return scaled, label


def _augment_and_preprocess(
    image: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.cast(image, tf.float32)
    img = tf.image.random_flip_left_right(img)

    # Upscale slightly then random-crop back to target size for border-free zoom and shift jitter.
    height, width = image_size
    upscaled = tf.image.resize(img, [height + 28, width + 28])
    img = tf.image.random_crop(upscaled, size=[height, width, 3])

    img = tf.image.random_brightness(img, max_delta=25.0)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    scaled = (img / 127.5) - 1.0
    return scaled, label


def _load_directory_dataset(
    directory: Path,
    image_size: tuple[int, int],
    shuffle: bool,
    seed: int = 42,
):
    return tf.keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=None,
        shuffle=shuffle,
        seed=seed if shuffle else None,
        crop_to_aspect_ratio=True,
    )


def create_datasets(
    split_dir: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 64,
    cache_in_ram: bool = True,
    seed: int = 42,
):
    """Build cached, prefetched tf.data pipelines for train, clean-train, val, and test."""
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"

    for directory in (train_dir, val_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected split directory not found: {directory}")

    raw_train = _load_directory_dataset(train_dir, image_size, shuffle=True, seed=seed)
    ordered_labels = list(raw_train.class_names)

    raw_train_eval = _load_directory_dataset(train_dir, image_size, shuffle=False)
    raw_val = _load_directory_dataset(val_dir, image_size, shuffle=False)
    raw_test = _load_directory_dataset(test_dir, image_size, shuffle=False)

    train_base = raw_train.map(_cast_to_uint8, num_parallel_calls=AUTOTUNE)
    val_base = raw_val.map(_cast_to_uint8, num_parallel_calls=AUTOTUNE)
    test_base = raw_test.map(_cast_to_uint8, num_parallel_calls=AUTOTUNE)

    if cache_in_ram:
        train_base = train_base.cache()
        val_base = val_base.cache()

    train_ds = (
        train_base.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)
        .map(lambda x, y: _augment_and_preprocess(x, y, image_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    train_eval_ds = (
        raw_train_eval.map(_preprocess_mobilenet, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_base.map(_preprocess_mobilenet, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_base.map(_preprocess_mobilenet, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, train_eval_ds, val_ds, test_ds, ordered_labels


def extract_dataset_targets(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect integer class targets in deterministic order from an unshuffled one-hot dataset."""
    return np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in dataset], axis=0)


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build Sequential MobileNetV2 model with BatchNorm head and named GAP embedding layer."""
    base_model = MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="head_conv"),
            layers.BatchNormalization(name="head_bn"),
            layers.ReLU(name="head_relu"),
            layers.MaxPooling2D((2, 2), name="head_pool"),
            layers.GlobalAveragePooling2D(name=EMBEDDING_LAYER_NAME),
            layers.Dense(256, activation="relu", name="head_dense"),
            layers.Dropout(0.4, name="head_dropout"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_backbone_layers(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_layers: int = 40,
    learning_rate: float = 3e-5,
    label_smoothing: float = 0.1,
) -> None:
    """Unfreeze top layers of MobileNetV2 while keeping BatchNormalization layers frozen for stability."""
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - fine_tune_layers)

    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def _merge_histories(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for hist in histories:
        if hist is None:
            continue
        data = hist.history if hasattr(hist, "history") else hist
        for key, values in data.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def save_training_plots(history_dict: dict[str, list[float]], output_path: Path) -> None:
    """Save training and validation accuracy/loss curves to output_path."""
    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
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


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _build_embedding_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Create a view over model layers up to EMBEDDING_LAYER_NAME (compatible with Keras 2 & 3)."""
    sub_layers = []
    for layer in model.layers:
        sub_layers.append(layer)
        if layer.name == EMBEDDING_LAYER_NAME:
            break
    if not sub_layers or sub_layers[-1].name != EMBEDDING_LAYER_NAME:
        raise ValueError(f"Embedding layer '{EMBEDDING_LAYER_NAME}' not found in model.")
    return models.Sequential(sub_layers)


def calibrate_ood_gate(
    model: tf.keras.Model,
    train_eval_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    ordered_labels: list[str],
    percentile: float = 5.0,
) -> dict[str, object]:
    """Compute per-class GAP embedding centroids and validation cosine similarity threshold."""
    extractor = _build_embedding_extractor(model)

    train_embs = _l2_normalize(extractor.predict(train_eval_ds, verbose=0))
    train_targets = extract_dataset_targets(train_eval_ds)

    centroids: dict[str, list[float]] = {}
    for idx, label in enumerate(ordered_labels):
        if label == OTHER_CLASS_NAME:
            continue
        cls_embs = train_embs[train_targets == idx]
        if len(cls_embs) == 0:
            continue
        centroid = _l2_normalize(cls_embs.mean(axis=0, keepdims=True))[0]
        centroids[label] = [round(float(v), 6) for v in centroid]

    val_embs = _l2_normalize(extractor.predict(val_ds, verbose=0))
    val_targets = extract_dataset_targets(val_ds)

    in_dist_sims = []
    for emb, target_idx in zip(val_embs, val_targets):
        label = ordered_labels[int(target_idx)]
        if label not in centroids:
            continue
        centroid_vec = np.asarray(centroids[label], dtype=np.float32)
        in_dist_sims.append(float(np.dot(emb, centroid_vec)))

    raw_threshold = float(np.percentile(in_dist_sims, percentile)) if in_dist_sims else 0.65
    cosine_threshold = round(float(np.clip(raw_threshold, 0.55, 0.82)), 4)

    return {
        "enabled": True,
        "embedding_layer": EMBEDDING_LAYER_NAME,
        "min_cosine_similarity": cosine_threshold,
        "min_confidence": 0.60,
        "max_entropy": 1.15,
        "class_centroids": centroids,
    }


def _clean_tfjs_initializer(obj: object) -> object:
    if isinstance(obj, dict) and "class_name" in obj and "config" in obj:
        return {"class_name": obj["class_name"], "config": obj["config"]}
    return obj


def _clean_tfjs_layer(layer: dict) -> dict:
    cname = layer.get("class_name")
    cfg = dict(layer.get("config", {}))

    if cname == "InputLayer":
        cfg.pop("optional", None)
    elif cname == "DepthwiseConv2D":
        cfg.pop("groups", None)
    elif cname == "Conv2D" and cfg.get("groups") == 1:
        cfg.pop("groups", None)

    for key in list(cfg.keys()):
        if key.endswith(("_initializer", "_regularizer", "_constraint")) and isinstance(cfg[key], dict):
            cfg[key] = _clean_tfjs_initializer(cfg[key])

    if cname in ("Functional", "Model", "Sequential") and "layers" in cfg:
        cfg["layers"] = [_clean_tfjs_layer(sub) for sub in cfg["layers"]]

    cleaned = dict(layer)
    cleaned["config"] = cfg
    return cleaned


def _export_tfjs_manual(model: tf.keras.Model, tfjs_dir: Path) -> None:
    """Serialize Keras Sequential model into browser-ready TFJS layers-model format."""
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

    raw_topology = json.loads(model.to_json())
    model_config = _clean_tfjs_layer(
        {
            "class_name": raw_topology["class_name"],
            "config": raw_topology["config"],
        }
    )

    tfjs_manifest = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.__version__}",
        "convertedBy": "TensorFlow.js LayersModel Normalizer",
        "modelTopology": {
            "keras_version": raw_topology.get("keras_version", tf.__version__),
            "backend": raw_topology.get("backend", "tensorflow"),
            "model_config": model_config,
        },
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
    ordered_labels: list[str],
    models_dir: Path,
    ood_config: dict[str, object] | None = None,
) -> None:
    """Export SavedModel, TFLite (+ label.txt), TFJS, and shared model_meta.json."""
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

    label_path = tflite_dir / "label.txt"
    label_path.write_text("\n".join(ordered_labels) + "\n", encoding="utf-8")
    print(f"TFLite model exported to: {tflite_path}")
    print(f"Labels saved to: {label_path}")

    try:
        import tensorflowjs as tfjs

        tfjs.converters.save_keras_model(model, str(tfjs_dir))
        print(f"TFJS model exported via tensorflowjs to: {tfjs_dir}")
    except Exception as exc:
        print(f"tensorflowjs converter unavailable ({exc}); using normalized layers-model exporter.")
        _export_tfjs_manual(model, tfjs_dir)
        print(f"TFJS model exported to: {tfjs_dir}")

    meta = {
        "version": "v2",
        "preprocessing": "mobilenet_v2",
        "crop": "center_square",
        "input_size": list(IMAGE_SIZE),
        "labels": ordered_labels,
        "other_class": OTHER_CLASS_NAME if OTHER_CLASS_NAME in ordered_labels else None,
        "ood": ood_config or {"enabled": False},
    }
    meta_path = models_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model metadata & OOD config saved to: {meta_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train (2-stage MobileNetV2 fine-tuning with tf.data cache), calibrate OOD gate, evaluate, and export."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset_split"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=3e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--target-acc", type=float, default=0.96)
    parser.add_argument("--no-cache", action="store_true", help="Disable uint8 RAM caching of train/val datasets.")
    parser.add_argument("--plot-path", type=Path, default=Path("training_history.png"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")
    if not gpus:
        print("[WARNING] No GPU detected. Training on CPU will be significantly slower.")

    train_ds, train_eval_ds, val_ds, test_ds, ordered_labels = create_datasets(
        split_dir=args.data_dir,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        cache_in_ram=not args.no_cache,
    )
    print(f"Classes ({len(ordered_labels)}): {ordered_labels}")

    model, base_model = build_model(
        num_classes=len(ordered_labels),
        image_size=IMAGE_SIZE,
        learning_rate=args.stage1_lr,
        label_smoothing=args.label_smoothing,
    )
    model.summary()

    ckpt_path = args.models_dir / "_best_weights.weights.h5"
    args.models_dir.mkdir(parents=True, exist_ok=True)

    stage1_callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    print("\n--- Stage 1: Head Warmup (MobileNetV2 frozen) ---")
    hist1 = model.fit(
        train_ds,
        epochs=args.stage1_epochs,
        validation_data=val_ds,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    print(f"\n--- Stage 2: Fine-Tuning top {args.fine_tune_layers} MobileNetV2 layers ---")
    unfreeze_top_backbone_layers(
        model=model,
        base_model=base_model,
        fine_tune_layers=args.fine_tune_layers,
        learning_rate=args.stage2_lr,
        label_smoothing=args.label_smoothing,
    )

    stage2_callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        ThresholdCallback(target_acc=args.target_acc),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    hist2 = model.fit(
        train_ds,
        epochs=args.stage2_epochs,
        validation_data=val_ds,
        callbacks=stage2_callbacks,
        verbose=1,
    )

    if ckpt_path.is_file():
        model.load_weights(str(ckpt_path))
        ckpt_path.unlink(missing_ok=True)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    merged_history = _merge_histories(hist1, hist2)
    save_training_plots(merged_history, args.plot_path)
    print(f"Training plot saved to: {args.plot_path}")

    print("\n--- Calibrating OOD Embedding Gate ---")
    ood_config = calibrate_ood_gate(model, train_eval_ds, val_ds, ordered_labels)
    print(f"OOD cosine similarity threshold: {ood_config['min_cosine_similarity']}")

    export_models(model, ordered_labels, args.models_dir, ood_config=ood_config)


if __name__ == "__main__":
    main()
