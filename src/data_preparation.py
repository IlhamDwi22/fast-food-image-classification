"""Dataset download, resolution inspection, and 80:10:10 stratified splitting."""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import kagglehub
from PIL import Image

DATASET_HANDLE = "utkarshsaxenadn/fast-food-classification-dataset"
DATASET_SUBFOLDER = "Fast Food Classification V2"
SOURCE_SPLITS = ("Train", "Valid", "Test")
SELECTED_CLASSES = (
    "Baked Potato",
    "Burger",
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Pizza",
)


def download_dataset(dataset_handle: str = DATASET_HANDLE) -> Path:
    """Download the Fast Food Classification dataset via kagglehub."""
    downloaded_path = Path(kagglehub.dataset_download(dataset_handle))
    return downloaded_path / DATASET_SUBFOLDER


def collect_raw_images(
    source_base: Path,
    raw_dir: Path,
    selected_classes: tuple[str, ...] = SELECTED_CLASSES,
) -> dict[str, int]:
    """Merge selected classes from source splits into raw_dir, skipping corrupt files."""
    if not source_base.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_base}")

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for cls in selected_classes:
        cls_target_dir = raw_dir / cls
        cls_target_dir.mkdir(parents=True, exist_ok=True)
        counter = 0

        for split_name in SOURCE_SPLITS:
            src_cls_dir = source_base / split_name / cls
            if not src_cls_dir.is_dir():
                continue

            for src_file in sorted(src_cls_dir.iterdir()):
                if not src_file.is_file():
                    continue
                try:
                    with Image.open(src_file) as img:
                        img.verify()
                    dst_file = cls_target_dir / f"{split_name}_{counter}{src_file.suffix}"
                    shutil.copy2(src_file, dst_file)
                    counter += 1
                except Exception:
                    continue

        counts[cls] = counter
    return counts


def inspect_image_resolutions(directory: Path) -> dict[str, object]:
    """Count images and unique (width, height) resolutions per class and overall."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    overall_sizes: set[tuple[int, int]] = set()
    total_images = 0
    per_class: dict[str, dict[str, object]] = {}

    for subdir in sorted(p for p in directory.iterdir() if p.is_dir()):
        image_files = [p for p in subdir.iterdir() if p.is_file()]
        class_sizes: set[tuple[int, int]] = set()

        for img_path in image_files:
            with Image.open(img_path) as img:
                overall_sizes.add(img.size)
                class_sizes.add(img.size)

        num_images = len(image_files)
        total_images += num_images
        sample_sizes = sorted(class_sizes)[:5]
        per_class[subdir.name] = {
            "count": num_images,
            "unique_resolutions": len(class_sizes),
            "sample_resolutions": sample_sizes,
        }

        print(f"{subdir.name}: {num_images} images ({len(class_sizes)} unique resolutions)")
        print(f"  Sample resolutions: {sample_sizes}")

    print(f"\nTotal images: {total_images}")
    print(f"Total unique resolutions across dataset: {len(overall_sizes)}")
    return {
        "total_images": total_images,
        "total_unique_resolutions": len(overall_sizes),
        "per_class": per_class,
    }


def split_dataset(
    raw_dir: Path,
    split_dir: Path,
    selected_classes: tuple[str, ...] = SELECTED_CLASSES,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Split raw_dir into train, val, and test directories with per-class stratification."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    if split_dir.exists():
        shutil.rmtree(split_dir)

    split_dirs = {
        "train": split_dir / "train",
        "val": split_dir / "val",
        "test": split_dir / "test",
    }
    for path in split_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for cls in selected_classes:
        cls_src = raw_dir / cls
        if not cls_src.is_dir():
            raise FileNotFoundError(f"Expected class directory missing: {cls_src}")

        images = sorted(p for p in cls_src.iterdir() if p.is_file())
        rng.shuffle(images)

        total_cls = len(images)
        train_end = int(total_cls * train_ratio)
        val_end = train_end + int(total_cls * val_ratio)

        partitions = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, files in partitions.items():
            target_cls_dir = split_dirs[split_name] / cls
            target_cls_dir.mkdir(parents=True, exist_ok=True)
            for src_file in files:
                shutil.copy2(src_file, target_cls_dir / src_file.name)

    summary = {
        name: sum(1 for _ in path.rglob("*") if _.is_file())
        for name, path in split_dirs.items()
    }
    print("\nDataset split summary:")
    for split_name, count in summary.items():
        print(f"  {split_name}: {count} images")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, verify resolutions, and split the Fast Food Classification dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Path to local 'Fast Food Classification V2' folder. If omitted, downloads via kagglehub.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_dataset"),
        help="Output directory for merged raw class folders (default: data/raw_dataset).",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/dataset_split"),
        help="Output directory for train/val/test splits (default: data/dataset_split).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_base = args.source_dir if args.source_dir is not None else download_dataset()
    print(f"Using source dataset: {source_base}")

    collect_raw_images(source_base, args.raw_dir)
    inspect_image_resolutions(args.raw_dir)
    split_dataset(
        raw_dir=args.raw_dir,
        split_dir=args.split_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
