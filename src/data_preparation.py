"""Dataset download, hard-negative 'Other' assembly, resolution audit, and 80:10:10 split."""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import kagglehub
from PIL import Image

FAST_FOOD_HANDLE = "utkarshsaxenadn/fast-food-classification-dataset"
FAST_FOOD_SUBFOLDER = "Fast Food Classification V2"
FOOD11_HANDLE = "trolukovich/food11-image-dataset"

SOURCE_SPLITS = ("Train", "Valid", "Test")
FAST_FOOD_CLASSES = (
    "Baked Potato",
    "Burger",
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Pizza",
)
# Unused classes inside Fast Food Classification V2 serve as near-OOD hard negatives.
UNUSED_FAST_FOOD_CLASSES = ("Hot Dog", "Sandwich", "Taco", "Taquito")
# Exclude Food-11 'Fried food' and 'Bread' to avoid label overlap with Crispy Chicken, Fries, Burger, and Donut.
FOOD11_OTHER_QUOTAS = {
    "Seafood": 400,
    "Meat": 200,
    "Rice": 200,
    "Soup": 150,
    "Noodles-Pasta": 150,
    "Vegetable-Fruit": 100,
}
OTHER_CLASS_NAME = "Other"


def download_fast_food_dataset(dataset_handle: str = FAST_FOOD_HANDLE) -> Path:
    """Download Fast Food Classification V2 via kagglehub."""
    downloaded_path = Path(kagglehub.dataset_download(dataset_handle))
    return downloaded_path / FAST_FOOD_SUBFOLDER


def download_food11_dataset(dataset_handle: str = FOOD11_HANDLE) -> Path:
    """Download Food-11 image dataset via kagglehub for non-fast-food hard negatives."""
    return Path(kagglehub.dataset_download(dataset_handle))


def _is_valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _gather_images_by_folder_name(root: Path, target_folder_name: str) -> list[Path]:
    """Recursively collect files from any subfolder matching target_folder_name (case-insensitive)."""
    target_lower = target_folder_name.lower()
    matches: list[Path] = []
    for subdir in sorted(root.rglob("*")):
        if subdir.is_dir() and subdir.name.lower() == target_lower:
            matches.extend(sorted(p for p in subdir.iterdir() if p.is_file()))
    return matches


def collect_raw_images(
    source_base: Path,
    raw_dir: Path,
    selected_classes: tuple[str, ...] = FAST_FOOD_CLASSES,
    include_other: bool = True,
    food11_root: Path | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Assemble target classes plus an optional balanced 'Other' hard-negative class."""
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
                if not _is_valid_image(src_file):
                    continue
                dst_file = cls_target_dir / f"{split_name}_{counter}{src_file.suffix}"
                shutil.copy2(src_file, dst_file)
                counter += 1

        counts[cls] = counter

    if include_other:
        counts[OTHER_CLASS_NAME] = _collect_other_class(
            source_base=source_base,
            target_dir=raw_dir / OTHER_CLASS_NAME,
            food11_root=food11_root,
            seed=seed,
        )

    return counts


def _collect_other_class(
    source_base: Path,
    target_dir: Path,
    food11_root: Path | None,
    seed: int,
) -> int:
    """Build a balanced 'Other' class from unused Fast Food V2 classes and optional Food-11 categories."""
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counter = 0

    ff_quota_per_class = 300 if food11_root is not None else 500
    for cls in UNUSED_FAST_FOOD_CLASSES:
        candidates: list[Path] = []
        for split_name in SOURCE_SPLITS:
            src_cls_dir = source_base / split_name / cls
            if src_cls_dir.is_dir():
                candidates.extend(sorted(p for p in src_cls_dir.iterdir() if p.is_file()))

        rng.shuffle(candidates)
        copied = 0
        for src_file in candidates:
            if copied >= ff_quota_per_class:
                break
            if not _is_valid_image(src_file):
                continue
            slug = cls.lower().replace(" ", "_")
            dst_file = target_dir / f"ff_{slug}_{counter}{src_file.suffix}"
            shutil.copy2(src_file, dst_file)
            copied += 1
            counter += 1

    if food11_root is not None and food11_root.exists():
        for category, quota in FOOD11_OTHER_QUOTAS.items():
            candidates = _gather_images_by_folder_name(food11_root, category)
            rng.shuffle(candidates)
            copied = 0
            for src_file in candidates:
                if copied >= quota:
                    break
                if not _is_valid_image(src_file):
                    continue
                slug = category.lower().replace("-", "_").replace(" ", "_")
                dst_file = target_dir / f"f11_{slug}_{counter}{src_file.suffix}"
                shutil.copy2(src_file, dst_file)
                copied += 1
                counter += 1

    return counter


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
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Split all class folders inside raw_dir into stratified train, val, and test directories."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    class_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found inside: {raw_dir}")

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

    for cls_src in class_dirs:
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
            target_cls_dir = split_dirs[split_name] / cls_src.name
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
        description="Download, assemble classes (+ optional Other hard negatives), verify resolutions, and split."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Path to local 'Fast Food Classification V2' folder. If omitted, downloads via kagglehub.",
    )
    parser.add_argument(
        "--food11-dir",
        type=Path,
        default=None,
        help="Path to local Food-11 folder. If omitted and --include-food11 is set, downloads via kagglehub.",
    )
    parser.add_argument(
        "--no-other",
        action="store_true",
        help="Train on the original 6 fast food classes only (disable 7th 'Other' class).",
    )
    parser.add_argument(
        "--no-food11",
        action="store_true",
        help="Build 'Other' from unused Fast Food V2 classes only without downloading Food-11.",
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
    source_base = args.source_dir if args.source_dir is not None else download_fast_food_dataset()
    print(f"Using Fast Food dataset: {source_base}")

    include_other = not args.no_other
    food11_root: Path | None = None
    if include_other and not args.no_food11:
        food11_root = args.food11_dir if args.food11_dir is not None else download_food11_dataset()
        print(f"Using Food-11 hard negatives: {food11_root}")

    collect_raw_images(
        source_base=source_base,
        raw_dir=args.raw_dir,
        include_other=include_other,
        food11_root=food11_root,
        seed=args.seed,
    )
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
