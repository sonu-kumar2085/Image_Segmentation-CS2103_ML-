from pathlib import Path
import json
import random
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent / "weizmann-horse-database"

RAW_IMAGE_DIR = PROJECT_ROOT / "raw" / "horse"
RAW_MASK_DIR = PROJECT_ROOT / "raw" / "mask"

PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_IMAGE_DIR = PROCESSED_DIR / "images"
PROCESSED_MASK_DIR = PROCESSED_DIR / "masks"
SPLIT_FILE = PROCESSED_DIR / "splits.json"

IMAGE_SIZE = (256, 256)
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# Change to "black" if the horse is black and the
# background is white in your original masks.
FOREGROUND_COLOR = "white"


# Helper: find matched image–mask pairs
def _find_pairs():
    """Return a sorted list of (image_path, mask_path) tuples
    whose filename stems match across the two raw directories."""
    image_files = {p.stem: p for p in RAW_IMAGE_DIR.iterdir() if p.is_file()}
    mask_files = {p.stem: p for p in RAW_MASK_DIR.iterdir() if p.is_file()}
    common_stems = sorted(image_files.keys() & mask_files.keys())
    pairs = [(image_files[s], mask_files[s]) for s in common_stems]
    return pairs


# Preprocessing: resize, normalise, save
def _preprocess_and_save(pairs):
    """Resize images and masks to IMAGE_SIZE, normalise, and save
    into the processed directories as PNG files."""
    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MASK_DIR.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in pairs:
        name = img_path.stem + ".png"

        #  image 
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.BILINEAR)
        img.save(PROCESSED_IMAGE_DIR / name)

        #  mask 
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(IMAGE_SIZE, Image.NEAREST)
        mask_arr = np.array(mask)

        # Binarise: ensure only 0 and 1 (or 0 and 255)
        if FOREGROUND_COLOR == "white":
            # Foreground (horse) has higher pixel values
            binary = (mask_arr > (mask_arr.max() / 2)).astype(np.uint8) * 255
        else:
            # Foreground (horse) has lower pixel values
            binary = (mask_arr < (mask_arr.max() / 2)).astype(np.uint8) * 255

        Image.fromarray(binary).save(PROCESSED_MASK_DIR / name)


# Split: create or load the deterministic split
def _create_split(filenames):
    # Shuffle filenames with a fixed seed and split 80:10:10.
    names = list(filenames)
    random.seed(SEED)
    random.shuffle(names)

    n = len(names)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": sorted(names[:n_train]),
        "val": sorted(names[n_train : n_train + n_val]),
        "test": sorted(names[n_train + n_val :]),
    }

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def _load_split():
    # Load an existing split from disk.
    with open(SPLIT_FILE) as f:
        return json.load(f)


# Public: prepare the full dataset
def prepare_dataset(force=False):
    """End-to-end pipeline: pair → preprocess → split.

    If processed data and splits.json already exist and
    force is False, skip preprocessing and reuse the split.
    Returns the splits dict.
    """
    pairs = _find_pairs()
    print(f"Found {len(pairs)} matched image-mask pairs.\n")

    already_processed = (
        PROCESSED_IMAGE_DIR.exists()
        and any(PROCESSED_IMAGE_DIR.iterdir())
        and SPLIT_FILE.exists()
    )

    if already_processed and not force:
        print("Processed data already exists — skipping preprocessing.")
        splits = _load_split()
    else:
        _preprocess_and_save(pairs)
        filenames = [p.stem + ".png" for _, p in pairs]
        splits = _create_split(filenames)

    print("Dataset preparation complete.")
    print(f"  Training images:   {len(splits['train'])}")
    print(f"  Validation images: {len(splits['val'])}")
    print(f"  Test images:       {len(splits['test'])}")
    print(f"  Split saved to: {SPLIT_FILE}\n")
    return splits


# PyTorch Dataset
class HorseSegmentationDataset(Dataset):
    """PyTorch dataset that loads preprocessed images and masks
    for a given split ('train', 'val', or 'test').

    Usage:
        train_ds = HorseSegmentationDataset("train")
        val_ds   = HorseSegmentationDataset("val")
        test_ds  = HorseSegmentationDataset("test")
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        if not SPLIT_FILE.exists():
            raise FileNotFoundError(
                f"Split file not found at {SPLIT_FILE}.\n"
                "Run  python code/dataset.py  first to preprocess the data."
            )

        with open(SPLIT_FILE) as f:
            all_splits = json.load(f)

        self.filenames = all_splits[split]
        self.split = split

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # Load image → float32 tensor [C, H, W] in [0, 1]
        img = Image.open(PROCESSED_IMAGE_DIR / name).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW

        # Load mask → float32 tensor [1, H, W] with values {0, 1}
        mask = Image.open(PROCESSED_MASK_DIR / name).convert("L")
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)     # HW → 1HW

        return img, mask


# Main: run preprocessing + quick verification
if __name__ == "__main__":
    splits = prepare_dataset()

    # Quick sanity check
    print("Dataset verification:")
    for split_name in ("train", "val", "test"):
        ds = HorseSegmentationDataset(split_name)
        print(f"  {split_name.capitalize():>10}: {len(ds)}")

    # Show tensor shapes and value ranges from one sample
    sample_img, sample_mask = HorseSegmentationDataset("train")[0]
    print(f"  Image shape: {tuple(sample_img.shape)}")
    print(f"  Mask shape:  {tuple(sample_mask.shape)}")
    print(f"  Image range: {sample_img.min().item():.1f} - {sample_img.max().item():.1f}")
    unique_vals = torch.unique(sample_mask).tolist()
    print(f"  Mask values: {unique_vals}")
