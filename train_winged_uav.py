#!/usr/bin/env python3
"""Lightweight winged-UAV trainer.

This script is intentionally small and CPU-friendly. It uses HOG features
with a RandomForest classifier, which is much lighter than a deep network and
works well for small training sets or filtered public datasets.

Typical flow:
1. Download a small subset or filtered annotations from public UAV datasets
   (VisDrone, DOTA-v2.0, UAVDT, AU-AIR).
2. Keep only winged-UAV labels (delta / rectangular wing shapes).
3. Run:
      python train_winged_uav.py --dataset-root /path/to/dataset --model-out models/winged_uav_model.pkl

This project intentionally avoids multi-GPU or huge repo workflows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from skimage.feature import hog
from skimage.transform import resize

DEFAULT_POSITIVE_LABELS = {
    "uav",
    "drone",
    "winged_uav",
    "winged",
    "aircraft",
    "delta",
    "delta_wing",
    "rectangular",
    "rectangle",
    "rectangular_wing",
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def ensure_rgb(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as img:
        return np.array(img.convert("RGB"))


def hog_feature(image: np.ndarray, size: int = 128) -> np.ndarray:
    gray = color.rgb2gray(image)
    gray = resize(gray, (size, size), anti_aliasing=True)
    feat = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return feat


def parse_yolo_boxes(label_path: Path, class_names: dict[int, str]) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return boxes

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        name = normalize_label(class_names.get(cls_id, str(cls_id)))
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        if name not in DEFAULT_POSITIVE_LABELS:
            continue
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def yolo_box_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return max(0, int(x)), max(0, int(y)), min(img_w, int(x2)), min(img_h, int(y2))


def iter_images(root: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        yield from sorted(root.rglob(ext))


def sample_negative_crops(img_path: Path, positive_boxes: list[tuple[int, int, int, int]], count: int = 4, size: int = 128) -> List[np.ndarray]:
    img = ensure_rgb(img_path)
    h, w = img.shape[:2]
    crops: List[np.ndarray] = []
    if h == 0 or w == 0:
        return crops

    for _ in range(count):
        crop_w = random.randint(max(24, int(w * 0.08)), max(32, int(w * 0.25)))
        crop_h = random.randint(max(24, int(h * 0.08)), max(32, int(h * 0.25)))
        x = random.randint(0, max(0, w - crop_w))
        y = random.randint(0, max(0, h - crop_h))
        x2 = min(w, x + crop_w)
        y2 = min(h, y + crop_h)
        overlap = False
        for bx1, by1, bx2, by2 in positive_boxes:
            if not (x2 < bx1 or x > bx2 or y2 < by1 or y > by2):
                overlap = True
                break
        if overlap:
            continue
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            continue
        crops.append(hog_feature(crop, size=size))
    return crops


def collect_training_data(dataset_root: Path, max_positives: int = 1000, max_negatives: int = 1000, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []

    # Case 1: YOLO dataset layout (images/ + labels/ + data.yaml)
    labels_dir = dataset_root / "labels"
    images_dir = dataset_root / "images"
    if labels_dir.exists() and images_dir.exists():
        class_file = dataset_root / "data.yaml"
        class_names: dict[int, str] = {}
        if class_file.exists():
            with class_file.open("r", encoding="utf-8") as fh:
                data = json.loads(json.dumps(__import__("yaml").safe_load(fh))) if False else None
        # yaml may be absent; use default numeric names when not available.
        for image_path in sorted(iter_images(images_dir)):
            label_path = labels_dir / image_path.with_suffix(".txt").name
            if not label_path.exists():
                continue
            positive_boxes = parse_yolo_boxes(label_path, class_names)
            if not positive_boxes:
                continue
            img = ensure_rgb(image_path)
            h, w = img.shape[:2]
            for _, cx, cy, bw, bh in positive_boxes[:10]:
                x1, y1, x2, y2 = yolo_box_to_xyxy(cx, cy, bw, bh, w, h)
                crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                X.append(hog_feature(crop, size=size))
                y.append(1)
                if len(X) >= max_positives:
                    break
            if len(X) >= max_positives:
                break

        # sample negatives from images without labels or outside positive boxes
        for image_path in sorted(iter_images(images_dir)):
            label_path = labels_dir / image_path.with_suffix(".txt").name
            positive_boxes: list[tuple[int, int, int, int]] = []
            if label_path.exists():
                for _, cx, cy, bw, bh in parse_yolo_boxes(label_path, class_names):
                    x1, y1, x2, y2 = yolo_box_to_xyxy(cx, cy, bw, bh, *ensure_rgb(image_path).shape[:2])
                    positive_boxes.append((x1, y1, x2, y2))
            negs = sample_negative_crops(image_path, positive_boxes, count=2, size=size)
            for neg in negs:
                X.append(neg)
                y.append(0)
                if len(y) >= max_positives + max_negatives:
                    break
            if len(y) >= max_positives + max_negatives:
                break

    # Case 2: COCO-style dataset
    if not X:
        ann_file = dataset_root / "instances.json"
        if ann_file.exists():
            with ann_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            category_by_id = {cat["id"]: normalize_label(cat["name"]) for cat in data.get("categories", [])}
            images = {img["id"]: img for img in data.get("images", [])}
            anns = {}
            for ann in data.get("annotations", []):
                if normalize_label(category_by_id.get(ann.get("category_id"), "")) in DEFAULT_POSITIVE_LABELS:
                    anns.setdefault(ann["image_id"], []).append(ann)
            for image_id, anns_for_image in list(anns.items())[:max_positives]:
                img_meta = images.get(image_id)
                if not img_meta:
                    continue
                image_path = dataset_root / img_meta["file_name"]
                if not image_path.exists():
                    continue
                img = ensure_rgb(image_path)
                for ann in anns_for_image[:8]:
                    x, y, w, h = ann["bbox"]
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(img.shape[1], int(x + w))
                    y2 = min(img.shape[0], int(y + h))
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    X.append(hog_feature(crop, size=size))
                    y.append(1)
                    if len(X) >= max_positives:
                        break
                if len(X) >= max_positives:
                    break

    if not X:
        raise ValueError(
            "No usable training data found. Provide a dataset with YOLO labels or COCO annotations, "
            "and keep only delta / rectangular winged-UAV objects."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


def train_and_save(dataset_root: Path, model_output: Path, max_positives: int = 1000, max_negatives: int = 1000, size: int = 128) -> None:
    X, y = collect_training_data(dataset_root, max_positives=max_positives, max_negatives=max_negatives, size=size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=["background", "winged_uav"]))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_output)
    print(f"Saved light model to {model_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight winged-UAV detector from filtered public datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder containing a filtered winged-UAV dataset")
    parser.add_argument("--model-out", type=Path, default=Path("models/winged_uav_model.pkl"), help="Where to save the trained model")
    parser.add_argument("--max-positives", type=int, default=1000, help="Maximum positive samples to keep for a lightweight model")
    parser.add_argument("--max-negatives", type=int, default=1000, help="Maximum negative background samples")
    parser.add_argument("--size", type=int, default=128, help="HOG image size; larger values are a bit slower but often more stable")
    args = parser.parse_args()
    train_and_save(args.dataset_root, args.model_out, max_positives=args.max_positives, max_negatives=args.max_negatives, size=args.size)


if __name__ == "__main__":
    main()
