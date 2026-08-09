import os
import shutil
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / 'uploads'
DATA_DIR = BASE / 'yolo_data'
MODELS_DIR = BASE / 'models'
DB = BASE / 'data.db'


def ensure_dirs():
    for p in [DATA_DIR / 'images' / 'train', DATA_DIR / 'images' / 'val', DATA_DIR / 'labels' / 'train', DATA_DIR / 'labels' / 'val']:
        p.mkdir(parents=True, exist_ok=True)


def bbox_from_shapes(shapes_json):
    try:
        shapes = json.loads(shapes_json)
        if not shapes:
            return None
        # take first candidate with bbox
        for s in shapes:
            if 'bbox' in s and s['bbox']:
                x, y, w, h = s['bbox']
                return int(x), int(y), int(w), int(h)
    except Exception:
        return None
    return None


def collect_from_db(limit=None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT id, filename, processed_path, detected, shapes_json, center_x, center_y, feedback FROM uploads')
    rows = c.fetchall()
    conn.close()
    return rows


def convert_to_yolo(x, y, w, h, img_w, img_h):
    # YOLO normalized center x,y and width,height
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def build_dataset(train_ratio=0.8):
    ensure_dirs()
    rows = collect_from_db()
    positives = []
    negatives = []
    for r in rows:
        uid, fname, processed, detected, shapes_json, cx, cy, fb = r
        src = UPLOADS / fname
        if not src.exists():
            continue
        # try to get bbox from shapes_json
        bb = bbox_from_shapes(shapes_json)
        if fb == 1 or (detected and bb is not None):
            positives.append((src, bb))
        else:
            negatives.append((src, None))

    # simple split
    import random
    random.shuffle(positives)
    split = int(len(positives) * train_ratio)
    train_pos = positives[:split]
    val_pos = positives[split:]

    # move/copy files and write labels
    def write_pair(dst_images_dir, dst_labels_dir, items):
        for src, bb in items:
            dst_img = dst_images_dir / src.name
            shutil.copy(src, dst_img)
            # try to read size
            from PIL import Image
            img = Image.open(src)
            iw, ih = img.size
            if bb:
                x, y, w, h = bb
            else:
                # use full image as weak label for negatives (class 0 will be skipped)
                x, y, w, h = 0, 0, iw, ih
            cx, cy, nw, nh = convert_to_yolo(x, y, w, h, iw, ih)
            label_file = dst_labels_dir / (src.stem + '.txt')
            # single class '0' for UAV
            with open(label_file, 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    write_pair(DATA_DIR / 'images' / 'train', DATA_DIR / 'labels' / 'train', train_pos)
    write_pair(DATA_DIR / 'images' / 'val', DATA_DIR / 'labels' / 'val', val_pos)

    # data yaml
    data_yaml = {
        'path': str(DATA_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': ['uav']
    }
    import yaml
    with open(DATA_DIR / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    print('Prepared YOLO dataset at', DATA_DIR)


if __name__ == '__main__':
    build_dataset()
