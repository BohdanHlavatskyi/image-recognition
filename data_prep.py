import os
import json
import argparse
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

"""
Prepare dataset from COCO-format annotations (or similar) into HOG features
and center regression targets. Produces numpy arrays saved to disk for training.

Usage:
  python data_prep.py --images-dir /path/to/images --ann-file instances.json --out-dir data

This script extracts positive samples from annotated UAV boxes and negative
samples by sampling random crops without UAVs. It saves `X_class.npy`,
`y_class.npy`, `X_reg.npy`, `y_reg.npy` into the `out-dir`.

Note: This is a simple preparer to bootstrap a lightweight detector. For
large-scale training, use an object-detection framework (Detectron2, YOLO, etc.).
"""


def extract_hog_img(img, size=128):
    if img.ndim == 3:
        img = rgb2gray(img)
    img = resize(img, (size, size), anti_aliasing=True)
    feat = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return feat


def load_coco_annotations(ann_file):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    images = {img['id']: img for img in data.get('images', [])}
    anns = {}
    for a in data.get('annotations', []):
        img_id = a['image_id']
        anns.setdefault(img_id, []).append(a)
    return images, anns


def main(images_dir, ann_file, out_dir, neg_per_image=2):
    os.makedirs(out_dir, exist_ok=True)
    images, anns = load_coco_annotations(ann_file)

    Xc = []
    yc = []
    Xr = []
    yr = []

    for img_id, img_meta in images.items():
        fname = img_meta.get('file_name')
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        img_anns = anns.get(img_id, [])

        # positive samples
        for a in img_anns:
            x, y, bw, bh = a['bbox']
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(w, x + bw))
            y2 = int(min(h, y + bh))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            feat = extract_hog_img(crop)
            Xc.append(feat)
            yc.append(1)
            # center regression target: normalized offsets from bbox top-left
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            # normalized to bbox size
            dx = (cx - x) / bw
            dy = (cy - y) / bh
            Xr.append(feat)
            yr.append((dx, dy))

        # negative samples: random crops without annotations
        for k in range(neg_per_image):
            import random
            bw = int(min(w, h) * random.uniform(0.05, 0.25))
            bh = bw
            x = int(random.uniform(0, w - bw))
            y = int(random.uniform(0, h - bh))
            # check overlap with any ann
            overlap = False
            for a in img_anns:
                ax, ay, abw, abh = a['bbox']
                if not (x + bw < ax or x > ax + abw or y + bh < ay or y > ay + abh):
                    overlap = True
                    break
            if overlap:
                continue
            crop = img[y:y + bh, x:x + bw]
            feat = extract_hog_img(crop)
            Xc.append(feat)
            yc.append(0)

    import numpy as _np
    Xc = _np.array(Xc)
    yc = _np.array(yc)
    Xr = _np.array(Xr)
    yr = _np.array(yr)
    _np.save(os.path.join(out_dir, 'X_class.npy'), Xc)
    _np.save(os.path.join(out_dir, 'y_class.npy'), yc)
    _np.save(os.path.join(out_dir, 'X_reg.npy'), Xr)
    _np.save(os.path.join(out_dir, 'y_reg.npy'), yr)
    print('Saved data to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out-dir', default='data')
    args = parser.parse_args()
    main(args.images_dir, args.ann_file, args.out_dir)
