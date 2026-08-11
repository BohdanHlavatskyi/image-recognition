#!/usr/bin/env python3
"""
Convert common UAV dataset annotation formats into YOLOv8 label format.
Supported inputs (best-effort):
 - Pascal VOC XML (object bbox in <bndbox>)
 - Simple CSV/TSV with columns: filename,xmin,ymin,xmax,ymax,class
 - Plain bbox txt with x1 y1 x2 y2 class per line

This script writes labels to a destination folder mirroring the image structure.

Note: DOTA contains rotated boxes and other formats — full DOTA conversion requires specialized parsing and is not implemented here.
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
from PIL import Image

CLASS_MAP = {
    # Map dataset class names to numeric class ids for YOLO; adjust as needed.
    'drone': 0,
    'uav': 0,
    'airplane': 1,
}


def voc_xml_to_yolo(xml_path: Path, img_size: Tuple[int,int], class_map: dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        cls = class_map.get(name, None)
        if cls is None:
            continue
        bnd = obj.find('bndbox')
        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)
        x = (xmin + xmax) / 2.0 / img_size[0]
        y = (ymin + ymax) / 2.0 / img_size[1]
        w = (xmax - xmin) / img_size[0]
        h = (ymax - ymin) / img_size[1]
        labels.append((cls, x, y, w, h))
    return labels


def write_yolo_label(label_path: Path, labels: List[Tuple]):
    with open(label_path, 'w') as f:
        for cls,x,y,w,h in labels:
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def convert_voc_dir(voc_dir: Path, images_dir: Path, out_labels_dir: Path, class_map: dict):
    os.makedirs(out_labels_dir, exist_ok=True)
    xml_files = list(voc_dir.glob('*.xml'))
    if not xml_files:
        print('No VOC XML files found in', voc_dir)
        return
    for xml in xml_files:
        stem = xml.stem
        img_path = None
        for ext in ['.jpg','.png','.jpeg']:
            cand = images_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print('Image for', xml, 'not found; skipping')
            continue
        with Image.open(img_path) as im:
            w,h = im.size
        labels = voc_xml_to_yolo(xml, (w,h), class_map)
        out_label = out_labels_dir / (stem + '.txt')
        write_yolo_label(out_label, labels)


def convert_simple_bbox_txt(txt_path: Path, images_dir: Path, out_labels_dir: Path, class_map: dict):
    # Expect lines: filename x1 y1 x2 y2 class_name
    os.makedirs(out_labels_dir, exist_ok=True)
    with open(txt_path,'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            fname = parts[0]
            x1,y1,x2,y2 = map(float, parts[1:5])
            cls_name = parts[5]
            cls = class_map.get(cls_name, None)
            if cls is None:
                continue
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                w,h = im.size
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            out_label = out_labels_dir / (Path(fname).stem + '.txt')
            with open(out_label,'a') as ol:
                ol.write(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--voc', help='Directory with VOC XML files')
    p.add_argument('--bbox-txt', help='Simple bbox txt file (filename x1 y1 x2 y2 class)')
    p.add_argument('--images', required=True, help='Images directory')
    p.add_argument('--out', required=True, help='Output labels directory (YOLO .txt)')
    args = p.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)

    if args.voc:
        convert_voc_dir(Path(args.voc), images_dir, out_dir, CLASS_MAP)
    if args.bbox_txt:
        convert_simple_bbox_txt(Path(args.bbox_txt), images_dir, out_dir, CLASS_MAP)
    print('Conversion finished. Check', out_dir)


if __name__ == '__main__':
    main()
