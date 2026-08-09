import os
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'yolo_data' / 'data.yaml'
MODELS = BASE / 'models'
MODELS.mkdir(exist_ok=True)


def train(epochs=20, imgsz=640, batch=8, model='yolov8n.pt'):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics not installed. Install with `pip install ultralytics`')
        raise

    if not DATA.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA}. Run train_from_uploads.py first.")

    yolo = YOLO(model)
    print('Starting training with', model)
    yolo.train(data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, name='uav_transfer')
    # trained weights are saved into runs/detect/uav_transfer
    print('Training finished. Check runs/detect/uav_transfer for weights and logs.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--model', type=str, default='yolov8n.pt')
    args = p.parse_args()
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model=args.model)
