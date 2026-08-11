# UAV Detector - Usage

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open http://localhost:8080 in your browser (or the forwarded preview port), upload an image, and use the feedback buttons to label results for reinforcement.

If 8080 is already in use, the app automatically picks the next free port.

Notes:
- Uploaded images are stored in `uploads/` and processed images in `processed/`.
- A SQLite DB `data.db` stores metadata and feedback for reinforcement learning.

Dataset preparation and training (optional):

- Prepare COCO-format annotations and images, then run:

```bash
python data_prep.py --images-dir /path/to/images --ann-file /path/to/instances.json --out-dir data
```

- Train detector (classifier + regressor):
 - Train detector (classifier + regressor):

```bash
python train_detector.py
```

This will write models into the `models/` directory which the app will use automatically.

YOLOv8 transfer-learning (recommended for robust detection):

1. Prepare dataset from uploads (weak labels are extracted if available):

```bash
python train_from_uploads.py
```

2. Install ultralytics (and torch) on a machine with GPU for reasonable speed:

```bash
pip install ultralytics
```

3. Train YOLOv8 with the generated `yolo_data/data.yaml`:

```bash
python yolo_train.py --epochs 30 --imgsz 640 --batch 8
```

4. After training, copy the best weights into `models/yolov8_best.pt` so the app will use them for inference.

Winged-UAV dataset sources (free and popular):
- DOTA-v2.0: https://captain-whu.github.io/DOTA/dataset.html
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- UAVDT: https://sites.google.com/site/dstunofficial/
- AU-AIR: https://github.com/auair/auair

Labeling policy for this project:
- Keep only winged UAV silhouettes: delta-wing and rectangular-wing shapes.
- Ignore rotorcraft / propeller drones.
- Use a binary or class label of winged_uav versus background to reduce confusion.

Notes:
- Training YOLOv8 is resource intensive; use a GPU-enabled environment (Colab, AWS/GCP, or local GPU).
- Convert public aerial-object datasets to YOLO format and place them under `yolo_data` before training.
