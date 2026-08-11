Dataset preparation notes

This folder contains helper scripts to gather and convert public UAV/drones datasets into a YOLOv8 training folder.

Workflow

1. Inspect `download_datasets.sh` and decide which datasets to fetch.
2. Run the download helper (it will create `data/raw/<Dataset>` folders):

   ```bash
   bash scripts/download_datasets.sh visdrone --sample
   ```

3. Place any full archives (if you manually downloaded them) into `data/raw/VisDrone` or `data/raw/DOTA` and extract.
4. Convert annotations into YOLO format using `scripts/convert_to_yolo.py`.

   Example:

   ```bash
   python3 scripts/convert_to_yolo.py --voc data/raw/VisDrone/Annotations --images data/raw/VisDrone/JPEGImages --out yolo_data/labels
   ```

5. Merge images in `yolo_data/images` and labels in `yolo_data/labels` and update `yolo_data/data.yaml` before training.

Notes and caveats

- VisDrone and DOTA are large (multiple GB). Downloading them in this container may exceed limits and take long.
- DOTA uses rotated boxes; converting all DOTA annotations to axis-aligned YOLO boxes requires custom logic and potential label changes.
- Human validation of labels is strongly recommended before training.
