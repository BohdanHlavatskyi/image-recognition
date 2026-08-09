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

3. Open http://localhost:5000 in your browser, upload an image, and use the feedback buttons to label results for reinforcement.

Notes:
- Uploaded images are stored in `uploads/` and processed images in `processed/`.
- A SQLite DB `data.db` stores metadata and feedback for reinforcement learning.
