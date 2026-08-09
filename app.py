import os
import io
import json
import sqlite3
from datetime import datetime
from uuid import uuid4

from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
DB_PATH = os.path.join(BASE_DIR, 'data.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.secret_key = 'dev'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            filename TEXT,
            processed_path TEXT,
            detected INTEGER,
            shapes_json TEXT,
            center_x REAL,
            center_y REAL,
            feedback INTEGER,
            created_at TEXT
        )'''
    )
    conn.commit()
    conn.close()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def detect_shapes_and_draw(img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError('Could not read image')
    orig = img.copy()
    h, w = img.shape[:2]

    # Simple sky mask heuristic: prefer blue-ish and bright regions in upper image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    # Blue-ish hue range (general) and decent saturation/brightness
    lower_blue = np.array((90, 20, 80))
    upper_blue = np.array((150, 255, 255))
    sky_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Also restrict to upper portion
    mask_top = np.zeros_like(sky_mask)
    mask_top[0:int(h * 0.7), :] = 255
    sky_mask = cv2.bitwise_and(sky_mask, mask_top)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Combine edges with sky mask to boost sky contours
    edges_skied = cv2.bitwise_or(edges, cv2.bitwise_and(edges, sky_mask))

    contours, _ = cv2.findContours(edges_skied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # ignore tiny
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if approx is None or len(approx) == 0:
            continue
        pts = approx.reshape(-1, 2)

        # geometry heuristics
        x, y, cw, ch = cv2.boundingRect(approx)
        rect_area = cw * ch
        if rect_area == 0:
            continue
        extent = float(area) / rect_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Accept triangles and quads (approx triangle/trapezoid)
        if len(pts) in (3, 4):
            M = cv2.moments(approx)
            if M['m00'] == 0:
                cx, cy = int(pts[0][0]), int(pts[0][1])
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

            # scoring: area, solidity (prefer solid shapes), extent, location (upper half favored)
            score = area * (0.8 + 0.4 * solidity) * (0.8 + 0.4 * extent)
            if cy < h * 0.6:
                score *= 1.25
            # further boost if inside sky mask region
            if sky_mask[cy, min(max(cx, 0), w - 1)] > 0:
                score *= 1.3

            candidates.append({'pts': pts.tolist(), 'area': area, 'center': (cx, cy), 'score': score, 'sides': len(pts), 'solidity': solidity, 'extent': extent, 'bbox': [x, y, cw, ch]})

    detected = False
    chosen = None
    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        chosen = candidates[0]
        detected = True

    # Draw all candidate polygons (light) and chosen in bright color
    for cand in candidates:
        pts = np.array(cand['pts'], dtype=np.int32)
        cv2.polylines(orig, [pts], True, (0, 215, 255), 2)

    if chosen:
        pts = np.array(chosen['pts'], dtype=np.int32)
        cv2.polylines(orig, [pts], True, (0, 255, 0), 3)
        cx, cy = chosen['center']
        cv2.circle(orig, (cx, cy), 6, (0, 0, 255), -1)
        cv2.drawMarker(orig, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

    if not detected:
        cv2.putText(orig, 'No UAV-like triangular/trapezoid shapes found', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, orig)

    shapes_json = json.dumps(candidates)
    center = chosen['center'] if chosen else (None, None)
    return detected, shapes_json, center


def image_file_to_data_uri(img_path):
    import base64
    ext = os.path.splitext(img_path)[1].lower().lstrip('.')
    img = cv2.imread(img_path)
    _, buf = cv2.imencode('.' + ext, img)
    b64 = base64.b64encode(buf).decode('ascii')
    return f'data:image/{ext};base64,{b64}'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        fn = secure_filename(file.filename)
        uid = str(uuid4())
        ext = fn.rsplit('.', 1)[1].lower()
        save_name = f"{uid}.{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(save_path)

        processed_name = f"{uid}_processed.{ext}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_name)

        try:
            detected, shapes_json, center = detect_shapes_and_draw(save_path, processed_path)
        except Exception as e:
            flash(f'Processing error: {e}')
            return redirect(url_for('index'))

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO uploads (id, filename, processed_path, detected, shapes_json, center_x, center_y, feedback, created_at) VALUES (?,?,?,?,?,?,?,?,?)',
                  (uid, save_name, processed_name, int(detected), shapes_json, center[0] if center[0] else None, center[1] if center[1] else None, None, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return render_template('index.html', processed_url=url_for('processed_file', filename=processed_name), detected=detected, uid=uid)
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))


@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    # Accept multipart/form-data with field 'image'
    if 'image' not in request.files:
        return {'error': 'no image provided'}, 400
    file = request.files['image']
    if file.filename == '':
        return {'error': 'empty filename'}, 400
    if not allowed_file(file.filename):
        return {'error': 'file type not allowed'}, 400

    fn = secure_filename(file.filename)
    uid = str(uuid4())
    ext = fn.rsplit('.', 1)[1].lower()
    save_name = f"{uid}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(save_path)

    processed_name = f"{uid}_processed.{ext}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_name)

    try:
        detected, shapes_json, center = detect_shapes_and_draw(save_path, processed_path)
    except Exception as e:
        return {'error': f'processing error: {e}'}, 500

    # Store metadata
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO uploads (id, filename, processed_path, detected, shapes_json, center_x, center_y, feedback, created_at) VALUES (?,?,?,?,?,?,?,?,?)',
              (uid, save_name, processed_name, int(detected), shapes_json, center[0] if center[0] else None, center[1] if center[1] else None, None, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    data_uri = image_file_to_data_uri(processed_path)
    return {'id': uid, 'detected': bool(detected), 'center': {'x': center[0], 'y': center[1]}, 'shapes': json.loads(shapes_json), 'image_data_uri': data_uri}


@app.route('/feedback', methods=['POST'])
def feedback():
    uid = request.form.get('uid')
    val = request.form.get('is_uav')
    if not uid or val is None:
        return ('', 400)
    fb = 1 if val == 'yes' else 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE uploads SET feedback = ? WHERE id = ?', (fb, uid))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
