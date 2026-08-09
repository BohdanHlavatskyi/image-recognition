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
import math
import joblib
from skimage.feature import hog
from skimage import color

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
DB_PATH = os.path.join(BASE_DIR, 'data.db')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model.pkl')
CLF_FILE = os.path.join(BASE_DIR, 'models', 'clf.pkl')
REG_FILE = os.path.join(BASE_DIR, 'models', 'reg.pkl')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

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

            # color contrast: compare mean LAB color inside contour vs outside bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            # convert to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mean_in = cv2.mean(lab, mask=mask)[:3]
            # outside: bounding box area excluding mask
            x1 = x
            y1 = y
            x2 = min(w, x + cw)
            y2 = min(h, y + ch)
            roi = lab[y1:y2, x1:x2]
            mask_roi = mask[y1:y2, x1:x2]
            if roi.size == 0:
                mean_out = mean_in
            else:
                inv_mask = cv2.bitwise_not(mask_roi)
                mean_out = cv2.mean(roi, mask=inv_mask)[:3]
            # perceptual distance in LAB
            color_contrast = math.sqrt(sum((mean_in[i] - mean_out[i]) ** 2 for i in range(3)))

            # scoring: area, solidity (prefer solid shapes), extent, color contrast, location
            score = area * (0.8 + 0.4 * solidity) * (0.8 + 0.4 * extent) * (1.0 + color_contrast / 50.0)
            if cy < h * 0.6:
                score *= 1.25
            if sky_mask[cy, min(max(cx, 0), w - 1)] > 0:
                score *= 1.2

            # propeller detection: look for multiple short lines radiating from center
            propeller_score = 0.0
            try:
                pad = int(max(cw, ch) * 0.9)
                sx = max(0, cx - pad)
                sy = max(0, cy - pad)
                ex = min(w, cx + pad)
                ey = min(h, cy + pad)
                window = edges_skied[sy:ey, sx:ex]
                # detect lines
                lines = cv2.HoughLinesP(window, 1, np.pi / 180, threshold=30, minLineLength=8, maxLineGap=6)
                if lines is not None:
                    angles = []
                    for l in lines:
                        x3, y3, x4, y4 = l[0]
                        # transform to image coords
                        mx = (x3 + x4) / 2 + sx
                        my = (y3 + y4) / 2 + sy
                        # distance from center
                        dist = math.hypot(mx - cx, my - cy)
                        if dist > max(cw, ch) * 1.2:
                            continue
                        angle = math.degrees(math.atan2((y4 - y3), (x4 - x3)))
                        angles.append(angle)
                    if angles:
                        # cluster unique angles (quantize)
                        q = set(int(a / 20.0) for a in angles)
                        propeller_score = len(q)
            except Exception:
                propeller_score = 0.0

            candidates.append({'pts': pts.tolist(), 'area': area, 'center': (cx, cy), 'score': score, 'sides': len(pts), 'solidity': solidity, 'extent': extent, 'bbox': [x, y, cw, ch], 'color_contrast': color_contrast, 'propeller_score': propeller_score})

    detected = False
    chosen = None
    if candidates:
        # If we have ML models, compute HOG for each bbox and get probability
        for cand in candidates:
            bx, by, bw, bh = cand['bbox']
            try:
                crop = img[by:by + bh, bx:bx + bw]
                feat = extract_hog_from_image(crop if isinstance(crop, str) else crop)
            except Exception:
                feat = None
            cand['uav_prob'] = 0.0
            cand['refined_center'] = cand['center']
            if feat is not None and CLF is not None:
                try:
                    prob = CLF.predict_proba([feat])[0]
                    # assume positive class at index 1
                    cand['uav_prob'] = float(prob[1]) if len(prob) > 1 else float(prob[0])
                    if REG is not None and cand['uav_prob'] > 0.5:
                        pred = REG.predict([feat])[0]
                        # pred is (dx, dy) normalized in bbox coords
                        dx, dy = float(pred[0]), float(pred[1])
                        rcx = int(bx + dx * bw)
                        rcy = int(by + dy * bh)
                        cand['refined_center'] = (rcx, rcy)
                except Exception:
                    pass

        # combine heuristic score with model probability
        candidates.sort(key=lambda x: (x.get('uav_prob', 0.0) * 2.0 + x['score']), reverse=True)
        chosen = candidates[0]
        detected = True if chosen.get('uav_prob', 0.0) > 0.25 or chosen['score'] > 1000 else False

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
        # classify final shape
        shape_type = 'unknown'
        if chosen.get('propeller_score', 0) >= 3:
            shape_type = 'propeller'
        elif chosen.get('sides') == 3:
            shape_type = 'triangle'
        elif chosen.get('sides') == 4:
            # aspect
            bx, by, bw, bh = chosen.get('bbox', [0, 0, 0, 0])
            ar = float(bw) / bh if bh > 0 else 0
            if ar < 0.6 or ar > 1.7:
                shape_type = 'rectangle'
            else:
                shape_type = 'quadrilateral'
        # draw label
        label = f"{shape_type} ({chosen.get('score'):.0f})"
        cv2.putText(orig, label, (max(5, cx - 20), max(20, cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # If YOLO model exists, run detection and draw boxes/centers as stronger signal
    if YOLO_MODEL is not None:
        try:
            res = YOLO_MODEL.predict(img_path, imgsz=640, conf=0.25, verbose=False)
            for r in res:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for b in boxes:
                    try:
                        xyxy = b.xyxy[0].tolist()
                    except Exception:
                        continue
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 128, 255), 2)
                    ccx = int((x1 + x2) / 2)
                    ccy = int((y1 + y2) / 2)
                    cv2.circle(orig, (ccx, ccy), 5, (0, 0, 255), -1)
                    cv2.putText(orig, f'uav {conf:.2f}', (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception:
            pass

    if not detected:
        cv2.putText(orig, 'No UAV-like triangular/trapezoid shapes found', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, orig)

    shapes_json = json.dumps(candidates)
    center = chosen['center'] if chosen else (None, None)
    return detected, shapes_json, center, chosen


def image_file_to_data_uri(img_path):
    import base64
    ext = os.path.splitext(img_path)[1].lower().lstrip('.')
    img = cv2.imread(img_path)
    _, buf = cv2.imencode('.' + ext, img)
    b64 = base64.b64encode(buf).decode('ascii')
    return f'data:image/{ext};base64,{b64}'

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

MODEL = load_model()
CLF = joblib.load(CLF_FILE) if os.path.exists(CLF_FILE) else None
REG = joblib.load(REG_FILE) if os.path.exists(REG_FILE) else None
YOLO_MODEL = None
try:
    from ultralytics import YOLO
    yolomodel_path = os.path.join(BASE_DIR, 'models', 'yolov8_best.pt')
    if os.path.exists(yolomodel_path):
        try:
            YOLO_MODEL = YOLO(yolomodel_path)
        except Exception:
            YOLO_MODEL = None
except Exception:
    YOLO_MODEL = None

def extract_hog_from_image(path, pixels=128):
    # Accept either a file path or an ndarray image (BGR as from OpenCV)
    if isinstance(path, np.ndarray):
        img = path
    else:
        img = cv2.imread(path)
    if img is None:
        return None
    # convert BGR to RGB and to grayscale
    if img.ndim == 3:
        img = color.rgb2gray(img[:, :, ::-1])
    from skimage.transform import resize
    img = resize(img, (pixels, pixels), anti_aliasing=True)
    feat = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return feat


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
            detected, shapes_json, center, chosen = detect_shapes_and_draw(save_path, processed_path)
        except Exception as e:
            flash(f'Processing error: {e}')
            return redirect(url_for('index'))

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO uploads (id, filename, processed_path, detected, shapes_json, center_x, center_y, feedback, created_at) VALUES (?,?,?,?,?,?,?,?,?)',
                  (uid, save_name, processed_name, int(detected), shapes_json, center[0] if center[0] else None, center[1] if center[1] else None, None, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        detected_type = chosen.get('shape_type') if chosen and 'shape_type' in chosen else None
        # chosen may not have shape_type; compute from chosen heuristic
        if detected and not detected_type:
            if chosen.get('propeller_score', 0) >= 3:
                detected_type = 'propeller'
            elif chosen.get('sides') == 3:
                detected_type = 'triangle'
            elif chosen.get('sides') == 4:
                detected_type = 'rectangle'

        # if we have an ML model, run it and override detected_type with model prediction probability
        if MODEL is not None:
            feat = extract_hog_from_image(processed_path)
            if feat is not None:
                try:
                    pred = MODEL.predict([feat])[0]
                    prob = MODEL.predict_proba([feat])[0]
                    # pred: 1 -> UAV, 0 -> not UAV
                    if pred == 1:
                        detected_type = 'uav_model'
                except Exception:
                    pass
        return render_template('index.html', processed_url=url_for('processed_file', filename=processed_name), detected=detected, uid=uid, detected_type=detected_type)
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))


@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, filename, processed_path, detected, feedback FROM uploads ORDER BY created_at DESC LIMIT 200')
    rows = c.fetchall()
    conn.close()
    uploads = []
    for r in rows:
        uploads.append({'id': r[0], 'filename': r[1], 'processed_path': r[2], 'detected': bool(r[3]), 'feedback': r[4]})
    return render_template('dashboard.html', uploads=uploads)


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
        detected, shapes_json, center, chosen = detect_shapes_and_draw(save_path, processed_path)
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
    detected_type = None
    if chosen:
        if chosen.get('propeller_score', 0) >= 3:
            detected_type = 'propeller'
        elif chosen.get('sides') == 3:
            detected_type = 'triangle'
        elif chosen.get('sides') == 4:
            detected_type = 'rectangle'

    # model prediction if available
    model_pred = None
    if MODEL is not None:
        feat = extract_hog_from_image(processed_path)
        if feat is not None:
            try:
                p = MODEL.predict([feat])[0]
                model_pred = int(p)
            except Exception:
                model_pred = None

    return {'id': uid, 'detected': bool(detected), 'detected_type': detected_type, 'model_pred': model_pred, 'center': {'x': center[0], 'y': center[1]}, 'shapes': json.loads(shapes_json), 'image_data_uri': data_uri}


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
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except OSError:
        # fallback to a different port if 5000 is in use
        alt = port + 1
        app.run(host='0.0.0.0', port=alt, debug=True)
