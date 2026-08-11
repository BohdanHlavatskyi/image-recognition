import os
import sqlite3
import joblib
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'data.db')
PROCESSED = os.path.join(BASE, 'processed')
MODEL_PATH = os.path.join(BASE, 'models', 'model.pkl')

def extract_hog(path, pixels=128):
    try:
        img = io.imread(path)
    except Exception:
        return None
    if img is None:
        return None
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = resize(img, (pixels, pixels), anti_aliasing=True)
    try:
        feat, _ = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True, feature_vector=True)
    except Exception:
        return None
    return feat

def load_labeled():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT id, processed_path, feedback FROM uploads WHERE feedback IS NOT NULL')
    rows = c.fetchall()
    conn.close()
    X = []
    y = []
    ids = []
    for uid, processed, fb in rows:
        p = os.path.join(PROCESSED, os.path.basename(processed))
        if not os.path.exists(p):
            continue
        f = extract_hog(p)
        if f is None:
            continue
        X.append(f)
        y.append(int(fb))
        ids.append(uid)
    return np.array(X), np.array(y), ids

def train_and_save():
    X, y, ids = load_labeled()
    if len(X) == 0:
        print('No labeled data found. Use feedback in the web UI to label images.')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print('Saved model to', MODEL_PATH)

if __name__ == '__main__':
    train_and_save()
