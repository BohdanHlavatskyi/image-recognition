import os
import joblib
import numpy as np
from sklearn.metrics import classification_report
from train_ml import load_labeled, extract_hog

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print('Model not found. Train first with train_ml.py')
        return
    X, y, ids = load_labeled()
    if len(X) == 0:
        print('No labeled data found.')
        return
    clf = joblib.load(MODEL_PATH)
    preds = clf.predict(X)
    print(classification_report(y, preds))

if __name__ == '__main__':
    evaluate()
