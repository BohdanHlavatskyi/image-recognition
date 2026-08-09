import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    Xc = np.load(os.path.join(DATA_DIR, 'X_class.npy'))
    yc = np.load(os.path.join(DATA_DIR, 'y_class.npy'))
    Xr = np.load(os.path.join(DATA_DIR, 'X_reg.npy'))
    yr = np.load(os.path.join(DATA_DIR, 'y_reg.npy'))
    return Xc, yc, Xr, yr


def train():
    Xc, yc, Xr, yr = load_data()
    if len(Xc) == 0:
        print('No data found in', DATA_DIR)
        return

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xc_train, yc_train)
    preds = clf.predict(Xc_test)
    print('Classifier report:')
    print(classification_report(yc_test, preds))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'clf.pkl'))

    # Train regressor on positives only
    if len(Xr) > 0:
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(Xr_train, yr_train)
        pred_reg = reg.predict(Xr_test)
        print('Regressor MSE:', mean_squared_error(yr_test, pred_reg))
        joblib.dump(reg, os.path.join(MODEL_DIR, 'reg.pkl'))

    print('Saved models to', MODEL_DIR)


if __name__ == '__main__':
    train()
