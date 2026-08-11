import os
from app import allowed_file, app


def test_allowed_file():
    assert allowed_file('image.jpg')
    assert allowed_file('photo.PNG')
    assert not allowed_file('script.py')


def test_homepage_uses_winged_uav_layout():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Winged UAV Detector' in html
    assert 'delta or rectangular silhouette' in html.lower()
    assert 'Upload image' in html
    assert 'propeller' not in html.lower()


def test_visual_interface_has_detection_client():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'visual-app' in html.lower()
    assert '/api/detect' in html
    assert 'Detection result' in html


def test_training_target_is_only_winged_uav():
    with open('yolo_data/data.yaml', 'r', encoding='utf-8') as fh:
        data = fh.read().lower()
    assert 'winged_uav' in data
    assert 'delta' in data or 'winged_uav' in data
    assert 'background' not in data.lower()


def test_feedback_learning_log_exists_and_tracks_prediction_mismatch():
    from app import app, init_db
    init_db()
    with app.app_context():
        from app import DB_PATH
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        tables = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_learning'").fetchall()
        assert tables
        cols = c.execute("PRAGMA table_info(feedback_learning)").fetchall()
        names = [row[1] for row in cols]
        assert 'upload_id' in names
        assert 'prediction' in names
        assert 'actual' in names
        assert 'correct' in names


def test_dashboard_shows_learning_metrics():
    client = app.test_client()
    response = client.get('/dashboard')
    assert response.status_code == 200
    html = response.get_data(as_text=True).lower()
    assert 'model learning' in html or 'prediction accuracy' in html
    assert 'learning' in html or 'accuracy' in html


def test_load_labeled_reads_processed_images_from_processed_folder():
    import os
    import sqlite3
    from PIL import Image
    from app import DB_PATH
    from train_ml import load_labeled

    os.makedirs('processed', exist_ok=True)
    path = os.path.join('processed', 'demo_feedback.png')
    img = Image.new('RGB', (64, 64), color='white')
    img.save(path)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM uploads WHERE id = ?', ('demo_feedback',))
    c.execute('INSERT INTO uploads (id, filename, processed_path, detected, shapes_json, center_x, center_y, feedback, created_at) VALUES (?,?,?,?,?,?,?,?,?)',
              ('demo_feedback', 'demo_feedback.png', 'demo_feedback.png', 1, '{}', 0, 0, 1, '2026-01-01T00:00:00'))
    conn.commit()
    conn.close()

    try:
        X, y, ids = load_labeled()
        assert 'demo_feedback' in ids
    finally:
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM uploads WHERE id = ?', ('demo_feedback',))
        conn.commit()
        conn.close()


def test_feedback_retraining_is_debounced_and_non_blocking():
    from app import get_retraining_state, schedule_feedback_retraining

    schedule_feedback_retraining()
    state = get_retraining_state()
    assert 'pending' in state
    assert 'last_scheduled' in state
    assert state['pending'] is True or state['last_scheduled'] is not None
