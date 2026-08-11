import os
from app import allowed_file, app


def test_allowed_file():
    assert allowed_file('image.jpg')
    assert allowed_file('photo.PNG')
    assert not allowed_file('script.py')


def test_homepage_uses_simple_drone_detector_layout():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'Drone Detector' in html
    assert 'Upload image' in html
    assert 'Detect' in html
