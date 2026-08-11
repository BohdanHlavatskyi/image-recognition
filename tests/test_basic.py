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
