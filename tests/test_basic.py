import os
from app import allowed_file

def test_allowed_file():
    assert allowed_file('image.jpg')
    assert allowed_file('photo.PNG')
    assert not allowed_file('script.py')
