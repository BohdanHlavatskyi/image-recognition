#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path

from app import UPLOAD_FOLDER, PROCESSED_FOLDER, DB_PATH, detect_shapes_and_draw


def reprocess(limit=None):
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    files.sort()
    if limit:
        files = files[-limit:]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for fname in files:
        uid = fname.rsplit('.', 1)[0]
        ext = fname.rsplit('.', 1)[1]
        save_path = os.path.join(UPLOAD_FOLDER, fname)
        processed_name = f"{uid}_processed.{ext}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
        try:
            detected, shapes_json, center, chosen = detect_shapes_and_draw(save_path, processed_path)
            c.execute('''
                UPDATE uploads SET processed_path=?, detected=?, shapes_json=?, center_x=?, center_y=? WHERE id=?
            ''', (processed_name, int(detected), shapes_json, center[0] if center[0] else None, center[1] if center[1] else None, uid))
            print(f"Processed {fname} -> {processed_name} detected={detected}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    conn.commit()
    conn.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Reprocess uploaded images using current models')
    p.add_argument('--limit', type=int, help='Only reprocess last N uploads')
    args = p.parse_args()
    reprocess(args.limit)
