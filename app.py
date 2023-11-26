import os
import sqlite3
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from gec import load_model, find_best_distances, compute_accuracy

app = Flask('gec')
CORS(app)

model = load_model()

DB_FILE = os.environ.get('GEC_DATA_FILE', './data/history.db')


def setup_db():
    db_dir = Path(DB_FILE).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        user_id TEXT,
        exercise_id INTEGER,
        exercise_score TEXT,
        mean_distance REAL,
        offset REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, exercise_id, timestamp)
    )
    ''')
    conn.commit()
    conn.close()


setup_db()


@app.route('/process', methods=['POST'])
def process_video():
    target_name = int(request.form['exercise_id'])
    user_id = str(request.form['user_id'])
    file = request.files['video']

    input_dir = Path('./input_video')
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_filename = secure_filename(f'{user_id}.mp4')
    sample_path = input_dir / sample_filename

    tries = 1
    while sample_path.is_file():
        sample_filename = secure_filename(f'{user_id}_{tries}.mp4')
        sample_path = input_dir / sample_filename
        tries += 1

    file.save(sample_path)

    distances, keypoints, offset = find_best_distances(model, sample_path, f'{target_name}.mp4',
                                                       step=env_float('GEC_OFFSET_STEP', 0.05),
                                                       initial_offset=env_float('GEC_OFFSET_INITIAL', 0.),
                                                       required_min_offset=env_float('GEC_OFFSET_UNTIL', 5.))
    sample_path.unlink(missing_ok=True)
    accuracy = compute_accuracy(distances,
                                lerp_min=env_float('GEC_ACCURACY_LERP_MIN', 0.15),
                                lerp_max=env_float('GEC_ACCURACY_LERP_MAX', 0.005))
    accuracy = f'{int(accuracy * 100):02d}%'
    offset = float(offset)
    mean_distance = float(np.mean(distances))

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (user_id, exercise_id, exercise_score, mean_distance, offset)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, target_name, accuracy, mean_distance, offset))

    conn.commit()
    conn.close()

    return jsonify({
        'exercise_score': accuracy,
        'mean_distance': mean_distance,
        'offset': offset,
    })


@app.route('/find', methods=['GET'])
def find():
    user_id = str(request.args.get('user_id'))
    exercise_id = int(request.args.get('exercise_id'))
    limit = request.args.get('limit', 10, type=int)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
    SELECT exercise_score, mean_distance, offset, timestamp FROM history
    WHERE user_id=? AND exercise_id=?
    ORDER BY timestamp DESC LIMIT ?
    ''', (user_id, exercise_id, limit))

    results = cursor.fetchall()
    conn.close()

    if results:
        return jsonify([
            {
                'exercise_score': str(record[0]),
                'mean_distance': float(record[1]),
                'offset': float(record[2]),
                'timestamp': str(record[3])
            }
            for record in results
        ])
    else:
        return jsonify({'error': 'No matching entry found'}), 404


def env_float(key, default):
    return float(os.environ.get(key, default))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
