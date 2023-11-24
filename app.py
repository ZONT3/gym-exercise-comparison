import os

import numpy as np
from flask import Flask, request, jsonify

from gec import load_model, find_best_distances, compute_accuracy

app = Flask(__name__)
model = load_model()


@app.route('/process', methods=['POST'])
def process_video():
    sample_path = request.form['sample']
    target_path = request.form['target']

    distances, keypoints, offset = find_best_distances(model, sample_path, target_path,
                                                       step=env_float('GEC_OFFSET_STEP', 0.05),
                                                       initial_offset=env_float('GEC_OFFSET_INITIAL', 0.),
                                                       required_min_offset=env_float('GEC_OFFSET_UNTIL', 5.))
    accuracy = compute_accuracy(distances,
                                lerp_min=env_float('GEC_ACCURACY_LERP_MIN', 0.15),
                                lerp_max=env_float('GEC_ACCURACY_LERP_MAX', 0.005))
    mean_distance = np.mean(distances)

    return jsonify({'mean_distance': float(mean_distance), 'offset': float(offset), 'accuracy': float(accuracy)})


def env_float(key, default):
    return float(os.environ.get(key, default))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
