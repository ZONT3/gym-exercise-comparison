import numpy as np
from flask import Flask, request, jsonify

from gec import load_model, find_best_distances, compute_accuracy

app = Flask(__name__)
model = load_model()


@app.route('/process', methods=['POST'])
def process_video():
    sample_path = request.form['sample']
    target_path = request.form['target']

    distances, keypoints, offset = find_best_distances(model, sample_path, target_path)
    accuracy = compute_accuracy(distances)
    mean_distance = np.mean(distances)

    return jsonify({'mean_distance': float(mean_distance), 'offset': float(offset), 'accuracy': float(accuracy)})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
