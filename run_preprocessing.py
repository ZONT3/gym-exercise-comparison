import glob
import json
import os.path
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from gec import load_model
from gec_visualize import single_visualize_keypoints


def main():
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='Video file path or glob (see below)')
    parser.add_argument('-g', '--glob', action='store_true', help='Enable glob search for multiple files')
    parser.add_argument('-o', '--output', default=None, type=str, help='Save the output to specified dir instead of '
                                                                       'displaying. Implicitly sets -g option')
    parser.add_argument('-c', '--check', default=None, type=str, help='Check the overall prediction scores for '
                                                                      'specified keypoints.json (created by -o option) '
                                                                      'and print all filenames with mean score equal '
                                                                      'or less than specified in this option')
    args = parser.parse_args()

    if args.check:
        check_keypoints_scores(args.path, float(args.check))
        return

    if not args.glob and not args.output:
        single_visualize_keypoints(args.path)
    else:
        model = load_model()
        keypoints = {}
        for file in glob.glob(args.path):
            result = single_visualize_keypoints(os.path.normpath(file), save=args.output, model=model, title=file)
            if args.output:
                keypoints[file] = result

        if args.output:
            json_file = Path(args.output) / 'keypoints.json'
            json_file.unlink(missing_ok=True)
            with open(json_file, 'w') as f:
                json.dump({k: v.tolist() for k, v in keypoints.items()}, f)


def check_keypoints_scores(json_file: str, score: float):
    keypoints = load_keypoints(json_file)
    print('\n'.join(f'{k}: {v:.3f}' for k, v in (
            (k, get_keypoint_score(v))
            for k, v in keypoints.items()
        ) if v <= score
    ))


def load_keypoints(json_file):
    with open(json_file, 'r') as f:
        keypoints = json.load(f)
    return {k: np.array(v) for k, v in keypoints.items()}


def get_keypoint_score(keypoint):
    return np.mean(keypoint[..., 2])


if __name__ == '__main__':
    main()
