import glob
import json
import os.path
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from gec import load_model, cache_video
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
    parser.add_argument('--cache', action='store_true', help='Read directory and cache all video keypoints to default '
                                                             'cache dir')
    args = parser.parse_args()

    if args.cache:
        cache_keypoints(args.path, os.environ.get('GEC_CACHE', './keypoints_cache'))
        return

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


def cache_keypoints(video_dir, cache_dir):
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f'Specified path is not a dir:, {video_dir}')

    model = load_model()
    for f in os.listdir(str(video_dir)):
        file_path = video_dir / f
        if file_path.is_file() and file_path.name.split('.')[-1] in ('mp4', 'mov'):
            cache_video(model, file_path, cache_dir)


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
