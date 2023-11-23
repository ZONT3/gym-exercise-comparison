import sys
from argparse import ArgumentParser

import numpy as np

from gec import load_model, find_best_distances


def main():
    parser = ArgumentParser()
    parser.add_argument('sample', type=str, help='Sample (user) video file')
    parser.add_argument('target', type=str, help='Target (coach) video file')
    parser.add_argument('-s', '--step', type=str, default='0.05', help='Offset searching step (seconds)')
    parser.add_argument('-o', '--initial-offset', type=str, default='0', help='Initial offset for search')
    parser.add_argument('-r', '--required-offset', type=str, default='5',
                        help='Minimum required offset to search until')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize video comparison on best offset')
    args = parser.parse_args()

    sample_path = args.sample
    target_path = args.target

    model = load_model()
    distances, keypoints, offset = find_best_distances(model, sample_path, target_path,
                                                       step=float(args.step), initial_offset=float(args.initial_offset),
                                                       required_min_offset=float(args.required_offset))

    if args.visualize:
        from gec_visualize import visualize_keypoints
        visualize_keypoints(sample_path, target_path, *keypoints,
                            crop_start=offset, title=f'Best offset: {offset:.2f}')

    # print('Best distances:', distances, file=sys.stderr)
    print('Best offset:', offset, file=sys.stderr)
    print(np.mean(distances))


if __name__ == '__main__':
    main()
