from pathlib import Path

from gec import load_model, find_best_distances
from gec_visualize import visualize_keypoints


def main():
    target_path = Path('data_raw/1/003_3.mp4')
    sample_path = Path('data_raw/example/20231122_193331.mp4')

    model = load_model()
    distances, keypoints, offset = find_best_distances(model, sample_path, target_path)

    visualize_keypoints(sample_path, target_path, *keypoints,
                        crop_start=offset, title=f'Best offset: {offset:.2f}')


if __name__ == '__main__':
    main()
