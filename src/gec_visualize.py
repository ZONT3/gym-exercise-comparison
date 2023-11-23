from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from gec import _load_videos, _read_frames, _normalize_keypoints, log, process_frames, load_model


def visualize_keypoints(sample_path, target_path, keypoints_sample, keypoints_target, framerate,
                        crop_start=0., crop_end=None, title='KeyPoints Visualization'):
    """
    Визуализировать сравнение двух видео
    :param sample_path: Путь к видео пользователя
    :param target_path: Путь к видео тренера
    :param keypoints_sample: Уже извлеченный скелет пользователя
    :param keypoints_target: Уже извлеченный скелет тренера
    :param framerate: Привести оба видео к одному FPS
    :param crop_start: Обрезать видео пользователя с этой секунды
    :param crop_end: Обрезать видео пользователя до этой секунды
    :param title: Названия окна с визуализацией
    :return:
    """
    sample_video, target_video = _load_videos(sample_path, target_path)

    frames_sample = _read_frames(sample_video, framerate)
    frames_target = _read_frames(target_video, framerate)
    sample_video.release()
    target_video.release()

    start_frame = int(crop_start * framerate)
    end_frame = int(crop_end * framerate) if crop_end else len(frames_sample)
    frames_sample = frames_sample[start_frame:end_frame]
    keypoints_sample = keypoints_sample[start_frame:end_frame]

    min_length = min(len(frames_sample), len(frames_target))
    frames_sample = frames_sample[:min_length]
    frames_target = frames_target[:min_length]
    keypoints_sample = keypoints_sample[:min_length]
    keypoints_target = keypoints_target[:min_length]

    _visualization_loop(framerate, frames_sample, frames_target, keypoints_sample, keypoints_target, title)
    cv2.destroyAllWindows()


def _visualization_loop(framerate, frames_sample, frames_target, keypoints_sample, keypoints_target, title, size=600):
    while True:
        for idx, data in enumerate(zip(frames_sample, frames_target, keypoints_sample, keypoints_target)):
            frame_sample, frame_target, keypoints_s, keypoints_t = data

            frame_sample_resized = tf.image.resize_with_pad(frame_sample, size, size).numpy().astype(np.uint8)
            frame_target_resized = tf.image.resize_with_pad(frame_target, size, size).numpy().astype(np.uint8)

            keypoints_s_coords = keypoints_s[:, :, :2] * size
            keypoints_t_coords = keypoints_t[:, :, :2] * size

            for coords_s, coords_t, score_s, score_t in zip(
                    keypoints_s_coords[0], keypoints_t_coords[0], keypoints_s[0, :, 2], keypoints_t[0, :, 2]
            ):
                cv2.circle(frame_sample_resized, (int(coords_s[1]), int(coords_s[0])), 5, _score_to_color(score_s), -1)
                cv2.circle(frame_target_resized, (int(coords_t[1]), int(coords_t[0])), 5, _score_to_color(score_t), -1)

            frame_sample_resized = cv2.cvtColor(frame_sample_resized, cv2.COLOR_RGB2BGR)
            frame_target_resized = cv2.cvtColor(frame_target_resized, cv2.COLOR_RGB2BGR)

            keypoints_s_normalized = _normalize_keypoints(keypoints_s[0, :, :2])
            keypoints_t_normalized = _normalize_keypoints(keypoints_t[0, :, :2])
            mean_distance = np.mean(np.sqrt(np.sum((keypoints_s_normalized - keypoints_t_normalized) ** 2, axis=1)))

            videos_frame = np.concatenate((frame_sample_resized, frame_target_resized), axis=1)

            result_frame = np.zeros((videos_frame.shape[0] + 30, videos_frame.shape[1], 3), dtype=np.uint8)
            result_frame[:videos_frame.shape[0], :, :] = videos_frame

            cv2.putText(result_frame, f"Frame: {idx:05d}, mean distance: {mean_distance:.4f}",
                        (10, videos_frame.shape[0] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(title, result_frame)
            if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                return


def single_visualize_keypoints(video_path, title='KeyPoints Visualization', model=None, save=None):
    """
    Визуализировать сравнение двух видео
    :param video_path: Путь к видео
    :param title: Названия окна с визуализацией
    :param model: Модель MoveNet, по умолчанию - gec.load_model()
    :param save: Путь для сохранения видео вместо вывода в плеере
    :return:
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f'Target video not found: {str(video_path)}')

    if save:
        save = Path(save)
        save.mkdir(parents=True, exist_ok=True)
        save = save / (Path(title).name + '.avi')
        save.unlink(missing_ok=True)

    log('Reading video...')
    video = cv2.VideoCapture(str(video_path))
    framerate = video.get(cv2.CAP_PROP_FPS)

    frames = _read_frames(video, framerate)
    video.release()

    if model is None:
        model = load_model()

    keypoints = process_frames(model, frames)

    result = _single_visualization_loop(framerate, frames, keypoints, title, save)
    cv2.destroyAllWindows()
    return result


def _single_visualization_loop(framerate, frames, keypoints, title, save, size=800):
    while True:
        output_frames = []
        for idx, data in enumerate(zip(frames, keypoints)):
            frame, cur_keypoints = data

            frame_resized = tf.image.resize_with_pad(frame, size, size).numpy().astype(np.uint8)
            keypoints_coords = cur_keypoints[:, :, :2] * size

            for coords, score in zip(keypoints_coords[0], cur_keypoints[0, :, 2]):
                cv2.circle(frame_resized, (int(coords[1]), int(coords[0])), 5, _score_to_color(score), -1)

            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

            if save is None:
                cv2.imshow(title, frame_resized)
                if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                    return None
            else:
                output_frames.append(frame_resized)

        if save is not None:
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(save), fourcc, framerate, (size, size))
            for frame in output_frames:
                video_writer.write(frame)
            video_writer.release()
            return keypoints


def _score_to_color(score):
    return int(255 * (1 - score)), int(255 * score), 0
