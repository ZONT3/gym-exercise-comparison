import cv2
import numpy as np
import tensorflow as tf

from gec import _load_videos, _read_frames, _normalize_keypoints


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


def _score_to_color(score):
    return int(255 * (1 - score)), int(255 * score), 0
