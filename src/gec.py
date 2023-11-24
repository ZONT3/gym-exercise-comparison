import _pickle as pkl
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger('gec')


def log(*args, **kwargs):
    logger.info(*args, **kwargs)


def load_model():
    """
    Загрузить модель MoveNet.
    Занимает много времени даже если она уже скачана
    :return: Загруженная модель
    """
    log('Preparing model...')
    model_url = "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder" \
                "/versions/4"
    model_hash = "ba50920a0dee34563eef6669110aab92916d223a"
    model_path = Path(os.environ.get('TFHUB_CACHE_DIR', None) or './tfhub_cache') / model_hash
    if model_path.is_dir():
        model = hub.load(str(model_path))
    else:
        model = hub.load(model_url)
    log('Model prepared')
    return model.signatures['serving_default']


def process_frames(model, frames):
    """
    Получить скелет для всех фреймов в видео
    :param model: модель MoveNet
    :param frames: фреймы видео в формате [frames_count, height, width, channels]
    :return:
    """
    processed_keypoints = []
    for frame in tqdm(frames, total=frames.shape[0], desc='Recognizing skeletons'):
        input_tensor = tf.image.resize_with_pad(frame, 256, 256)
        input_tensor = tf.cast(tf.expand_dims(input_tensor, axis=0), dtype=tf.int32)

        outputs = model(input_tensor)
        keypoints = outputs['output_0']

        processed_keypoints.append(keypoints.numpy()[0])

    return np.array(processed_keypoints)


def find_best_distances(model, sample_path, target_path, step=0.05, required_min_offset=5., initial_offset=0.,
                        cache_dir=os.environ.get('GEC_CACHE', './keypoints_cache')):
    """
    Подобрать наилучший оффсет для видео пользователя и вывести расстояния для всех узлов скелета
    (в сравнении с видео от тренера)
    :param model: модель MoveNet
    :param sample_path: Путь к видео пользователя
    :param target_path: Путь к видео тренера
    :param step: Шаг подбора оффсета (сек)
    :param required_min_offset: подбирать оффсет как минимум до этого значения
    :param initial_offset: Начальное значение оффсета
    :param cache_dir: Директория для кэширования обработанных видео тренера
    :return: Tuple[distances, keypoints_args, offset]:
            Наилучшие расстояния узлов скелета пользователя относительно тренера, формат:
                    [frames_count, keypoints_count (17)],
            Извлеченные скелеты (не обрезанные),
            Наилучший оффсет
    """
    # log(f'Trying offset {initial_offset}...')
    distance, keypoints_args = compute_distances(model, sample_path=sample_path, target_path=target_path,
                                                 crop_start=initial_offset, cache_dir=cache_dir)
    # log(f'Mean distance: {np.mean(distance)}')

    offset = initial_offset + step
    distances = []
    while offset <= required_min_offset or distance is not None:
        distances.append(distance)

        # log(f'Trying offset {offset:.2f}...')
        distance, _ = compute_distances(model, keypoints_args=keypoints_args, crop_start=offset, cache_dir=cache_dir,
                                        sample_less_ok=(offset <= required_min_offset))
        # if distance is not None:
        #     log(f'Mean distance: {np.mean(distance)}')
        # else:
        #     log(f'Stop condition reached.')

        offset += step

    distances_mean = np.array([np.mean(x) for x in distances])
    best_idx = int(np.argmin(distances_mean))

    return distances[best_idx], keypoints_args, initial_offset + step * best_idx


def compute_distances(model, *, sample_path=None, target_path=None, crop_start=0., crop_end=None, sample_less_ok=True,
                      cache_dir=None, keypoints_args: Optional[Tuple[np.ndarray, np.ndarray, float]] = None):
    """
    Сравнить видео пользователя с видео тренера и вывести расстояния между узлами их скелетов
    :param model: Модель MoveNet
    :param sample_path: Путь к видео от пользователя
    :param target_path: Путь к видео от тренера
    :param crop_start: Обрезать видео пользователя с этой секунды
    :param crop_end: Обрезать видео пользователя до этой секунды
    :param sample_less_ok: Не возвращать None если обрезанное видео пользователя меньше по длине, чем тренера
    :param cache_dir: Директория для кэширования обработанных видео тренера
    :param keypoints_args: Уже извлеченные скелеты обоих видео. Если задан - sample_path и target_path не используются
    :return: Расстояния узлов скелета пользователя относительно тренера, формат: [frames_count, keypoints_count (17)]
    """
    if not keypoints_args:
        if not sample_path or not target_path:
            raise ValueError('Paths should be specified if not using already extracted keypoints')
        keypoints_sample, keypoints_target, min_fps = _extract_keypoints(
            model, sample_path, target_path, cache_dir=cache_dir
        )
    else:
        keypoints_sample, keypoints_target, min_fps = keypoints_args

    keypoints_sample_cropped, keypoints_target_cropped = _crop_keypoints(
        keypoints_sample, keypoints_target, crop_end, crop_start, min_fps, sample_less_ok
    )

    if any(x is None for x in (keypoints_sample_cropped, keypoints_target_cropped)):
        return None, None

    keypoints_sample_norm = _normalize_keypoints(keypoints_sample_cropped[..., :2])
    keypoints_target_norm = _normalize_keypoints(keypoints_target_cropped[..., :2])

    distances = np.sqrt(
        np.sum((keypoints_sample_norm - keypoints_target_norm) ** 2, axis=3)
    ).reshape((-1, 17))

    distances *= keypoints_target_cropped[..., 2].reshape((-1, 17))
    distances *= keypoints_sample_cropped[..., 2].reshape((-1, 17))

    return distances, (keypoints_sample, keypoints_target, min_fps)


def compute_accuracy(distances, lerp_min=0.10, lerp_max=0.005):
    mean = np.mean(distances)
    accuracy = (mean - lerp_min) / (lerp_max - lerp_min)
    return max(min(accuracy, 1.), 0.)


def _crop_keypoints(keypoints_sample, keypoints_target, crop_end, crop_start, framerate, sample_less_ok=True):
    start_frame = int(crop_start * framerate)
    end_frame = int(crop_end * framerate) if crop_end else len(keypoints_sample)

    keypoints_sample_cropped = keypoints_sample[start_frame:end_frame]

    if not sample_less_ok and keypoints_sample_cropped.shape[0] < keypoints_target.shape[0]:
        return None, None

    min_len = min(keypoints_sample_cropped.shape[0], keypoints_target.shape[0])
    keypoints_sample_cropped = keypoints_sample_cropped[:min_len]
    keypoints_target_cropped = keypoints_target[:min_len]
    return keypoints_sample_cropped, keypoints_target_cropped


def _extract_keypoints(model, sample_path, target_path, cache_dir=None):
    sample_video, target_video = _load_videos(sample_path, target_path, cache_dir is not None)

    cache_name = Path(target_path).name
    keypoints_target, framerate_target = _get_cached(cache_dir, cache_name) if cache_dir is not None else (None, None)

    if keypoints_target is None and target_video is None:
        raise FileNotFoundError(f'Target video not found both in cache and on path {target_path}')

    sample_fps = sample_video.get(cv2.CAP_PROP_FPS)
    target_fps = target_video.get(cv2.CAP_PROP_FPS) if framerate_target is None else framerate_target
    min_fps = min(sample_fps, target_fps)

    if keypoints_target is None:
        frames_target = _read_frames(target_video, min_fps)
        target_video.release()
        keypoints_target = process_frames(model, frames_target)
        _cache_kp(keypoints_target, min_fps, cache_dir, cache_name)
    elif target_fps > min_fps:
        keypoints_target = _reduce_framerate(keypoints_target, target_fps, min_fps)

    frames_sample = _read_frames(sample_video, min_fps)
    sample_video.release()
    keypoints_sample = process_frames(model, frames_sample)

    return keypoints_sample, keypoints_target, min_fps


def cache_video(model, video_path, cache_dir):
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f'Video file {str(video_path)} not found')

    video = cv2.VideoCapture(str(video_path))
    frames = _read_frames(video)
    keypoints = process_frames(model, frames)
    _cache_kp(keypoints, video.get(cv2.CAP_PROP_FPS), cache_dir, video_path.name)


def _cache_kp(keypoints, framerate, cache_dir, video_name):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'keypoints.pkl'

    try:
        if cache_file.is_file():
            with open(cache_file, 'rb') as f:
                cache = pkl.load(f)
        else:
            cache = {}

        cache[video_name] = (keypoints, framerate)

        with open(cache_file, 'wb') as f:
            pkl.dump(cache, f)

    except Exception as e:
        logger.exception(f'Cannot cache keypoints for {video_name}', exc_info=e)


def _get_cached(cache_dir, video_name):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'keypoints.pkl'

    if not cache_file.is_file():
        return None, None

    # noinspection PyBroadException
    try:
        with open(cache_file, 'rb') as f:
            cache = pkl.load(f)
        if video_name in cache:
            log(f'Using cached target keypoints: {video_name}')
            return cache[video_name]
    except Exception as e:
        logger.exception(f'Cannot load cached keypoints for {video_name}', exc_info=e)
    return None, None


def _load_videos(sample_path, target_path, target_missing_ok=False):
    sample_path = Path(sample_path)
    target_path = Path(target_path)

    if not sample_path.is_file():
        raise FileNotFoundError(f'Sample video not found: {str(sample_path)}')
    if not target_missing_ok and not target_path.is_file():
        raise FileNotFoundError(f'Target video not found: {str(target_path)}')
    elif not target_path.is_file():
        target_path = None

    log('Reading videos...')
    sample_video = cv2.VideoCapture(str(sample_path))
    target_video = cv2.VideoCapture(str(target_path)) if target_path is not None else None
    return sample_video, target_video


def _normalize_keypoints(keypoints):
    x_coords = keypoints[..., 0]
    y_coords = keypoints[..., 1]

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_coords = (x_coords - x_min) / (x_max - x_min)

    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    y_coords = (y_coords - y_min) / (y_max - y_min)

    normalized_keypoints = keypoints.copy()
    normalized_keypoints[..., 0] = x_coords
    normalized_keypoints[..., 1] = y_coords

    return normalized_keypoints


def _reduce_framerate(frames, src_framerate, target_framerate):
    frame_skip_ratio = src_framerate / target_framerate
    new_frames = [x for i, x in enumerate(frames) if i % frame_skip_ratio < 1]
    return np.stack(new_frames)


def _read_frames(video: cv2.VideoCapture, target_framerate=None):
    frames = []
    frame_skip_ratio = 1. if target_framerate is None else video.get(cv2.CAP_PROP_FPS) / target_framerate
    frame_counter = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_counter % frame_skip_ratio < 1:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_counter += 1

    return np.stack(frames)
