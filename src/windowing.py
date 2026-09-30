"""
시계열 윈도우 분할

하나의 안구운동 레코딩(15초)을 CNN-LSTM 입력에 맞게
짧은 시간 구간(Window)들의 시퀀스로 변환한다.

구조: [Window_1, Window_2, ..., Window_N] -> LSTM
       각 Window 내부는 CNN이 로컬 패턴을 추출
"""

import numpy as np


def compute_features(df):
    """
    원시 (x, y) 좌표에서 안구운동 특징을 계산한다.
    최소 구성: x, y 위치 + 속도(velocity)
    TODO: fixation/saccade 여부, pupil area 등 컬럼이 있으면 추가
    """
    x = df["x"].values
    y = df["y"].values
    t = df["time"].values

    dt = np.diff(t, prepend=t[0])
    dt[dt == 0] = 1e-6  # 0으로 나누기 방지

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    velocity = np.sqrt(dx**2 + dy**2) / dt

    features = np.stack([x, y, velocity], axis=1)  # shape: (T, 3)
    return features


def split_into_windows(features, sampling_rate, window_ms, stride_ms):
    """
    (T, F) 형태의 시계열 특징을 겹치는 짧은 윈도우들로 분할한다.
    반환: (N_windows, window_len, F)

    주의: sampling_rate/window_ms/stride_ms는 프로젝트(ETPAD vs 딥페이크)마다
    값이 다르므로 기본값을 두지 않는다. 호출할 때 해당 프로젝트의 config에서
    명시적으로 넘길 것 (예: config.SAMPLING_RATE_HZ).
    """
    window_len = int(sampling_rate * window_ms / 1000)
    stride = int(sampling_rate * stride_ms / 1000)

    windows = []
    for start in range(0, len(features) - window_len + 1, stride):
        windows.append(features[start:start + window_len])

    return np.array(windows)


def build_sequence(features, sampling_rate, window_ms, stride_ms, n_windows):
    """
    윈도우 배열에서 LSTM 입력으로 쓸 고정 길이 시퀀스를 만든다.
    윈도우 수가 부족하면 0으로 패딩, 많으면 앞에서부터 자름.
    반환: (n_windows, window_len, F)
    """
    windows = split_into_windows(features, sampling_rate, window_ms, stride_ms)

    if len(windows) == 0:
        return None

    if len(windows) >= n_windows:
        return windows[:n_windows]
    else:
        pad_shape = (n_windows - len(windows),) + windows.shape[1:]
        padding = np.zeros(pad_shape)
        return np.concatenate([windows, padding], axis=0)
