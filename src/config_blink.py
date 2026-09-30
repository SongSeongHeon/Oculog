"""
config_blink.py

깜빡임(EAR) 기반 딥페이크 탐지 파이프라인 설정. config.py(시선 파이프라인)와
거의 동일하되, 데이터 경로와 입력 채널 수만 다르다.

시선 파이프라인과 독립적으로 실행되지만, 도메인 제외 목록(EXCLUDED_SUBJECTS)은
동일한 이유(뉴스/브이로그 도메인 불일치)로 그대로 공유한다.
"""

import os
import config as gaze_config  # EXCLUDED_SUBJECTS, MODEL_DIR, OUTPUT_DIR 등 공유

BASE_DIR = gaze_config.BASE_DIR
DATA_DIR = gaze_config.DATA_DIR

# ── 깜빡임 프로젝트 데이터 경로 ──────────────────────────────
DEEPFAKE_DATA_DIR = os.path.join(DATA_DIR, "deepfake_blink")
GENUINE_DIR = os.path.join(DEEPFAKE_DATA_DIR, "genuine")
FAKE_DIR = os.path.join(DEEPFAKE_DATA_DIR, "fake")

MODEL_DIR = gaze_config.MODEL_DIR
OUTPUT_DIR = gaze_config.OUTPUT_DIR

# ── 신호 처리 파라미터 (시선과 동일) ─────────────────────────
SAMPLING_RATE_HZ = gaze_config.SAMPLING_RATE_HZ
RECORDING_DURATION_SEC = gaze_config.RECORDING_DURATION_SEC

# ── 시계열 윈도우 파라미터 (시선과 동일 — 같은 시퀀스 길이로 맞춰야
#    나중에 앙상블 시 두 모델의 예측 시점이 대응됨) ────────────
WINDOW_SIZE_MS = gaze_config.WINDOW_SIZE_MS
WINDOW_STRIDE_MS = gaze_config.WINDOW_STRIDE_MS
N_WINDOWS_PER_SEQUENCE = gaze_config.N_WINDOWS_PER_SEQUENCE

# ── 라벨 정의 (동일) ────────────────────────────────────────
LABEL_GENUINE = gaze_config.LABEL_GENUINE
LABEL_FAKE = gaze_config.LABEL_FAKE

# ── 학습 파라미터 (동일) ────────────────────────────────────
BATCH_SIZE = gaze_config.BATCH_SIZE
EPOCHS = gaze_config.EPOCHS
LEARNING_RATE = gaze_config.LEARNING_RATE
RANDOM_SEED = gaze_config.RANDOM_SEED

# ── 데이터 분할 (동일) ──────────────────────────────────────
TEST_SIZE_RATIO = gaze_config.TEST_SIZE_RATIO

# ── 도메인 불일치 영상 제외 목록 (시선 파이프라인과 동일하게 공유) ──
EXCLUDED_SUBJECTS = gaze_config.EXCLUDED_SUBJECTS

# ── 깜빡임 전용: 입력 특징 채널 수 ───────────────────────────
# ear, ear_velocity 2채널 (시선의 x, y, velocity 3채널과 다름)
N_FEATURES = 2
