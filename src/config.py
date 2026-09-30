"""
config.py

오큘로그(Oculog) 프로젝트 공통 설정.
데이터 경로, 전처리 파라미터, 모델 하이퍼파라미터를 한 곳에서 관리한다.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── 딥페이크 프로젝트 데이터 경로 ────────────────────────────
DEEPFAKE_DATA_DIR = os.path.join(DATA_DIR, "deepfake_gaze_ptgaze")
GENUINE_DIR = os.path.join(DEEPFAKE_DATA_DIR, "genuine")  # 진짜 영상에서 뽑은 시선
FAKE_DIR = os.path.join(DEEPFAKE_DATA_DIR, "fake")        # 딥페이크 영상에서 뽑은 시선

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── 신호 처리 파라미터 ──────────────────────────────────────
# 주의: 웹캠/영상 fps에 맞춰야 함. 대부분 30fps 기준.
# ETPAD(EyeLink 1000Hz)와는 샘플링레이트가 크게 다르므로 재조정 필요.
SAMPLING_RATE_HZ = 30
RECORDING_DURATION_SEC = 10   # 논문(60초)보다 단축 — 실시간 데모/데이터 규모 고려

# ── 시계열 윈도우 파라미터 ──────────────────────────────────
# ETPAD 버전(200ms/100ms)보다 길게 — 샘플링레이트가 훨씬 낮으므로
# (30Hz에서 200ms window면 프레임 6개뿐이라 정보가 너무 적음)
WINDOW_SIZE_MS = 1000          # 1초짜리 윈도우
WINDOW_STRIDE_MS = 500         # 0.5초씩 이동 (50% 오버랩)
N_WINDOWS_PER_SEQUENCE = 18    # 10초 길이를 0.5초 stride로 커버 (약 (10-1)/0.5+1=19 -> 18~19)

# ── 라벨 정의 ──────────────────────────────────────────────
LABEL_GENUINE = 0
LABEL_FAKE = 1

# ── 학습 파라미터 ──────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 30                    # 논문(50) 대비 축소 — CPU 학습시간 고려
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ── 데이터 분할 ──────────────────────────────────────────
TEST_SIZE_RATIO = 0.2

# ── 도메인 불일치 영상 제외 목록 ──────────────────────────
# 오판 사례를 육안으로 검토한 결과, 뉴스 방송이나 카메라를 향해
# 혼자 설명/강연하는 브이로그(대화 상대가 없는 발화 형식)가
# 딥페이크 특유의 "인위적으로 안정된 시선"과 혼동되는 경향이
# 확인됐다. 이런 영상들은 우리 목표 시나리오(1:1 화상통화, 대화
# 상대가 있어 시선이 자연스럽게 움직이는 상황)와 근본적으로
# 다른 발화 형식이라 판단해 학습/평가에서 제외한다.
# 해당 subject의 원본(genuine)과 그로부터 파생된 조작본(fake)을
# 모두 제외해야 하므로, preprocessing.py에서 subject_id 단위로 거른다.
EXCLUDED_SUBJECTS = {
    # 뉴스/방송
    "179", "147", "169", "149", "106", "152", "402", "212", "788",
    "655", "895", "651", "742", "150", "263", "268", "521",
    "999", "834", "107", "036", "832",
    # 카메라를 향한 1인 설명/브이로그 (대화 상대 없음)
    "453", "748", "462", "751", "633", "764", "819",
}
