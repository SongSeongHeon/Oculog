"""
app.py

오큘로그 메인 웹 서비스. oculog 환경(WSL2, tensorflow)에서 실행한다.

업로드된 영상을 받아 ptgaze_server.py(별도 프로세스, oculog_ptgaze 환경)에
전달해 시선 시계열을 추출하고, windowing.py로 시퀀스를 만든 뒤 학습된
CNN-LSTM 모델로 진짜/딥페이크를 판정한다.

사전 조건:
    1. ptgaze_server.py가 oculog_ptgaze 환경에서 먼저 실행되어 있어야 한다
       (기본적으로 http://localhost:5001)
    2. models/oculog_cnn_lstm.keras (또는 config.FINAL_MODEL_NAME)가 존재해야 한다

실행 (oculog 환경, WSL2):
    python app.py
    -> http://localhost:5000 에서 대기
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

import config
import windowing
import config_blink
import blink_extractor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB 업로드 제한

PTGAZE_SERVER_URL = "http://localhost:5001"
FINAL_MODEL_NAME = "cnn_lstm"  # evaluate.py로 확정한 최종 모델
MODEL_PATH = os.path.join(config.MODEL_DIR, f"oculog_{FINAL_MODEL_NAME}.keras")

FINAL_BLINK_MODEL_NAME = "cnn_lstm"  # evaluate_ensemble.py로 확정한 깜빡임 모델
BLINK_MODEL_PATH = os.path.join(
    config_blink.MODEL_DIR, f"oculog_blink_{FINAL_BLINK_MODEL_NAME}.keras"
)

# evaluate_ensemble.py로 확정한 최적 가중치 (시선:깜빡임 = 0.9:0.1일 때
# test set 기준 AUC가 가장 좋았음 — 0.8664 -> 0.8850). 시선 신호가
# 훨씬 강력하므로(AUC 0.87 vs 0.78) 깜빡임은 보조 신호로 적은 비중만 반영.
GAZE_WEIGHT = 0.9
BLINK_WEIGHT = 1.0 - GAZE_WEIGHT

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

_model = None
_blink_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"학습된 모델을 찾을 수 없습니다: {MODEL_PATH}. "
                "먼저 train.py로 모델을 학습해야 합니다."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def get_blink_model():
    global _blink_model
    if _blink_model is None:
        if not os.path.exists(BLINK_MODEL_PATH):
            raise FileNotFoundError(
                f"학습된 깜빡임 모델을 찾을 수 없습니다: {BLINK_MODEL_PATH}. "
                "먼저 train_blink.py로 모델을 학습해야 합니다."
            )
        _blink_model = tf.keras.models.load_model(BLINK_MODEL_PATH)
    return _blink_model


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def samples_to_dataframe(samples: list):
    """ptgaze_server가 반환한 [[time, yaw, pitch], ...] 리스트를
    windowing.compute_features()가 기대하는 (time, x, y) 형식으로 변환.
    video_to_gaze_ptgaze.py의 저장 규약과 동일: x=yaw, y=pitch."""
    import pandas as pd

    arr = np.array(samples)
    if arr.size == 0:
        return None
    return pd.DataFrame({"time": arr[:, 0], "x": arr[:, 1], "y": arr[:, 2]})


def normalize_like_training(features: np.ndarray) -> np.ndarray:
    """
    dataset.py의 normalize_features()와 동일한 방식(z-score)으로 정규화한다.
    주의: 학습 시에는 전체 학습 데이터셋의 평균/표준편차를 사용했는데,
    이 데모는 그 통계치를 별도로 저장해두지 않았으므로 근사적으로
    이 샘플 자체의 통계로 정규화한다. 샘플 수가 충분히 많으면(수백 프레임)
    학습 시 통계와 크게 다르지 않을 것으로 기대하지만, 완벽히 동일하지는
    않다는 한계가 있다. 더 정확히 하려면 학습 시 계산한 mean/std를
    별도 파일로 저장해 여기서 불러와야 한다 (향후 개선 과제).
    """
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1e-6
    return (features - mean) / std


def blink_samples_to_dataframe(blink_samples: list):
    """ptgaze_server가 반환한 [[time, ear], ...] 리스트를 DataFrame으로 변환."""
    import pandas as pd

    arr = np.array(blink_samples)
    if arr.size == 0:
        return None
    return pd.DataFrame({"time": arr[:, 0], "ear": arr[:, 1]})


def resample_to_target_rate(samples: list, target_hz: float) -> list:
    """실시간 스트리밍으로 모은 저밀도 샘플(예: 5Hz)을 학습 데이터와 같은
    밀도(target_hz, 보통 30Hz)로 선형보간(interpolation)한다.

    배경: 모델은 window_len(=sampling_rate * window_ms / 1000)이 고정된
    shape(예: 30프레임)을 기대하도록 학습됐다. sampling_rate 자체를 실시간
    실제값(5Hz)으로 바꾸면 window_len이 달라져 모델 입력 shape이 깨진다
    (실제로 "expected shape=(None,18,30,3), found shape=(1,18,4,3)" 에러
    발생). 따라서 sampling_rate는 항상 학습 시 고정값을 쓰고, 대신
    이 함수로 원본 시간축(time)에 맞춰 30Hz 밀도로 다시 샘플링해
    프레임 수 자체를 맞춰야 한다.

    samples: [[time, feat1, feat2, ...], ...] 형식, time은 초 단위
    반환: 동일 형식이지만 time이 target_hz 간격으로 균일하게 재샘플링됨
    """
    if len(samples) < 2:
        return samples

    arr = np.array(samples, dtype=float)
    times = arr[:, 0]
    values = arr[:, 1:]

    t_start, t_end = times[0], times[-1]
    duration = t_end - t_start
    if duration <= 0:
        return samples

    n_target = max(2, int(duration * target_hz) + 1)
    new_times = np.linspace(t_start, t_end, n_target)

    new_values = np.empty((n_target, values.shape[1]))
    for col in range(values.shape[1]):
        new_values[:, col] = np.interp(new_times, times, values[:, col])

    return [[new_times[i]] + list(new_values[i]) for i in range(n_target)]


def predict_gaze_score(samples: list):
    """시선 시계열로 판정 점수(0~1)를 계산. 실패 시 None을 반환하고
    error 메시지를 함께 준다."""
    if len(samples) < 30:
        return None, "얼굴을 안정적으로 인식하지 못했습니다. 얼굴이 잘 보이도록 다시 시도해주세요."

    # 실시간 스트리밍은 저밀도(예: 5Hz)로 모이므로, 학습 때와 같은
    # 30Hz 밀도로 보간해서 window_len(모델 입력 shape)을 맞춘다.
    # 업로드(이미 30fps 원본)는 사실상 값이 거의 그대로 유지된다.
    samples = resample_to_target_rate(samples, config.SAMPLING_RATE_HZ)

    df = samples_to_dataframe(samples)
    features = windowing.compute_features(df)
    sequence = windowing.build_sequence(
        features,
        sampling_rate=config.SAMPLING_RATE_HZ,
        window_ms=config.WINDOW_SIZE_MS,
        stride_ms=config.WINDOW_STRIDE_MS,
        n_windows=config.N_WINDOWS_PER_SEQUENCE,
    )
    if sequence is None:
        return None, "데이터가 너무 짧아 분석할 수 없습니다 (최소 10초 필요)."

    sequence = normalize_like_training(sequence)
    batch = np.expand_dims(sequence, axis=0)
    model = get_model()
    score = float(model.predict(batch, verbose=0)[0][0])
    return score, None


def predict_blink_score(blink_samples: list):
    """깜빡임 시계열로 판정 점수(0~1)를 계산. 데이터가 부족하거나 없으면
    (None, 이유)를 반환 — 이 경우 앙상블은 시선 단독으로 대체된다."""
    if not blink_samples or len(blink_samples) < 30:
        return None, "깜빡임 데이터 부족"

    blink_samples = resample_to_target_rate(blink_samples, config_blink.SAMPLING_RATE_HZ)

    df = blink_samples_to_dataframe(blink_samples)
    features = blink_extractor.compute_blink_features(df)
    sequence = windowing.build_sequence(
        features,
        sampling_rate=config_blink.SAMPLING_RATE_HZ,
        window_ms=config_blink.WINDOW_SIZE_MS,
        stride_ms=config_blink.WINDOW_STRIDE_MS,
        n_windows=config_blink.N_WINDOWS_PER_SEQUENCE,
    )
    if sequence is None:
        return None, "깜빡임 시퀀스 생성 실패"

    sequence = normalize_like_training(sequence)
    batch = np.expand_dims(sequence, axis=0)
    model = get_blink_model()
    score = float(model.predict(batch, verbose=0)[0][0])
    return score, None


def predict_from_samples(samples: list, blink_samples: list = None) -> dict:
    """시선(+가능하면 깜빡임) 시계열로부터 앙상블 판정을 수행한다.
    업로드(/analyze/upload)와 스트리밍(/analyze/stream/predict) 둘 다 재사용한다.

    앙상블 가중치(GAZE_WEIGHT:BLINK_WEIGHT = 0.9:0.1)는 evaluate_ensemble.py로
    test set에서 확정한 값이다 (AUC 0.8664 -> 0.8850으로 개선 확인됨).
    깜빡임 데이터가 없거나 너무 짧으면 시선 단독 점수를 그대로 사용한다
    (앙상블이 실패해도 서비스 자체는 계속 동작하도록).
    """
    gaze_score, gaze_error = predict_gaze_score(samples)
    if gaze_score is None:
        return {"error": gaze_error, "_status": 422}

    blink_score, blink_error = None, None
    if blink_samples:
        blink_score, blink_error = predict_blink_score(blink_samples)

    if blink_score is not None:
        final_score = GAZE_WEIGHT * gaze_score + BLINK_WEIGHT * blink_score
        used_blink = True
    else:
        final_score = gaze_score
        used_blink = False

    label = "딥페이크로 추정" if final_score >= 0.5 else "실제 영상으로 추정"
    confidence = final_score if final_score >= 0.5 else (1 - final_score)

    return {
        "label": label,
        "is_fake": final_score >= 0.5,
        "score": final_score,
        "gaze_score": gaze_score,
        "blink_score": blink_score,
        "used_blink_ensemble": used_blink,
        "confidence": round(confidence * 100, 1),
        "frames_used": len(samples),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze/upload", methods=["POST"])
def analyze_upload():
    if "video" not in request.files:
        return jsonify({"error": "영상 파일이 없습니다."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
    if not allowed_file(video_file.filename):
        return jsonify({"error": "지원하지 않는 파일 형식입니다 (mp4/avi/mov/mkv만 가능)."}), 400

    suffix = os.path.splitext(video_file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = requests.post(
                f"{PTGAZE_SERVER_URL}/extract",
                files={"video": (video_file.filename, f)},
                timeout=300,
            )

        if resp.status_code != 200:
            return jsonify({"error": f"시선 추출 서버 오류: {resp.text}"}), 502

        data = resp.json()
        result = predict_from_samples(data["samples"], data.get("blink_samples"))
        status = result.pop("_status", 200)
        if status == 200:
            result["detection_rate"] = round(data.get("detection_rate", 0) * 100, 1)
        return jsonify(result), status

    except requests.exceptions.ConnectionError:
        return jsonify(
            {"error": "시선 추출 서버에 연결할 수 없습니다. ptgaze_server.py가 실행 중인지 확인해주세요."}
        ), 503
    except Exception as e:
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {e}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 실시간 웹캠 스트리밍 ───────────────────────────────────
# 프레임 자체는 브라우저 -> ptgaze_server로 직접 안 보내고, 브라우저가
# app.py를 거치지 않고 ptgaze_server(5001)에 직접 요청하도록 프론트엔드를
# 구성한다 (app.py는 중계 부담을 피하고, 최종 판정 단계에서만 개입).
# 이 엔드포인트는 세션에 쌓인 데이터를 가져와 판정만 수행한다.

@app.route("/analyze/stream/predict", methods=["POST"])
def analyze_stream_predict():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id가 필요합니다."}), 400

    try:
        resp = requests.get(
            f"{PTGAZE_SERVER_URL}/stream/samples",
            params={"session_id": session_id},
            timeout=10,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"시선 추출 서버 오류: {resp.text}"}), 502

        stream_data = resp.json()
        result = predict_from_samples(
            stream_data["samples"], stream_data.get("blink_samples")
        )
        status = result.pop("_status", 200)
        if status == 200:
            result["detection_rate"] = round(stream_data.get("detection_rate", 0) * 100, 1)
        return jsonify(result), status

    except requests.exceptions.ConnectionError:
        return jsonify(
            {"error": "시선 추출 서버에 연결할 수 없습니다. ptgaze_server.py가 실행 중인지 확인해주세요."}
        ), 503
    except Exception as e:
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {e}"}), 500


@app.route("/config/ptgaze-server", methods=["GET"])
def get_ptgaze_server_url():
    """프론트엔드가 ptgaze_server에 직접 프레임을 보낼 수 있도록 URL을 알려준다."""
    return jsonify({"url": PTGAZE_SERVER_URL})


import upload_progress
upload_progress.register(app, predict_from_samples, PTGAZE_SERVER_URL, allowed_file)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
