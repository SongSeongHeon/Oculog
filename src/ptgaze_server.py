"""
ptgaze_server.py

시선 추출 전용 내부 API 서버. oculog_ptgaze 환경(WSL2)에서 실행한다.

tensorflow(protobuf>=6.31)와 ptgaze/mediapipe(protobuf<5)가 같은 프로세스에
공존할 수 없어서, 웹 서비스를 두 프로세스로 분리했다. 이 서버는 두 가지
모드를 지원한다:

  1. 업로드 모드 (/extract): 영상 파일 하나를 통째로 받아 시선 시계열 반환
  2. 스트리밍 모드 (/stream/*): 실시간 웹캠 프레임을 하나씩 받아 세션별로
     누적하고, 필요할 때 지금까지 모인 시선 시계열을 반환

모델 예측, 화면(UI)은 전혀 다루지 않는다 — 그건 app.py(메인 서버, oculog
환경)의 역할이다.

실행 (oculog_ptgaze 환경, WSL2):
    python ptgaze_server.py
    -> http://localhost:5001 에서 대기
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import tempfile
import threading

import numpy as np
import cv2
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from video_to_gaze_ptgaze import PtgazeExtractor, GazeSample
import blink_extractor

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # 브라우저가 localhost:5000(app.py)에서 로드된 페이지에서 이 서버
    # (localhost:5001)로 직접 fetch 요청을 보내므로 크로스 오리진 허용이 필요하다.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/stream/start", methods=["OPTIONS"])
@app.route("/stream/frame", methods=["OPTIONS"])
@app.route("/stream/samples", methods=["OPTIONS"])
@app.route("/stream/reset", methods=["OPTIONS"])
@app.route("/stream/end", methods=["OPTIONS"])
def handle_options():
    return "", 204

_extractor = None
_extractor_lock = threading.Lock()  # 여러 요청이 동시에 모델을 건드리지 않도록

# 스트리밍 세션 저장소: {session_id: {"samples": [...], "start_time": ..., "last_seen": ...}}
_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TIMEOUT_SEC = 60  # 이 시간 동안 프레임이 안 오면 세션을 정리


def get_extractor(sample_source=None):
    """모델을 한 번만 로드해서 재사용. 스트리밍은 카메라 파라미터 계산을
    위한 대표 이미지를 sample_source(numpy 프레임)로 받을 수도 있다."""
    global _extractor
    if _extractor is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ptgaze_server] 모델 로드 중 (device={device})...")

        if isinstance(sample_source, str):
            _extractor = PtgazeExtractor(device=device, sample_video_path=sample_source)
        else:
            tmp_path = _frame_to_temp_video(sample_source)
            try:
                _extractor = PtgazeExtractor(device=device, sample_video_path=tmp_path)
            finally:
                os.remove(tmp_path)
    return _extractor


def _frame_to_temp_video(frame: np.ndarray) -> str:
    """단일 프레임으로 1프레임짜리 임시 mp4를 만든다 (카메라 파라미터
    계산용, generate_dummy_camera_params가 video_path만 받으므로)."""
    h, w = frame.shape[:2]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    writer.write(frame)
    writer.release()
    return path


def _cleanup_stale_sessions():
    now = time.time()
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items() if now - s["last_seen"] > SESSION_TIMEOUT_SEC]
        for sid in stale:
            del _sessions[sid]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _extractor is not None})


# ── 업로드 모드 ────────────────────────────────────────────

@app.route("/extract", methods=["POST"])
def extract():
    if "video" not in request.files:
        return jsonify({"error": "video 파일이 없습니다."}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with _extractor_lock:
            extractor = get_extractor(sample_source=tmp_path)
            result = extractor.process_video(tmp_path)
            blink_result = blink_extractor.extract_ear_from_video(extractor, tmp_path)

        total = result["total_frames"]
        samples = result["samples"]
        detection_rate = len(samples) / total if total else 0.0
        multiface_rate = result["multi_face_frames"] / total if total else 0.0
        samples_list = [[s.time_sec, s.yaw_deg, s.pitch_deg] for s in samples]

        blink_samples_list = [[s.time_sec, s.ear] for s in blink_result["samples"]]

        return jsonify(
            {
                "samples": samples_list,
                "blink_samples": blink_samples_list,
                "total_frames": total,
                "detection_rate": detection_rate,
                "multiface_rate": multiface_rate,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 스트리밍 모드 ──────────────────────────────────────────

@app.route("/stream/start", methods=["POST"])
def stream_start():
    """새 스트리밍 세션을 시작하고 세션 ID를 발급한다."""
    _cleanup_stale_sessions()
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = {
            "samples": [],
            "blink_samples": [],
            "start_time": time.time(),
            "last_seen": time.time(),
            "frame_count": 0,
        }
    return jsonify({"session_id": session_id})


@app.route("/stream/frame", methods=["POST"])
def stream_frame():
    """프레임 하나를 받아 세션 버퍼에 시선 좌표를 누적한다.
    요청 형식: multipart/form-data, 필드 "session_id"(텍스트), "frame"(이미지 blob)
    """
    session_id = request.form.get("session_id")
    if session_id not in _sessions:
        return jsonify({"error": "유효하지 않은 세션입니다. /stream/start를 먼저 호출하세요."}), 404

    if "frame" not in request.files:
        return jsonify({"error": "frame 이미지가 없습니다."}), 400

    file_bytes = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "이미지를 디코딩하지 못했습니다."}), 400

    try:
        with _extractor_lock:
            extractor = get_extractor(sample_source=frame)
            sample = extractor.process_frame(frame)

            # 시선과 같은 프레임에서 EAR도 계산 (별도 검출 없이, 동일한
            # undistort+detect_faces 로직을 한 번 더 실행 — process_frame이
            # 내부적으로 얼굴 검출 결과를 반환하지 않는 구조라 재사용 대신
            # 별도 호출한다. 실시간 부담이 커지면 이후 process_frame 자체를
            # (gaze, ear) 튜플을 함께 반환하도록 확장하는 최적화 고려 가능)
            ear_value = None
            undistorted = cv2.undistort(
                frame,
                extractor.gaze_estimator.camera.camera_matrix,
                extractor.gaze_estimator.camera.dist_coefficients,
            )
            faces = extractor.gaze_estimator.detect_faces(undistorted)
            if len(faces) == 1:
                ear_value = blink_extractor.compute_ear(faces[0].landmarks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with _sessions_lock:
        session = _sessions[session_id]
        session["last_seen"] = time.time()
        session["frame_count"] += 1
        if sample is not None:
            t = time.time() - session["start_time"]
            session["samples"].append([t, sample.yaw_deg, sample.pitch_deg])
        if ear_value is not None:
            t = time.time() - session["start_time"]
            session.setdefault("blink_samples", []).append([t, ear_value])

        n_samples = len(session["samples"])
        n_frames = session["frame_count"]

    return jsonify({
        "face_detected": sample is not None,
        "samples_collected": n_samples,
        "frames_received": n_frames,
    })


@app.route("/stream/samples", methods=["GET"])
def stream_samples():
    """지금까지 세션에 쌓인 시선 시계열 전체를 반환한다 (판정용)."""
    session_id = request.args.get("session_id")
    if session_id not in _sessions:
        return jsonify({"error": "유효하지 않은 세션입니다."}), 404

    with _sessions_lock:
        session = _sessions[session_id]
        samples = list(session["samples"])
        blink_samples = list(session.get("blink_samples", []))
        n_frames = session["frame_count"]

    detection_rate = len(samples) / n_frames if n_frames else 0.0

    return jsonify({
        "samples": samples,
        "blink_samples": blink_samples,
        "total_frames": n_frames,
        "detection_rate": detection_rate,
    })


@app.route("/stream/reset", methods=["POST"])
def stream_reset():
    """세션의 샘플 버퍼를 비운다 (슬라이딩 윈도우 갱신용)."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.args.get("session_id")
    if session_id not in _sessions:
        return jsonify({"error": "유효하지 않은 세션입니다."}), 404

    with _sessions_lock:
        _sessions[session_id]["samples"] = []
        _sessions[session_id]["blink_samples"] = []
        _sessions[session_id]["frame_count"] = 0
        _sessions[session_id]["start_time"] = time.time()

    return jsonify({"status": "reset"})


@app.route("/stream/end", methods=["POST"])
def stream_end():
    """세션을 완전히 종료하고 정리한다."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.args.get("session_id")
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return jsonify({"status": "ended"})


import upload_jobs
upload_jobs.register(app, get_extractor, _extractor_lock)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
