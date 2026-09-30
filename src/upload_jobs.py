"""
upload_jobs.py

업로드 영상 분석을 "작업(job)"으로 돌리면서 실제 진행률을 알려주는 엔드포인트.
ptgaze_server.py(oculog_ptgaze 환경, 포트 5001)에 붙여서 쓴다.

기존 /extract는 그대로 두고(롤백 가능) 아래 두 개만 추가한다.
  POST /extract/start     영상 파일을 받아 백그라운드로 분석 시작 -> {"job_id": ...}
  GET  /extract/progress  ?job_id=... -> {"state", "progress"(0~1), "frame", "total", "result"|"error"}

기존 /extract는 영상을 두 번 읽는다(시선 한 번, 깜빡임 한 번). 여기서는 한 번만 읽으면서
시선과 EAR를 같이 뽑는다. 프레임마다 하는 일(undistort -> 얼굴 검출 -> 1명이면 시선/EAR 기록)은
process_video / extract_ear_from_video와 같다. 결과 JSON 형식도 /extract와 같다.

ptgaze_server.py에 붙이는 방법 (파일 맨 아래 `if __name__ == "__main__":` 바로 위에 2줄):
    import upload_jobs
    upload_jobs.register(app, get_extractor, _extractor_lock)
"""

from __future__ import annotations

import os
import time
import uuid
import tempfile
import threading

import cv2
import numpy as np
from flask import request, jsonify

import blink_extractor

JOB_TTL_SEC = 600           # 끝난 작업 결과를 보관하는 시간
PROGRESS_EVERY_N_FRAMES = 5

_jobs = {}
_jobs_lock = threading.Lock()


def process_video_both(estimator, video_path, on_progress=None):
    """한 번의 패스로 시선(yaw, pitch)과 EAR를 뽑는다. /extract와 같은 형식의 dict 반환."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    gaze, blink = [], []
    frame_idx = total = multi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        undistorted = cv2.undistort(
            frame, estimator.camera.camera_matrix, estimator.camera.dist_coefficients
        )
        faces = estimator.detect_faces(undistorted)
        if len(faces) >= 2:
            multi += 1
        elif len(faces) == 1:
            face = faces[0]
            estimator.estimate_gaze(undistorted, face)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            t = frame_idx / fps
            gaze.append([t, float(yaw), float(pitch)])
            blink.append([t, blink_extractor.compute_ear(face.landmarks)])
        frame_idx += 1
        if on_progress and frame_idx % PROGRESS_EVERY_N_FRAMES == 0:
            on_progress(frame_idx, frame_count)
    cap.release()

    return {
        "samples": gaze,
        "blink_samples": blink,
        "total_frames": total,
        "detection_rate": len(gaze) / total if total else 0.0,
        "multiface_rate": multi / total if total else 0.0,
    }


def _cleanup_old_jobs():
    now = time.time()
    with _jobs_lock:
        old = [j for j, v in _jobs.items()
               if v["state"] in ("done", "error") and now - v["finished"] > JOB_TTL_SEC]
        for j in old:
            del _jobs[j]


def _run_job(job_id, tmp_path, get_extractor, extractor_lock):
    def on_progress(frame, total):
        with _jobs_lock:
            job = _jobs[job_id]
            job["frame"], job["total"] = frame, total
            # 영상 길이를 모르면(total<=0) progress는 None으로 두고 프런트가 진행 중 표시만 한다
            job["progress"] = min(0.99, frame / total) if total > 0 else None

    try:
        with extractor_lock:  # 모델은 한 번에 하나만 (스트리밍 요청과도 공유)
            with _jobs_lock:
                _jobs[job_id]["state"] = "running"
            extractor = get_extractor(sample_source=tmp_path)
            result = process_video_both(extractor.gaze_estimator, tmp_path, on_progress)
        with _jobs_lock:
            _jobs[job_id].update(state="done", progress=1.0, result=result, finished=time.time())
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update(state="error", error=str(e), finished=time.time())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def register(app, get_extractor, extractor_lock):
    @app.route("/extract/start", methods=["POST"])
    def extract_start():
        if "video" not in request.files:
            return jsonify({"error": "video 파일이 없습니다."}), 400
        video_file = request.files["video"]
        suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        _cleanup_old_jobs()
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"state": "queued", "progress": 0.0, "frame": 0, "total": 0,
                             "result": None, "error": None, "finished": 0.0}
        threading.Thread(target=_run_job, args=(job_id, tmp_path, get_extractor, extractor_lock),
                         daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/extract/progress", methods=["GET"])
    def extract_progress():
        job_id = request.args.get("job_id")
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return jsonify({"error": "유효하지 않은 job_id입니다."}), 404
            out = {"state": job["state"], "progress": job["progress"],
                   "frame": job["frame"], "total": job["total"]}
            if job["state"] == "done":
                out["result"] = job["result"]
            if job["state"] == "error":
                out["error"] = job["error"]
        return jsonify(out)
