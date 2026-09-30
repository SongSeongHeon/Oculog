"""
upload_progress.py

업로드 영상 분석의 진행률을 브라우저에 전달하는 엔드포인트. app.py(oculog 환경, 포트 5000)에 붙여서 쓴다.
ptgaze_server의 /extract/start, /extract/progress(upload_jobs.py)를 중계하고,
분석이 끝나면 기존 predict_from_samples로 판정까지 한다. 기존 /analyze/upload는 그대로 둔다.

  POST /analyze/upload/start     영상을 받아 분석 시작 -> {"job_id": ...}
  GET  /analyze/upload/progress  ?job_id=... -> {"state", "progress"} 또는 끝나면 {"state": "done", "result": {...}}

app.py에 붙이는 방법 (파일 맨 아래 `if __name__ == "__main__":` 바로 위에 2줄):
    import upload_progress
    upload_progress.register(app, predict_from_samples, PTGAZE_SERVER_URL, allowed_file)
"""

from __future__ import annotations

import requests
from flask import request, jsonify


def register(app, predict_from_samples, ptgaze_url, allowed_file):
    no_server = ("시선 추출 서버에 연결할 수 없습니다. ptgaze_server.py가 실행 중인지 확인해주세요.")

    @app.route("/analyze/upload/start", methods=["POST"])
    def analyze_upload_start():
        if "video" not in request.files:
            return jsonify({"error": "영상 파일이 없습니다."}), 400
        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
        if not allowed_file(video_file.filename):
            return jsonify({"error": "지원하지 않는 파일 형식입니다 (mp4/avi/mov/mkv만 가능)."}), 400
        try:
            resp = requests.post(
                f"{ptgaze_url}/extract/start",
                files={"video": (video_file.filename, video_file.stream)},
                timeout=120,
            )
            if resp.status_code != 200:
                return jsonify({"error": f"시선 추출 서버 오류: {resp.text}"}), 502
            return jsonify({"job_id": resp.json()["job_id"]})
        except requests.exceptions.ConnectionError:
            return jsonify({"error": no_server}), 503
        except Exception as e:
            return jsonify({"error": f"분석을 시작하지 못했습니다: {e}"}), 500

    @app.route("/analyze/upload/progress", methods=["GET"])
    def analyze_upload_progress():
        job_id = request.args.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id가 필요합니다."}), 400
        try:
            resp = requests.get(f"{ptgaze_url}/extract/progress", params={"job_id": job_id}, timeout=10)
            if resp.status_code == 404:
                return jsonify({"error": "분석 작업을 찾을 수 없습니다."}), 404
            if resp.status_code != 200:
                return jsonify({"error": f"시선 추출 서버 오류: {resp.text}"}), 502
            job = resp.json()
            if job["state"] == "error":
                return jsonify({"error": f"시선 추출 중 오류: {job.get('error')}"}), 502
            if job["state"] != "done":
                return jsonify({"state": job["state"], "progress": job.get("progress"),
                                "frame": job.get("frame"), "total": job.get("total")})

            data = job["result"]
            result = predict_from_samples(data["samples"], data.get("blink_samples"))
            status = result.pop("_status", 200)
            if status == 200:
                result["detection_rate"] = round(data.get("detection_rate", 0) * 100, 1)
                return jsonify({"state": "done", "progress": 1.0, "result": result})
            return jsonify(result), status
        except requests.exceptions.ConnectionError:
            return jsonify({"error": no_server}), 503
        except Exception as e:
            return jsonify({"error": f"분석 중 오류가 발생했습니다: {e}"}), 500
