"""
blink_extractor.py

ptgaze가 검출한 468개 mediapipe Face Mesh 랜드마크에서 EAR(Eye Aspect
Ratio, 눈 개폐 비율) 시계열을 추출한다. 시선(gaze) 파이프라인과는 완전히
독립적으로 동작하며, video_to_gaze_ptgaze.py의 PtgazeExtractor가 이미
내부적으로 계산해둔 face.landmarks(468, 2)를 재사용한다.

배경: 딥페이크는 깜빡임 빈도/패턴이 부자연스러운 경우가 많다는 것이
선행 연구(Li et al., 2018 등)에서 보고됨. 시선 신호만으로는 웹캠 도메인
오탐 문제가 해결되지 않아, 보조 신호로 깜빡임을 추가해 앙상블하는
방향을 시도한다.

EAR 공식 (Soukupová & Čech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p4: 눈 양쪽 끝 (수평)
    p2, p3, p5, p6: 위아래 눈꺼풀 (수직, 두 지점씩)

여기서는 mediapipe 공식 눈 랜드마크(FACEMESH_LEFT_EYE/RIGHT_EYE) 중
수평 양끝 2점 + 수직 위아래 2쌍(4점)을 뽑아 표준 6점 EAR로 계산한다.
좌우 눈 EAR의 평균을 최종 EAR 값으로 사용한다.

저장 포맷은 시선 데이터와 동일하게 (time, ear) 2컬럼, 공백구분,
헤더없음으로 맞춰서 windowing.py를 그대로 재사용할 수 있게 한다
(다만 windowing.compute_features는 x, y 2개 컬럼을 가정하므로, 깜빡임은
1채널이라 별도의 간단한 feature 계산 함수를 이 파일에 둔다).
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import numpy as np


# mediapipe 공식 FACEMESH_LEFT_EYE / FACEMESH_RIGHT_EYE 인덱스 집합에서
# 표준 6점 EAR 계산에 쓸 지점을 선정 (양끝 수평 2점 + 위아래 수직 2쌍).
# gaze_extractor.py(초기 MediaPipe 버전)에서 이미 검증했던 것과 동일한
# 인덱스 조합을 재사용한다 — 468개 랜드마크 순서가 ptgaze도 동일하므로
# 그대로 적용 가능함 (video_to_gaze_ptgaze.py 실측으로 468개 확인됨).
LEFT_EYE_CORNERS = (362, 263)      # (안쪽, 바깥쪽)
LEFT_EYE_TOP_BOTTOM = (386, 374)   # (위, 아래)

RIGHT_EYE_CORNERS = (33, 133)      # (바깥쪽, 안쪽)
RIGHT_EYE_TOP_BOTTOM = (159, 145)  # (위, 아래)


def _eye_aspect_ratio(landmarks: np.ndarray, corners: tuple, top_bottom: tuple) -> float:
    """단일 눈의 EAR을 계산. landmarks는 (468, 2) 형태."""
    c1 = landmarks[corners[0]]
    c2 = landmarks[corners[1]]
    t = landmarks[top_bottom[0]]
    b = landmarks[top_bottom[1]]

    eye_width = np.linalg.norm(c2 - c1)
    eye_height = np.linalg.norm(b - t)

    if eye_width < 1e-6:
        return 0.0
    return float(eye_height / eye_width)


def compute_ear(landmarks: np.ndarray) -> float:
    """양쪽 눈 EAR의 평균을 반환."""
    left = _eye_aspect_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_TOP_BOTTOM)
    right = _eye_aspect_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_TOP_BOTTOM)
    return (left + right) / 2.0


@dataclass
class BlinkSample:
    time_sec: float
    ear: float


def extract_ear_from_video(extractor, video_path: str) -> dict:
    """PtgazeExtractor(video_to_gaze_ptgaze.py)의 gaze_estimator를 재사용해
    영상 하나에서 EAR 시계열을 뽑는다. 시선 필터링 로직(다중얼굴 제외 등)과
    동일한 기준을 적용해 일관성을 유지한다."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    samples = []
    frame_idx = 0
    total_frames = 0
    no_face_frames = 0
    multi_face_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        undistorted = cv2.undistort(
            frame,
            extractor.gaze_estimator.camera.camera_matrix,
            extractor.gaze_estimator.camera.dist_coefficients,
        )
        faces = extractor.gaze_estimator.detect_faces(undistorted)

        if len(faces) == 0:
            no_face_frames += 1
        elif len(faces) >= 2:
            multi_face_frames += 1
        else:
            face = faces[0]
            ear = compute_ear(face.landmarks)
            t = frame_idx / fps
            samples.append(BlinkSample(t, ear))

        frame_idx += 1

    cap.release()

    return {
        "samples": samples,
        "total_frames": total_frames,
        "no_face_frames": no_face_frames,
        "multi_face_frames": multi_face_frames,
    }


def save_ear_samples(samples: list, output_path: str) -> None:
    """time ear 형식(공백구분, 헤더없음)으로 저장."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(f"{s.time_sec:.4f} {s.ear:.6f}\n")


def compute_blink_features(df) -> np.ndarray:
    """
    EAR 시계열(time, ear)에서 windowing.py의 compute_features()와
    유사한 방식으로 특징을 계산한다. EAR 값 자체 + EAR 변화율(velocity)을
    2채널로 구성해, 시선 파이프라인의 (T, 3)과 유사한 (T, 2) 형태로 맞춘다.

    df: pandas.DataFrame, 컬럼 ["time", "ear"]
    반환: (T, 2) numpy array — [ear, ear_velocity]
    """
    ear = df["ear"].values
    t = df["time"].values

    dt = np.diff(t, prepend=t[0])
    dt[dt == 0] = 1e-6

    d_ear = np.diff(ear, prepend=ear[0])
    ear_velocity = d_ear / dt

    features = np.stack([ear, ear_velocity], axis=1)
    return features


def process_folder(extractor, input_folder: str, output_folder: str,
                    skip_existing: bool = True,
                    min_detection_rate: float = 0.8,
                    max_multiface_rate: float = 0.05,
                    min_samples: int = 30) -> dict:
    """video_to_gaze_ptgaze.process_folder()와 동일한 필터링 기준을 적용해
    폴더 단위로 EAR 시계열을 배치 추출한다."""
    video_paths = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    if not video_paths:
        print(f"[경고] {input_folder} 안에서 mp4를 찾지 못했습니다.")
        return {"success": 0, "skipped": 0, "failed": 0, "filtered": 0}

    stats = {"success": 0, "skipped": 0, "failed": 0, "filtered": 0}
    n = len(video_paths)

    for i, video_path in enumerate(video_paths, 1):
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_folder, f"{basename}.txt")

        if skip_existing and os.path.exists(output_path):
            stats["skipped"] += 1
            continue

        try:
            result = extract_ear_from_video(extractor, video_path)
            total = result["total_frames"]
            samples = result["samples"]

            detection_rate = len(samples) / total if total else 0.0
            multiface_rate = result["multi_face_frames"] / total if total else 0.0

            if (
                detection_rate < min_detection_rate
                or multiface_rate > max_multiface_rate
                or len(samples) < min_samples
            ):
                print(
                    f"[{i}/{n}] [제외] {basename}.mp4: "
                    f"검출률={detection_rate:.0%}, 다중얼굴률={multiface_rate:.0%}, "
                    f"샘플={len(samples)}개"
                )
                stats["filtered"] += 1
                continue

            save_ear_samples(samples, output_path)
            print(
                f"[{i}/{n}] {basename}.mp4 -> {len(samples)} EAR 샘플 "
                f"(검출률 {detection_rate:.0%})"
            )
            stats["success"] += 1
        except Exception as e:
            print(f"[{i}/{n}] [실패] {basename}.mp4: {e}")
            stats["failed"] += 1

    return stats
