"""
video_to_gaze_ptgaze.py

MPIIFaceGaze(ptgaze) 기반 영상 -> 시선 시계열 배치 변환기.

기존 video_to_gaze.py(MediaPipe 근사)와 저장 포맷은 동일하게 맞췄다
(time, x, y / 공백구분 / 헤더없음) — preprocessing.py, windowing.py,
dataset.py는 전혀 수정하지 않고 그대로 재사용 가능하다.

ptgaze 명령줄 도구(`python -m ptgaze ...`) 대신 이 스크립트를 쓰는 이유:
  1. 명령줄 도구는 매 프레임 결과 영상을 인코딩해서 저장하는데, 이 인코딩
     시간이 상당 부분을 차지한다. 학습 데이터로는 좌표값만 필요하므로
     시각화/저장 단계를 완전히 생략해 속도를 높인다.
  2. 여러 영상을 하나의 프로세스 안에서 순차 처리해서, 모델(gaze
     estimator)을 매번 새로 로드하는 오버헤드를 없앤다 (명령줄 도구는
     영상 1개당 프로세스를 새로 띄워 모델을 매번 다시 로드했다).

좌표 규약:
  ptgaze의 pitch/yaw(도 단위)를 그대로 x, y로 저장한다.
  x = yaw(좌우 응시각), y = pitch(상하 응시각).
  MediaPipe 버전(-1~1 정규화 비율)과 값의 스케일 자체는 다르지만,
  windowing.py의 compute_features()는 x, y, time 컬럼만 있으면 되므로
  파이프라인 호환에는 문제가 없다. 다만 두 방식으로 만든 데이터를
  같은 모델에 섞어서 학습하면 안 된다 (스케일이 다름) — 반드시
  genuine/fake 데이터 전체를 한 방식으로 통일해서 재생성할 것.

실행 (oculog_ptgaze 환경, WSL2):
    python video_to_gaze_ptgaze.py
"""

from __future__ import annotations

import os
import sys
import time
import glob
import pathlib
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# ptgaze 내부 모듈 (pip install ptgaze==0.2.8 로 설치된 실제 패키지 API 기준)
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.utils import (
    expanduser_all,
    generate_dummy_camera_params,
    download_mpiifacegaze_model,
    check_path_all,
)


def build_config(device: str = "cuda", sample_video_path: str = None):
    """ptgaze 패키지 안의 mpiifacegaze.yaml을 읽어와, 명령줄 도구
    (python -m ptgaze --mode mpiifacegaze --face-detector mediapipe
    --device cuda --video ... --no-screen)와 동일한 설정을 재현한다.

    generate_dummy_camera_params()는 config.demo.video_path(또는
    image_path)에서 프레임 크기를 읽어와 더미 카메라 파라미터를 만든다.
    우리는 Demo 클래스를 거치지 않고 여러 영상을 순회하므로, 카메라
    파라미터 생성을 위해 대표 영상 하나(sample_video_path)를 임시로
    config.demo.video_path에 지정해준다. FaceForensics++ 영상들은
    해상도가 거의 동일(일반적으로 다양한 소스 유튜브 해상도)하므로
    첫 영상 기준으로 계산한 카메라 파라미터를 전체 배치에 재사용한다.
    """
    import ptgaze
    package_root = pathlib.Path(ptgaze.__file__).parent.resolve()

    config_path = package_root / "data/configs/mpiifacegaze.yaml"
    config = OmegaConf.load(config_path)
    config.PACKAGE_ROOT = package_root.as_posix()

    config.face_detector.mode = "mediapipe"
    config.device = device
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
        warnings.warn("CUDA 사용 불가, CPU로 전환합니다.")

    config.gaze_estimator.use_dummy_camera_params = True

    if sample_video_path is None:
        raise ValueError(
            "카메라 파라미터 계산용 대표 영상(sample_video_path)이 필요합니다."
        )
    # generate_dummy_camera_params가 프레임 크기를 읽을 수 있도록 임시로 지정
    config.demo.video_path = sample_video_path

    expanduser_all(config)
    generate_dummy_camera_params(config)
    # 이후 단계에서는 필요 없으므로 원상복구(참고용, 실제 캡처는 우리가 직접 연다)
    config.demo.video_path = None

    # 체크포인트(사전학습 가중치) 자동 다운로드 — 이미 받아져 있으면 스킵됨
    download_mpiifacegaze_model()
    check_path_all(config)

    OmegaConf.set_readonly(config, False)
    return config


@dataclass
class GazeSample:
    time_sec: float
    yaw_deg: float
    pitch_deg: float


class PtgazeExtractor:
    """영상 하나를 받아 프레임별 pitch/yaw를 뽑는다. 모델은 한 번만 로드해서
    여러 영상에 재사용한다 (배치 처리 시 오버헤드 최소화)."""

    def __init__(self, device: str = "cuda", sample_video_path: str = None):
        self.config = build_config(device, sample_video_path=sample_video_path)
        self.gaze_estimator = GazeEstimator(self.config)

    def close(self):
        pass  # GazeEstimator에 close()가 없어 아무 것도 하지 않음 (이전 버전의 버그 수정)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process_frame(self, frame) -> Optional[GazeSample]:
        """단일 프레임(BGR numpy array, 실시간 스트리밍용)을 처리해
        GazeSample 하나를 반환. 얼굴이 없거나 다중 검출되면 None을 반환한다.
        process_video()와 동일한 판정 로직을 프레임 단위로 재사용."""
        undistorted = cv2.undistort(
            frame,
            self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients,
        )
        faces = self.gaze_estimator.detect_faces(undistorted)

        if len(faces) != 1:
            return None

        face = faces[0]
        self.gaze_estimator.estimate_gaze(undistorted, face)
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        return GazeSample(0.0, float(yaw), float(pitch))  # time_sec은 호출부에서 채움

    def process_video(self, video_path: str) -> dict:
        """영상 하나를 처리해 통계와 GazeSample 리스트를 반환.

        다중 얼굴이 검출된 프레임은 시선 주체가 불확실하므로 건너뛴다
        (예: 뉴스 인터뷰처럼 여러 사람이 동시에 화면에 나오는 영상).
        얼굴이 아예 없는 프레임도 건너뛴다.

        반환값의 num_faces_seen(프레임별 검출 얼굴 수 로그)으로 호출부에서
        이 영상을 최종 채택할지 판단한다 (process_folder의 필터링 기준).
        """
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
                self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients,
            )
            faces = self.gaze_estimator.detect_faces(undistorted)

            if len(faces) == 0:
                no_face_frames += 1
            elif len(faces) >= 2:
                # 여러 얼굴이 동시에 검출됨 -> 시선 주체가 불확실하므로 건너뜀
                multi_face_frames += 1
            else:
                face = faces[0]
                self.gaze_estimator.estimate_gaze(undistorted, face)
                pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
                t = frame_idx / fps
                samples.append(GazeSample(t, float(yaw), float(pitch)))

            frame_idx += 1

        cap.release()

        return {
            "samples": samples,
            "total_frames": total_frames,
            "no_face_frames": no_face_frames,
            "multi_face_frames": multi_face_frames,
        }


def save_samples(samples: list, output_path: str) -> None:
    """time x y 형식(공백구분, 헤더없음)으로 저장. x=yaw, y=pitch."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(f"{s.time_sec:.4f} {s.yaw_deg:.6f} {s.pitch_deg:.6f}\n")


def process_folder(
    extractor: PtgazeExtractor,
    input_folder: str,
    output_folder: str,
    skip_existing: bool = True,
    min_detection_rate: float = 0.8,
    max_multiface_rate: float = 0.05,
    min_samples: int = 30,
) -> dict:
    """영상 폴더를 배치 처리하되, 품질 기준에 못 미치는 영상은 제외한다.

    제외 기준 (오판 사례 분석 결과 반영, 464/488/447 등):
      - min_detection_rate: 전체 프레임 대비 얼굴이 검출된 비율이 이보다
        낮으면 제외 (얼굴이 자주 화면 밖으로 나가거나 가려지는 영상)
      - max_multiface_rate: 다중 얼굴 검출 프레임 비율이 이보다 높으면 제외
        (여러 사람이 동시에 나오는 뉴스/인터뷰 영상 등 - 464, 447 사례)
      - min_samples: 최종 유효 샘플 수가 이보다 적으면 제외 (윈도우를
        채우기에 너무 짧은 시퀀스)
    """
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

        t0 = time.time()
        try:
            result = extractor.process_video(video_path)
            total = result["total_frames"]
            samples = result["samples"]

            detection_rate = len(samples) / total if total else 0.0
            multiface_rate = result["multi_face_frames"] / total if total else 0.0

            if (
                detection_rate < min_detection_rate
                or multiface_rate > max_multiface_rate
                or len(samples) < min_samples
            ):
                elapsed = time.time() - t0
                print(
                    f"[{i}/{n}] [제외] {basename}.mp4: "
                    f"검출률={detection_rate:.0%}, 다중얼굴률={multiface_rate:.0%}, "
                    f"샘플={len(samples)}개, {elapsed:.1f}초"
                )
                stats["filtered"] += 1
                continue

            save_samples(samples, output_path)
            elapsed = time.time() - t0
            print(
                f"[{i}/{n}] {basename}.mp4 -> {len(samples)} 샘플 "
                f"(검출률 {detection_rate:.0%}), {elapsed:.1f}초"
            )
            stats["success"] += 1
        except Exception as e:
            print(f"[{i}/{n}] [실패] {basename}.mp4: {e}")
            stats["failed"] += 1

    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    genuine_input = "faceforensics/original_sequences/youtube/c23/videos"
    fake_input = "faceforensics/manipulated_sequences/FaceShifter/c23/videos"
    genuine_output = "../data/deepfake_gaze_ptgaze/genuine"
    fake_output = "../data/deepfake_gaze_ptgaze/fake"
    # 주의: MediaPipe 버전 결과와 섞이지 않도록 별도 폴더(deepfake_gaze_ptgaze)에 저장.
    # config.py의 GENUINE_DIR/FAKE_DIR을 이 경로로 갱신해야 dataset.py가 이 데이터를 읽는다.

    # 카메라 파라미터 계산용 대표 영상 (원본 폴더에서 첫 번째 mp4 사용)
    sample_candidates = sorted(glob.glob(os.path.join(genuine_input, "*.mp4")))
    if not sample_candidates:
        raise RuntimeError(f"{genuine_input} 안에 mp4가 없습니다.")
    sample_video_path = sample_candidates[0]
    print(f"카메라 파라미터 기준 영상: {sample_video_path}")

    with PtgazeExtractor(device=device, sample_video_path=sample_video_path) as extractor:
        print("\n=== GENUINE(원본) 처리 시작 ===")
        t0 = time.time()
        stats_g = process_folder(extractor, genuine_input, genuine_output)
        print(f"GENUINE 완료: {stats_g}, 소요 {time.time()-t0:.1f}초")

        print("\n=== FAKE(FaceShifter) 처리 시작 ===")
        t0 = time.time()
        stats_f = process_folder(extractor, fake_input, fake_output)
        print(f"FAKE 완료: {stats_f}, 소요 {time.time()-t0:.1f}초")


if __name__ == "__main__":
    main()
