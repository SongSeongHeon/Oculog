# 오큘로그(Oculog)

시선(gaze)과 눈 깜빡임(EAR) 시계열 패턴으로 화상통화 딥페이크를 판별하는 웹 데모.

웹캠 수준의 단일 카메라 영상만으로 상대방이 실제 사람인지 딥페이크인지 판별하는 것을 목표로 한다.
얼굴의 픽셀 아티팩트가 아니라 **눈의 움직임 패턴**이라는 생체 신호에 의존한다.

이 저장소는 **웹 데모를 실행하는 데 필요한 파일만** 담고 있다. (학습/평가 코드와 데이터셋은 포함하지 않는다.)

## 접근 방법

Kohler et al., *"DeepFake Detection in Dyadic Video Calls using Point of Gaze Tracking"*
(IEEE Access, 2026, [arXiv:2509.25503](https://arxiv.org/abs/2509.25503))의 핵심 아이디어를 계승했다.
초기에는 MediaPipe 기반 시선 근사를 썼으나, 오판 사례 분석 후 원 논문과 같은 **MPIIFaceGaze(ptgaze) 기반 시선 추정**으로
전환했고, 보조 신호로 **눈 깜빡임(EAR)** 을 결합했다.

```
영상/웹캠 ─ ptgaze ─→ 시선 시계열(time, yaw, pitch) ─┐
          └ 얼굴 랜드마크 ─→ 깜빡임 시계열(time, EAR) ─┤
                                                      ▼
   30Hz 리샘플링 → 1초 윈도우 18개(0.5초 stride) → z-score → CNN-LSTM
                                                      ▼
            시선 모델 점수 × 0.9 + 깜빡임 모델 점수 × 0.1 → 0.5 이상이면 딥페이크
```

- 입력 shape: 시선 `(18, 30, 3)`(x, y, 속도), 깜빡임 `(18, 30, 2)`(EAR, 변화율). 10초 분량이 필요하다.
- 라벨: 0 = 실제, 1 = 딥페이크.
- 학습 데이터: FaceForensics++(원본 + FaceShifter)에 웹캠/화상면접 영상을 보강. 참가자 단위로 train/test를 분리했다.

## 성능

### 내부 검증 (FaceForensics++ test set, 현재 저장된 모델)

| 모델 | AUC | 비고 |
|---|---|---|
| 시선 CNN-LSTM | **0.866** | 정확도 77.30%(임계값 0.5) / 80.54%(EER 임계값 0.085), EER 19.47% |
| 깜빡임 CNN-LSTM | 0.775 | |
| 앙상블 (0.9 : 0.1) | **0.885** | (subject, label) 단위로 다시 집계한 값 |

원 논문의 정확도 82%, AUC 0.88과 비슷한 수준이다.
(웹캠 영상을 추가해 재학습하기 전 모델은 정확도 82.16%, AUC 0.882였다.)

### 외부 검증 (학습에 쓰지 않은 데이터, 재학습·임계값 조정 없음)

FaceForensics++로만 학습한 모델을 다른 데이터셋에 그대로 적용했다. 정규화는 학습 때와 같은 통계를 썼고,
임계값은 내부 검증에서 정한 값(시선 0.085)과 서비스 기준(0.5)으로 **고정**했다.

| 데이터셋 | 대상 | AUC (95% 신뢰구간) |
|---|---|---|
| Celeb-DF v2 공식 테스트셋 (진짜 167 / 가짜 327) | 시선 | **0.515** (0.461~0.571) — 우연 수준 |
| VCF 화상회의 벤치마크, c23 압축·1080p (진짜 155 / 가짜 617) | 시선 | **0.596** (0.569~0.626) |
| | 깜빡임 | 0.527 (0.486~0.564) — 신호 없음 |
| | 앙상블 | 0.594 (0.564~0.625) — 내부에서 본 이득이 재현되지 않음 |

VCF는 Deep-Live-Cam(실시간 웹캠 얼굴 교체)과 SimSwap으로 만든 영상이다. 신뢰구간은 같은 클립이 진짜 1개·가짜 4개로
반복되는 구조를 반영해 클립 단위로 재표집했다.

- 교체 방식별 시선 AUC: deeplivecam_enhance 0.636, deeplivecam 0.615, simswap_224 0.581, simswap_512 0.550.
  같은 클립의 진짜/가짜를 짝지어 비교하면 Deep-Live-Cam 계열에서 가짜 점수가 더 높은 비율이 72~76%(Wilcoxon p<0.001)였다.
- 서비스 임계값 0.5에서 VCF 정확도는 60.8%, 균형정확도 56.6%였고 **진짜 영상의 약 절반(49.7%만 통과)이 가짜로 오판**됐다.

**해석.** 내부 AUC 0.87이 외부에서는 0.52~0.60으로 떨어졌다. 내부 성능 재현(0.866), fps 일치, 정규화 방식 변경(AUC 0.491)으로
구현 오류는 아님을 확인했다. 도메인 탐지 실험(FF++와 Celeb-DF를 구분하는 AUC 0.838)과 FF++에서 학습한 단순 특징이 Celeb-DF로
전이되지 않는 결과(AUC 0.489)는 **학습 데이터 특유의 신호를 배웠을 가능성**을 시사한다. 다만 이는 가설이며 이 실험만으로는 확정할 수 없다.
현재 모델은 실사용 수준이 아니다.

### 실사용 테스트

- 본인 웹캠(리샘플링 수정 후): 업로드 영상은 90% 이상, 실시간은 76~99% 확률로 genuine 판정.
- 학습에 쓴 FaceShifter 영상 1건: FAKE 95.8%로 정상 검출.
- 학습에 없던 딥페이크 데모 영상 1건: REAL 99.5%로 오판. (영상 1건이라 일반화 실패로 단정하지 않는다.)

## 시행착오 기록

| 시도 | 결과 | 판정 |
|---|---|---|
| 얼굴 품질 자동 필터 (다중 얼굴 5% 초과 / 검출률 80% 미만 / 유효 샘플 30개 미만 영상 제외) | 정확도 73.5% → 83.78%, AUC 0.828 → 0.870 | ✅ 채택 |
| 눈 깜빡임(EAR) 앙상블 결합 | AUC 0.8664 → 0.8850 (내부). 외부 검증에서는 재현되지 않음 | ⚠️ 내부만 효과 |
| 데이터 증강 (노이즈/시간왜곡/스케일) | 정확도 83.78% → 80.54% 하락 | ❌ 폐기 |
| 뉴스·브이로그 등 "정면 응시" 29명 제외 | 정확도 83.78% → 79.19% 하락 (화상통화 자체가 정면 응시 상황이라 목표 도메인까지 제거됨) | ❌ 폐기 |
| 유튜브 화상면접 17개 genuine 보강 | test 정확도 유지, 실사용 오탐은 개선 안 됨 | ⚠️ 부분 효과 |
| 실시간 입력 30Hz 리샘플링 | 실시간 입력(초당 5프레임)과 학습(30fps)의 밀도 불일치로 생긴 오판(최대 99.6%)을 해소 | ✅ 채택 |

## 실행 방법

conda 환경이 **두 개** 필요하다. ptgaze/mediapipe가 요구하는 protobuf<5 와 tensorflow가 요구하는 protobuf>=6.31 범위가 겹치지 않아
한 환경에 설치할 수 없기 때문이다.

```bash
# 1) 시선/깜빡임 추출 환경 (Python 3.9)
conda create -n oculog_ptgaze python=3.9 -y
conda activate oculog_ptgaze
pip install -r requirements-ptgaze.txt
python src/patch_ptgaze.py        # ptgaze 구버전 API 호환 패치 (설치 후 1회)

# 2) 웹 서버/예측 환경 (Python 3.11)
conda create -n oculog python=3.11 -y
conda activate oculog
pip install -r requirements-oculog.txt
```

- `patch_ptgaze.py`는 ptgaze 0.2.8이 최신 torch/numpy/scipy와 맞지 않는 부분(torchvision 가중치 API, `np.float` 등 삭제된 타입,
  OpenCV 회전 벡터 shape)을 고친다. 여러 번 실행해도 안전하지만 ptgaze를 재설치하면 다시 실행해야 한다.
- **첫 실행 때 인터넷이 필요하다.** ptgaze가 사전학습 가중치(MPIIFaceGaze, torchvision ResNet)를 내려받는다.
- GPU가 없어도 CPU로 동작한다. (CPU에서는 분석이 느리다.)

### 웹 데모 실행 (터미널 2개, 이 순서대로)

```bash
# [터미널 1] 시선+깜빡임 추출 서버  (http://localhost:5001)
conda activate oculog_ptgaze
cd src
python ptgaze_server.py

# [터미널 2] 메인 웹 서버  (http://localhost:5000)
conda activate oculog
cd src
python app.py
```

브라우저에서 `http://localhost:5000`에 접속한다.

| 탭 | 동작 |
|---|---|
| 실시간 통화 | 웹캠에서 0.2초마다 프레임을 받아 10초 분량이 모이면 판정 |
| 화면 공유 | Zoom/Teams 화면을 공유하고, 상대방 얼굴 영역을 드래그로 지정해 판정 |
| 영상 업로드 | 영상을 올리면 미리보기와 **실제 분석 진행률**을 보여주고 판정 (mp4/avi/mov/mkv, 최대 200MB) |

판정이 끝나면 오른쪽 패널에 최종 판정·신뢰도·분석 프레임 수·얼굴 검출률과 시선/깜빡임 개별 점수가 표시된다.

## 폴더 구조

```
oculog/
├── models/
│   ├── oculog_cnn_lstm.keras          # 시선 모델 (약 1.8MB)
│   └── oculog_blink_cnn_lstm.keras    # 깜빡임 모델 (약 1.8MB)
├── src/
│   ├── app.py                   # 메인 웹 서버 (oculog 환경, 포트 5000): UI, 리샘플링, 앙상블 판정
│   ├── upload_progress.py       # 업로드 분석 진행률 중계 엔드포인트
│   ├── ptgaze_server.py         # 시선/깜빡임 추출 서버 (oculog_ptgaze 환경, 포트 5001)
│   ├── upload_jobs.py           # 업로드 분석 작업/진행률 엔드포인트
│   ├── video_to_gaze_ptgaze.py  # ptgaze 기반 시선 추출기
│   ├── blink_extractor.py       # EAR(눈 개폐 비율) 계산
│   ├── windowing.py             # 특징 추출 + 윈도우 분할
│   ├── config.py, config_blink.py   # 샘플링·윈도우 파라미터 (모델 입력 shape과 맞아야 하므로 바꾸지 말 것)
│   ├── patch_ptgaze.py          # ptgaze 호환 패치 (설치 후 1회)
│   └── templates/index.html     # 웹 UI
├── requirements-oculog.txt      # 웹 서버/예측 환경
└── requirements-ptgaze.txt      # 시선/깜빡임 추출 환경
```

## 알려진 한계

- **일반화**: 위 외부 검증 결과처럼 학습에 쓰지 않은 데이터에서는 성능이 크게 떨어진다. 고품질 얼굴 교체는 원본의 눈 움직임을 대부분
  보존하기 때문일 수 있다(가설). VCF는 c23·1080p 한 조건만 평가했고, 저해상도·강한 압축 조건은 검증하지 못했다.
- **오탐**: 서비스 임계값(0.5)에서 진짜 영상도 상당수 가짜로 판정한다. 판정 애매 구간을 두는 것이 필요하다.
- **정규화 근사**: 웹 데모는 학습 때 쓴 전체 데이터셋 통계 대신 요청마다 해당 샘플 자체의 통계로 정규화한다.
- **도메인 불일치**: FaceForensics++는 유튜브 압축 영상 기반이라 웹캠 원본과 조건이 다르다.
- **인구통계 편향**: 연령·성별·인종에 따른 성능 차이는 검증하지 않았다.

## 데이터

FaceForensics++, Celeb-DF v2, VCF 등 데이터셋은 이용약관상 재배포가 불가하거나 용량이 커서 포함하지 않는다.
각 데이터셋은 공식 배포처에서 신청 후 내려받아야 한다.

- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics
- VCF(화상회의 딥페이크 벤치마크): https://github.com/mirmashel/vcf_dataset

## 참고 논문

- Kohler, Vijaykumar & Imtiaz — DeepFake Detection in Dyadic Video Calls using Point of Gaze Tracking (IEEE Access, 2026)
- Rössler et al. (2019) — FaceForensics++ (ICCV)
- Li et al. (2020) — Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics (CVPR)
- Demir & Ciftci (2021) — Where Do Deep Fakes Look? Synthetic Face Detection via Gaze Tracking (ETRA)
- Li, Chang & Lyu (2018) — In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking
