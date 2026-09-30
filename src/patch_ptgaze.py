"""
patch_ptgaze.py

ptgaze(0.2.8)는 오래된 라이브러리라 최신 torch/torchvision/scipy/numpy와
일부 API가 맞지 않는다. 이 스크립트는 실제로 PC에서 하나씩 겪고 해결했던
5개 패치를 자동으로 적용한다.

실행 (oculog_ptgaze 환경, ptgaze가 이미 pip install 되어 있는 상태):
    python patch_ptgaze.py

멱등성: 이미 패치된 부분은 다시 건드리지 않는다. 여러 번 실행해도 안전하다.
"""

import os
import re
import sys


def find_ptgaze_root():
    """설치된 ptgaze 패키지의 경로를 찾는다."""
    try:
        import ptgaze
    except ImportError:
        print("[에러] ptgaze가 설치되어 있지 않습니다. 먼저 'pip install ptgaze'를 실행하세요.")
        sys.exit(1)
    return os.path.dirname(ptgaze.__file__)


def patch_1_resnet_weights(root):
    """torchvision.models.resnet.model_urls -> Weights API"""
    path = os.path.join(root, "models", "mpiifacegaze", "backbones", "resnet_simple.py")
    if not os.path.exists(path):
        print(f"[건너뜀] 패치1: 파일 없음 ({path})")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "ResNet18_Weights" in content:
        print("[건너뜀] 패치1: 이미 적용됨")
        return

    old = """        pretrained_name = config.model.backbone.pretrained
        if pretrained_name:
            state_dict = torch.hub.load_state_dict_from_url(
                torchvision.models.resnet.model_urls[pretrained_name])
            self.load_state_dict(state_dict, strict=False)"""

    new = """        pretrained_name = config.model.backbone.pretrained
        if pretrained_name:
            weights_map = {
                'resnet18': torchvision.models.ResNet18_Weights.DEFAULT,
                'resnet34': torchvision.models.ResNet34_Weights.DEFAULT,
                'resnet50': torchvision.models.ResNet50_Weights.DEFAULT,
            }
            state_dict = weights_map[pretrained_name].get_state_dict(progress=True)
            self.load_state_dict(state_dict, strict=False)"""

    if old not in content:
        print(f"[경고] 패치1: 예상 패턴을 찾지 못했습니다. 수동 확인이 필요합니다. ({path})")
        return

    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[완료] 패치1: resnet_simple.py의 model_urls -> Weights API")


def patch_2_3_deprecated_numpy_types(root):
    """np.float, np.int, np.bool, np.object, np.str -> 순수 파이썬 타입.
    ptgaze 패키지 전체를 훑어서 일괄 치환한다 (여러 파일에 흩어져 있었음)."""
    replacements = [
        (re.compile(r"np\.float\b(?!\d)"), "float"),
        (re.compile(r"np\.int\b(?!\d)"), "int"),
        (re.compile(r"np\.bool\b(?!\d)"), "bool"),
        (re.compile(r"np\.object\b"), "object"),
        (re.compile(r"np\.str\b"), "str"),
    ]

    total_files = 0
    total_replacements = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            original = content
            count = 0
            for pattern, repl in replacements:
                matches = pattern.findall(content)
                count += len(matches)
                content = pattern.sub(repl, content)
            if content != original:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                total_files += 1
                total_replacements += count

    if total_replacements > 0:
        print(f"[완료] 패치2/3: np.float/int/bool 등 {total_replacements}개 치환 "
              f"({total_files}개 파일)")
    else:
        print("[건너뜀] 패치2/3: 치환 대상 없음 (이미 적용됐거나 해당 없음)")


def patch_4_rotation_flatten(root):
    """cv2.solvePnP의 (3,1) 형태 rvec/tvec -> scipy가 요구하는 (3,) 형태로 변환"""
    path = os.path.join(root, "common", "face_model.py")
    if not os.path.exists(path):
        print(f"[건너뜀] 패치4: 파일 없음 ({path})")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    changed = False

    old1 = "rot = Rotation.from_rotvec(rvec)"
    new1 = "rot = Rotation.from_rotvec(rvec.flatten())"
    if old1 in content:
        content = content.replace(old1, new1)
        changed = True

    old2 = "face.head_position = tvec"
    new2 = "face.head_position = tvec.flatten()"
    if old2 in content and "tvec.flatten()" not in content:
        content = content.replace(old2, new2)
        changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("[완료] 패치4: rvec/tvec .flatten() 추가 (shape 불일치 해결)")
    else:
        print("[건너뜀] 패치4: 이미 적용됐거나 대상 없음")


def main():
    root = find_ptgaze_root()
    print(f"ptgaze 설치 경로: {root}\n")

    patch_1_resnet_weights(root)
    patch_2_3_deprecated_numpy_types(root)
    patch_4_rotation_flatten(root)

    print("\n=== 패치 적용 완료 ===")
    print("확인을 위해 다음을 실행해보세요:")
    print("  python -m ptgaze --mode mpiifacegaze --face-detector mediapipe "
          "--device cpu --video <영상경로> --output-dir /tmp/ptgaze_test --no-screen")


if __name__ == "__main__":
    main()
