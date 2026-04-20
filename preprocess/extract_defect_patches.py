"""LabelMe JSON 어노테이션에서 결함(Crack/Bow) 패치를 추출.

입력 이미지에는 학습 데이터와 동일한 FFT 전처리를 적용한 후, 폴리곤 영역만
alpha 채널 마스크로 잘라 `defect_patches/{crack,bow,other}` 에 PNG 로 저장한다.
`data_loader_black.py` 가 학습 시 이 패치들을 augmentation source 로 사용한다.

환경변수 오버라이드:
  - DATA_ROOT: 데이터 루트 (기본 ./data/ceramics)
  - PATCH_INPUT_DIR: JSON+이미지 경로 (기본 $DATA_ROOT/test_orig)
  - PATCH_OUTPUT_DIR: 패치 저장 경로 (기본 $DATA_ROOT/defect_patches)
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from pipeline import DATA_ROOT, TARGET_SIZE, imread_unicode, preprocess_gray, to_gray

INPUT_DIR = os.environ.get("PATCH_INPUT_DIR", f"{DATA_ROOT}/test_orig")
OUTPUT_DIR = os.environ.get("PATCH_OUTPUT_DIR", f"{DATA_ROOT}/defect_patches")
# bounding box 주변 여유 픽셀. 너무 꽉 잘라내지 않기 위함.
PADDING = int(os.environ.get("PATCH_PADDING", "25"))


def ensure_dirs() -> None:
    """crack / bow / other 3개 하위 폴더 생성."""
    for sub in ("crack", "bow", "other"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def parse_json(json_path: str):
    """LabelMe JSON 에서 shapes / 원본 해상도 정보 추출."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data.get("shapes", []),
        data.get("imagePath", ""),
        data.get("imageHeight", 0),
        data.get("imageWidth", 0),
    )


def scale_points(points, orig_h, orig_w, target_h, target_w):
    """원본 해상도 좌표를 TARGET_SIZE 기준으로 스케일링."""
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    return [[p[0] * scale_x, p[1] * scale_y] for p in points]


def polygon_to_mask(points, height, width):
    """폴리곤을 255 채워진 바이너리 마스크로 변환."""
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_patch(image, mask, points, padding: int = PADDING):
    """bounding box + padding 으로 patch 를 잘라 RGBA 로 반환.

    alpha 채널에는 polygon 마스크를 그대로 넣어서, 학습 단계에서
    실제 결함 모양만 원본에 블렌딩할 수 있게 한다.
    """
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    # padding 적용하되 이미지 경계 벗어나지 않도록 clip.
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    # 너무 작은 patch 는 학습에 노이즈만 되므로 스킵.
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None

    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy()

    # grayscale 이면 3채널로 복사, 컬러면 그대로. 마지막 채널은 alpha.
    if cropped_img.ndim == 2:
        rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = cropped_img
        rgba[:, :, 1] = cropped_img
        rgba[:, :, 2] = cropped_img
        rgba[:, :, 3] = cropped_mask
    else:
        rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_img
        rgba[:, :, 3] = cropped_mask
    return rgba


def get_label_category(label: str) -> str:
    """라벨 문자열에서 카테고리 추출. 'crack*', 'bow*' 외에는 other 로 분류."""
    label_lower = label.lower()
    if "crack" in label_lower:
        return "crack"
    if "bow" in label_lower:
        return "bow"
    return "other"


def find_image_for(json_path: str) -> str | None:
    """JSON 과 같은 base name 을 갖는 이미지 파일 경로를 찾는다."""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = os.path.join(INPUT_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> None:
    ensure_dirs()

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")
    print("Applying FFT preprocessing (same as training data)")

    patch_counts = {"crack": 0, "bow": 0, "other": 0}
    # TARGET_SIZE = (W, H) 순이므로 인덱스를 뒤집어 (H, W) 로 사용.
    target_h, target_w = TARGET_SIZE[1], TARGET_SIZE[0]

    for json_path in tqdm(json_files, desc="Extracting patches"):
        try:
            shapes, _, orig_h, orig_w = parse_json(json_path)
            if not shapes or orig_h == 0 or orig_w == 0:
                continue

            img_path = find_image_for(json_path)
            if img_path is None:
                continue

            image = imread_unicode(img_path)
            if image is None:
                continue

            # 학습과 동일한 전처리를 돌려야 patch 스타일이 일치.
            gray = to_gray(image)
            preprocessed = preprocess_gray(gray)

            base_name = os.path.splitext(os.path.basename(json_path))[0]
            for i, shape in enumerate(shapes):
                label = shape.get("label", "")
                points = shape.get("points", [])
                # polygon 이외 shape 나 3점 미만 라벨은 스킵.
                if shape.get("shape_type") != "polygon" or len(points) < 3:
                    continue

                category = get_label_category(label)
                scaled = scale_points(points, orig_h, orig_w, target_h, target_w)
                mask = polygon_to_mask(scaled, target_h, target_w)
                patch = extract_patch(preprocessed, mask, scaled)
                if patch is None:
                    continue

                out_path = os.path.join(OUTPUT_DIR, category, f"{base_name}_patch{i}.png")
                # 한글 경로 호환을 위해 imencode + tofile 조합 사용.
                ok, encoded = cv2.imencode(".png", patch)
                if ok:
                    encoded.tofile(out_path)
                    patch_counts[category] += 1
        except Exception as exc:
            print(f"Error processing {json_path}: {exc}")
            continue

    print("\nExtraction complete!")
    for cat, count in patch_counts.items():
        print(f"  {cat}: {count}")
    print(f"\nPatches saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
