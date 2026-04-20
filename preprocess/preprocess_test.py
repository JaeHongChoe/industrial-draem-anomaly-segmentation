"""테스트 원본 폴더를 전처리해서 test 폴더로 저장.

augmentation 없이 pipeline.preprocess_file 만 돌리므로, 학습 데이터와 동일한
FFT + 후처리 결과가 test 이미지에도 적용된다.

환경변수 오버라이드:
  - DATA_ROOT: 데이터 루트 (기본 ./data/ceramics)
  - TEST_INPUT: 입력 폴더 (기본 $DATA_ROOT/test_orig)
  - TEST_OUTPUT: 출력 폴더 (기본 $DATA_ROOT/test)
  - NUM_WORKERS: 병렬 워커 수 (기본 cpu_count)
"""
from __future__ import annotations

import multiprocessing
import os

import cv2
from tqdm import tqdm

from pipeline import DATA_ROOT, VALID_EXTS, preprocess_file

INPUT_FOLDER = os.environ.get("TEST_INPUT", f"{DATA_ROOT}/test_orig")
OUTPUT_FOLDER = os.environ.get("TEST_OUTPUT", f"{DATA_ROOT}/test")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", max(1, multiprocessing.cpu_count())))


def process_image(img_file: str) -> str | None:
    """단일 파일 처리. 확장자/리소스 포크 필터링 후 preprocess_file 호출."""
    if not img_file.lower().endswith(VALID_EXTS):
        return None
    if img_file.startswith("._"):  # macOS resource fork
        return None

    img_path = os.path.join(INPUT_FOLDER, img_file)
    try:
        processed = preprocess_file(img_path)
        if processed is None:
            return f"[WARN] Cannot read: {img_path}"
        # 원본 확장자 무시하고 JPEG 로 통일 (학습 파이프라인과 일치시킴).
        out_name = f"{os.path.splitext(img_file)[0]}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return None
    except Exception as exc:  # pragma: no cover
        return f"[ERROR] {img_file}: {exc}"


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    img_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(VALID_EXTS)]
    print(f"Input: {INPUT_FOLDER}  ({len(img_files)} files)")
    print(f"Output: {OUTPUT_FOLDER}")

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_image, img_files),
            total=len(img_files),
        ))

    errors = [r for r in results if r]
    if errors:
        print(f"\n{len(errors)} errors:")
        for err in errors[:10]:
            print(err)


if __name__ == "__main__":
    main()
