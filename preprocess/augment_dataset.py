"""입력 폴더의 원본 이미지를 전처리한 뒤, 이미지당 N장의 증강본을 출력 폴더에 저장.

preprocess_all.py 와 다르게 val/train split 은 하지 않고,
이미 정해진 입력 폴더 하나에 대해 '전처리 + augmentation' 을 돌린다.

환경변수 오버라이드:
  - DATA_ROOT: 데이터 루트 (기본 ./data/ceramics)
  - AUG_INPUT: 입력 폴더 (기본 $DATA_ROOT/train_g2_orig_ok_new)
  - AUG_OUTPUT: 출력 폴더 (기본 $DATA_ROOT/train)
  - AUG_COUNT: 이미지당 증강 수 (기본 14)
  - AUG_MODE:  "basic" (flip/rotate/brightness/contrast) 또는 "full" (모든 기법) — 기본 "basic"
  - NUM_WORKERS: 병렬 워커 수
"""
from __future__ import annotations

import multiprocessing
import os

from PIL import Image
from tqdm import tqdm

from augment import apply_augmentation
from pipeline import DATA_ROOT, VALID_EXTS, preprocess_file

INPUT_FOLDER = os.environ.get("AUG_INPUT", f"{DATA_ROOT}/train_g2_orig_ok_new")
OUTPUT_FOLDER = os.environ.get("AUG_OUTPUT", f"{DATA_ROOT}/train")
AUG_COUNT = int(os.environ.get("AUG_COUNT", "14"))
AUG_MODE = os.environ.get("AUG_MODE", "basic")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", max(1, multiprocessing.cpu_count())))


def process_image(img_file: str) -> str | None:
    """단일 이미지에 대해 '전처리 → augmentation AUG_COUNT 회' 반복 저장."""
    if not img_file.lower().endswith(VALID_EXTS):
        return None
    if img_file.startswith("._"):  # macOS resource fork
        return None

    img_path = os.path.join(INPUT_FOLDER, img_file)
    try:
        # 1) pipeline 전처리 (FFT + 후처리 + 리사이즈).
        processed = preprocess_file(img_path)
        if processed is None:
            return f"[WARN] Cannot read: {img_path}"

        # 2) PIL 변환 후 같은 base 이미지에 대해 AUG_COUNT 번 증강 저장.
        pil_base = Image.fromarray(processed)
        base_name = os.path.splitext(img_file)[0]
        for i in range(AUG_COUNT):
            augmented = apply_augmentation(pil_base, mode=AUG_MODE)
            out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_aug_{i}.jpg")
            augmented.save(out_path, quality=100)
        return None
    except Exception as exc:  # pragma: no cover
        return f"[ERROR] {img_file}: {exc}"


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    img_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(VALID_EXTS)]
    print(f"Input: {INPUT_FOLDER}  ({len(img_files)} files)")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Augmentations per image: {AUG_COUNT}  (mode={AUG_MODE})")

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
