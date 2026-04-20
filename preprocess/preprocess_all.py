"""여러 입력 폴더를 한번에 전처리해서 train/val 로 랜덤 split.

환경변수로 오버라이드:
  - DATA_ROOT: 데이터 루트 (기본 ./data/ceramics)
  - INPUT_FOLDERS: ':' 로 구분된 입력 폴더 목록 (기본값: g2/g3 계열 4종)
  - VAL_RATIO: val 비율 (기본 0.1)
  - NUM_WORKERS: 병렬 워커 수 (기본 min(8, cpu_count))
"""
from __future__ import annotations

import multiprocessing
import os
import random

import cv2
from tqdm import tqdm

from pipeline import DATA_ROOT, VALID_EXTS, preprocess_file

# 기본 입력 폴더. INPUT_FOLDERS 환경변수로 덮어쓸 수 있다.
DEFAULT_INPUT_FOLDERS = [
    f"{DATA_ROOT}/train_g2_orig_ok",
    f"{DATA_ROOT}/train_g2_orig_ok_new",
    f"{DATA_ROOT}/train_g3_orig_ok",
    f"{DATA_ROOT}/train_g3_orig_ok_new",
]


def _parse_input_folders() -> list[str]:
    """INPUT_FOLDERS 환경변수가 있으면 ':' 분리, 없으면 DEFAULT 사용."""
    raw = os.environ.get("INPUT_FOLDERS")
    if not raw:
        return DEFAULT_INPUT_FOLDERS
    return [p for p in raw.split(":") if p]


TRAIN_OUTPUT = os.environ.get("TRAIN_OUTPUT", f"{DATA_ROOT}/train")
VAL_OUTPUT = os.environ.get("VAL_OUTPUT", f"{DATA_ROOT}/val")
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(8, multiprocessing.cpu_count())))


def process_single_file(args: tuple[str, str, str]) -> str | None:
    """워커 프로세스 엔트리: (입력경로, 출력폴더, 저장base) → 실패 메시지 또는 None."""
    img_path, out_folder, base_name = args
    try:
        processed = preprocess_file(img_path)
        if processed is None:
            return f"[WARN] Cannot read: {img_path}"
        out_path = os.path.join(out_folder, f"{base_name}.jpg")
        cv2.imwrite(out_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return None
    except Exception as exc:  # pragma: no cover - logging convenience
        return f"[ERROR] {img_path}: {exc}"


def collect_files(input_folders: list[str]) -> list[tuple[str, str]]:
    """여러 폴더를 훑어 (img_path, base_name) 페어를 모은다.

    base_name 은 '폴더명_파일명' 으로 만들어 폴더 간 동일 파일명 충돌을 방지한다.
    """
    all_files: list[tuple[str, str]] = []
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        folder_name = os.path.basename(folder.rstrip("/"))
        for img_file in os.listdir(folder):
            if not img_file.lower().endswith(VALID_EXTS):
                continue
            if img_file.startswith("._"):  # macOS resource fork
                continue
            img_path = os.path.join(folder, img_file)
            base_name = f"{folder_name}_{os.path.splitext(img_file)[0]}"
            all_files.append((img_path, base_name))
    return all_files


def main() -> None:
    os.makedirs(TRAIN_OUTPUT, exist_ok=True)
    os.makedirs(VAL_OUTPUT, exist_ok=True)

    input_folders = _parse_input_folders()
    all_files = collect_files(input_folders)
    print(f"Found {len(all_files)} total images from {len(input_folders)} folders")

    # 랜덤 셔플 후 앞쪽 VAL_RATIO 만큼을 val, 나머지는 train 으로 분할.
    random.shuffle(all_files)
    val_count = int(len(all_files) * VAL_RATIO)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    print(f"Train: {len(train_files)} images  /  Val: {len(val_files)} images")

    # 워커에 전달할 job 리스트 (train + val 을 한 풀에서 처리).
    jobs: list[tuple[str, str, str]] = []
    for img_path, base_name in train_files:
        jobs.append((img_path, TRAIN_OUTPUT, base_name))
    for img_path, base_name in val_files:
        jobs.append((img_path, VAL_OUTPUT, base_name))

    print(f"Processing with {NUM_WORKERS} workers...")
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_file, jobs, chunksize=10),
            total=len(jobs),
        ))

    # 실패 메시지 수집 후 상위 10개만 출력 (한꺼번에 다 찍으면 로그가 길어지므로).
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for err in errors[:10]:
            print(err)

    print(f"\nDone! Train: {TRAIN_OUTPUT}  /  Val: {VAL_OUTPUT}")


if __name__ == "__main__":
    main()
