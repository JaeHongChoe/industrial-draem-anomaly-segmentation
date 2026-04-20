# Preprocess

원본 이미지를 DRAEM 학습에 사용할 수 있도록 변환하고, 결함 patch 를 뽑아주는 전처리 모음입니다.
기존에는 `a1.py` ~ `a5.py` 가 각각 비슷한 FFT + 후처리 + augmentation 코드를 따로 갖고 있었지만,
중복을 제거하면서 공용 모듈 (`pipeline.py`, `augment.py`) + 용도별 스크립트 4종으로 정리했습니다.

## 폴더 구성

```text
preprocess/
├── pipeline.py                 # FFT central-crop + post-processing 공용 모듈
├── augment.py                  # PIL/NumPy 기반 augmentation (basic/full 모드)
├── preprocess_all.py           # 여러 폴더 → train/val split (multiprocessing)
├── preprocess_test.py          # test 폴더 전처리 (augmentation 없음)
├── augment_dataset.py          # 전처리된 폴더에 대해 이미지당 N장 증강
├── extract_defect_patches.py   # LabelMe JSON → alpha 마스크 결함 패치
└── README.md
```

## 공용 파이프라인 (`pipeline.py`)

모든 스크립트가 공유하는 단계는 다음과 같습니다.

1. Grayscale 변환 (1 채널).
2. FFT central crop: reflection padding 후 FFT, 중심 반경 `CROP_R=170` 영역만 남기고 주변 고주파 노이즈 제거. 역FFT 로 복원.
3. Tone matching: mean / std 를 원본 이미지에 맞춰 전반적인 밝기 / 대비 복원.
4. Post-processing:
   - Gaussian blur (`sigma=1.0`)
   - Median blur (`ksize=3`)
   - 국소 Wiener filter (`ksize=5`, `noise_var=100.0`)
   - Contrast stretching (10% ~ 97% percentile)
   - Gamma correction (`gamma=1.4`)
   - Unsharp mask (`amount=2.2`, `sigma=0.5`)
5. Resize → `TARGET_SIZE=(640, 640)`.

상수는 전부 `pipeline.py` 최상단에 정리되어 있고, `DATA_ROOT` 환경변수로 데이터 루트만 바꾸면 기본 경로가 자동으로 맞춰집니다.

## 공용 Augmentation (`augment.py`)

`apply_augmentation(image, mode="full")` 한 함수로 통일되어 있습니다.

- 항상 적용: flip (상하/좌우), rotation ±180°.
- 확률 50%: brightness / contrast 조정.
- `mode="full"` 에서만 추가: Gaussian blur, Gaussian noise, elastic transform, CLAHE, random erasing (각 30%).

`mode="basic"` 은 과도한 변형 없이 flip / rotate / brightness / contrast 까지만 적용하는 보수적인 버전입니다.

## 스크립트별 역할

| 스크립트 | 기본 입력 | 기본 출력 | 비고 |
|----------|-----------|-----------|------|
| `preprocess_all.py` | `train_g2_orig_ok`, `train_g2_orig_ok_new`, `train_g3_orig_ok`, `train_g3_orig_ok_new` | `train/`, `val/` | `VAL_RATIO` (기본 10%) 로 랜덤 split |
| `preprocess_test.py` | `test_orig/` | `test/` | augmentation 없음 |
| `augment_dataset.py` | `train_g2_orig_ok_new/` | `train/` | 이미지당 `AUG_COUNT` (기본 14) 장 증강 |
| `extract_defect_patches.py` | `test_orig/*.json + 이미지` | `defect_patches/{crack,bow,other}/` | LabelMe JSON polygon → alpha 마스크 PNG |

모든 경로 기본값은 `DATA_ROOT` (기본 `./data/ceramics`) 기준으로 계산되며, 환경변수로 덮어쓸 수 있습니다.

## 환경변수 요약

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_ROOT` | `./data/ceramics` | 데이터 루트 |
| `NUM_WORKERS` | `cpu_count()` (일부는 `min(8, cpu_count())`) | multiprocessing 워커 수 |
| `INPUT_FOLDERS` | g2/g3 계열 4종 | `preprocess_all.py` 입력 (`:` 구분) |
| `TRAIN_OUTPUT`, `VAL_OUTPUT`, `VAL_RATIO` | `$DATA_ROOT/train`, `$DATA_ROOT/val`, `0.1` | `preprocess_all.py` 출력 |
| `TEST_INPUT`, `TEST_OUTPUT` | `$DATA_ROOT/test_orig`, `$DATA_ROOT/test` | `preprocess_test.py` |
| `AUG_INPUT`, `AUG_OUTPUT`, `AUG_COUNT`, `AUG_MODE` | `$DATA_ROOT/train_g2_orig_ok_new`, `$DATA_ROOT/train`, `14`, `basic` | `augment_dataset.py` |
| `PATCH_INPUT_DIR`, `PATCH_OUTPUT_DIR`, `PATCH_PADDING` | `$DATA_ROOT/test_orig`, `$DATA_ROOT/defect_patches`, `25` | `extract_defect_patches.py` |

## 일반적인 실행 순서

```bash
export DATA_ROOT=./data/ceramics

# 1. 학습용 원본을 전처리하고 train/val 로 split
python preprocess/preprocess_all.py

# 2. 테스트용 원본을 전처리
python preprocess/preprocess_test.py

# 3. 결함 patch 추출 (LabelMe JSON 필요)
python preprocess/extract_defect_patches.py

# 4. (선택) 이미지당 N장 증강본 추가
AUG_COUNT=14 AUG_MODE=basic python preprocess/augment_dataset.py
```

## 메모

- 결함 patch 는 학습 시 `data_loader_black.py` 가 `defect_patches/` 를 읽어 `augmented_image` / `anomaly_mask` 합성 소스로 사용합니다. 따라서 결함 class 별 폴더 구조 (`crack`, `bow`, `other`) 를 유지하는 것이 중요합니다.
- 마스크는 alpha 채널로 저장되므로, 학습 단계에서 RGBA 를 읽어 alpha > 0 인 영역만 블렌딩됩니다.
- FFT central crop 의 radius 170, padding 8 은 원본 카메라 / 해상도 / 타원 결함 spectrum 을 기준으로 튜닝된 값입니다. 다른 해상도에서는 `pipeline.py` 상단 상수를 같이 바꿔야 합니다.
