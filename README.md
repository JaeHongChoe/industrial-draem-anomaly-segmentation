# Semiconductor Parts Anomaly Segmentation

반도체 부품 이미지에서 결함 후보를 분할하고 제품 단위 불량 판정을 보조하기 위한 이상 탐지 파이프라인입니다.

이 저장소는 `원본 이미지 전처리 → 결함 패치 추출 → 정상 이미지 기반 재구성 모델 학습 → 가성 결함 패치 기반 분할 학습 → 추론 / 시각화`
흐름을 한곳에서 볼 수 있도록 정리한 코드 묶음입니다. 실제 이미지와 라벨 데이터는 보안상 포함하지 않습니다.

핵심 목적은 **정상 샘플로 깨끗한 이미지를 재구성하고, 실제 결함 패치와 가성 결함 패치를 정상 이미지에 합성해 결함 위치를 분할하도록 학습시키는 것**입니다.

## Project Snapshot

- Domain: 반도체 부품 비전 검사 / 결함 탐지
- Problem: 결함 샘플이 적고 제품 종류가 다양해, 단일 제품·단일 모델 가정의 방법론을 그대로 쓰기 어려움
- Approach: DRAEM의 "정상 재구성 + 합성 결함 분할" 학습 아이디어를 가져오되, 다품종 단일 모델 환경에 맞춰 AE 기반 재구성 구조와 결함 패치 합성 로직을 구성
- Output: anomaly probability map, input/reconstruction/probability/overlay 4분할 시각화, image/pixel 단위 평가 지표
- Scope: 공개 가능한 코드 구조와 학습/평가 흐름 정리. 실제 데이터와 체크포인트는 포함하지 않음

## What to Review First

평가자가 빠르게 구조를 보려면 아래 파일부터 보면 됩니다.

1. `preprocess/pipeline.py`  
   X-ray/비전 이미지의 FFT central-crop, 노이즈 제거, 대비 보정, sharpening 등 공용 전처리 흐름입니다.

2. `preprocess/extract_defect_patches.py`  
   LabelMe polygon annotation을 alpha mask가 포함된 결함 패치로 변환합니다.

3. `data_loader_black.py`  
   정상 이미지 위에 실제 결함 패치 또는 합성 결함을 배치해 학습용 image/mask pair를 생성합니다.

4. `model.py`, `model_unet.py`  
   정상 이미지 재구성용 AE 계열 모델과 anomaly mask 분할용 U-Net 계열 모델입니다.

5. `train.py`, `test.py`  
   학습, checkpoint/resume, TensorBoard 시각화, AUC/AP 계산, 4분할 결과 저장까지 이어지는 실행 엔트리입니다.

## 접근 배경

이 파이프라인이 해결하려는 문제와 단계별 논리는 다음과 같습니다.

- 대상 데이터는 저조도 계열의 단일 채널 (grayscale) 이미지이고, 정상 이미지는 많지만 결함 이미지는 매우 제한적이었습니다.
- 일반적인 Supervised 방식으로는 결함 샘플 부족 + 결함 분포 치우침으로 인해 학습이 불안정했습니다.
- DRAEM (Discriminatively trained Reconstruction Anomaly Embedding Model) 을 기반으로,
  - AutoEncoder 에게는 정상 이미지 재구성만 학습시키고,
  - Segmentation U-Net 에게는 "원본 vs 재구성" 차이에서 anomaly mask 를 뽑도록 학습시키는 2-스테이지 구조를 도입했습니다.
- 결함 합성은 50% 실제 결함 patch (LabelMe 어노테이션에서 추출) + 50% Perlin noise 를 원본에 블렌딩해 사용하도록 구성했습니다.

대표 구성:

- Reconstruction: `AE_high` (1→32→64→128→256→512→1024 encoder, symmetric decoder).
- Discriminative: `DiscriminativeSubNetwork` (U-Net, 2-channel 입력 → anomaly mask).
- Loss: `MSE + SSIM + Focal Loss`.
- Scheduler: `ReduceLROnPlateau` (validation SSIM loss 기준).

## 이 폴더의 범위

이 폴더에는 아래 단계가 포함됩니다.

1. 원본 이미지 FFT central-crop + 후처리 전처리
2. 학습 / 검증 split 및 augmentation 생성
3. LabelMe JSON 으로부터 결함 patch (alpha mask 포함) 추출
4. DRAEM AutoEncoder + U-Net 학습 / 재개 / 체크포인트 관리
5. 추론 + 결과 시각화 (Input / Recon / Prob / Overlay 4 분할 이미지)

## 중요 전제

- 이 폴더의 파일들은 원본 작업 코드에서 복사 / 정리된 코드입니다.
- 여기서 수정해도 원본 위치가 자동으로 같이 바뀌지는 않습니다.
- 경로는 가능한 한 `DATA_ROOT`, `RUNS_ROOT` 환경변수로 주입해 기본값 (`./data/ceramics`, `./runs`) 위에서 동작하도록 구성되어 있습니다.
- 실제 이미지 / 라벨 데이터는 저장소에 포함하지 않습니다. 동일한 폴더 구조에 직접 넣어 실행하는 구조입니다.
- 모델 파일 `model.py` 에는 실험 과정에서 사용한 다양한 AE / RAE 변형이 함께 들어 있지만, 현재 기본 설정은 `AE_high` 한 가지만 사용합니다.

## 폴더 구성

```text
industrial-draem-anomaly-segmentation/
├── preprocess/                      # 전처리 / augmentation / 패치 추출
│   ├── pipeline.py                  # FFT central-crop + post-processing 공용 모듈
│   ├── augment.py                   # PIL/NumPy 기반 augmentation (basic/full 모드)
│   ├── preprocess_all.py            # 여러 폴더 → train/val split
│   ├── preprocess_test.py           # test 폴더 전처리
│   ├── augment_dataset.py           # train 폴더에 대해 이미지당 N장 증강
│   ├── extract_defect_patches.py    # LabelMe JSON → alpha 마스크 결함 패치
│   └── README.md
├── config.py                        # DATA_ROOT/RUNS_ROOT 기반 경로 및 하이퍼파라미터
├── train.py                         # DRAEM 학습 엔트리 (argparse)
├── test.py                          # DRAEM 추론 + 시각화 엔트리 (argparse)
├── data.py                          # DataLoader 래퍼 (train/valid/test)
├── data_loader_black.py             # Custom Dataset (결함 patch + Perlin noise 합성)
├── model.py                         # AE / AE_high / RAE 등 AutoEncoder 구현
├── model_unet.py                    # DiscriminativeSubNetwork (U-Net)
├── loss.py                          # FocalLoss + SSIM
├── perlin.py                        # Perlin noise 생성기
├── tensorboard_visualizer.py        # 이미지 / loss 로그 어댑터
├── util.py                          # run name, resume, scheduler 복원 등 유틸
├── experiments/
│   └── tile_v4/                     # 후속 실험: 1024 타일 + DINOv3 인코더 분할 (README 참고)
├── requirements.txt
└── README.md
```

### `preprocess/`

원본 이미지 → 학습용 이미지로 바꾸는 전처리 전담 폴더입니다.

- `pipeline.py`: FFT central-crop (radius 170, padding 8, 640x640 resize) + Gaussian / Median / Wiener / Contrast stretch / Gamma / Unsharp mask 후처리. 모든 스크립트가 이 모듈을 import 해서 공유합니다.
- `augment.py`: flip / rotate / brightness / contrast / (옵션) blur / noise / elastic / CLAHE / erasing. `mode="basic"` 과 `mode="full"` 두 가지 프리셋 제공.
- `preprocess_all.py`: 여러 입력 폴더를 한 번에 전처리해서 `train/` 과 `val/` 로 랜덤 split. 기본 val 비율 10%.
- `preprocess_test.py`: 테스트 전용 (augmentation 없음).
- `augment_dataset.py`: 이미 전처리된 train 폴더에 대해 이미지당 N장 augmentation 저장.
- `extract_defect_patches.py`: LabelMe JSON polygon 을 읽어 `defect_patches/{crack,bow,other}/*.png` 로 alpha 채널 포함 RGBA 패치 저장. `data_loader_black.py` 가 학습 시 이 폴더를 결함 합성 소스로 사용합니다.

자세한 사용법은 [preprocess/README.md](preprocess/README.md) 참고.

### `train.py` / `test.py`

argparse 기반 엔트리 포인트입니다. `config.py` 의 기본값을 받되, CLI 인자로 덮어씁니다.

- `train.py`: AE + U-Net 동시 학습, 매 epoch 에 epoch별 / latest / class별 / best 체크포인트를 저장.
- `test.py`: 지정한 base model 이름을 받아 `.pckl` 체크포인트를 로드, 테스트 이미지에 대해 anomaly probability map 과 4분할 시각화 이미지, AUC / AP 지표를 함께 생성합니다.

### `data_loader_black.py`

학습 Dataset 본체. 주요 특징:

- 1-channel (grayscale) 입력, 640x640 기준.
- 결함 합성: 50% 확률로 `defect_patches/` 에서 실제 결함 patch 를 elastic + soft-edge 블렌딩, 50% 확률로 Perlin noise 기반 합성.
- `augmented_image`, `anomaly_mask` 를 함께 반환해 U-Net 학습에 그대로 연결.

### `model.py`

AutoEncoder 변형들이 모여 있습니다 (`AE`, `AE_high`, `AE_half`, `AE_high_half`, `RAE` 계열, `U2NET`, `AE_ultra` 등). 실험 과정의 흔적이지만 현재 기본 설정은 `AE_high` 한 가지만 활성 상태입니다. 다른 모델로 바꾸고 싶으면 `--model_class_name` 인자 또는 `config.model_class_name` 값을 변경하면 됩니다.

## 후속 실험: Tile V4 (`experiments/tile_v4/`)

640x640 전체 이미지 학습에서는 작은 결함이 다운샘플링으로 사라지는 한계가 있어,
1024 해상도를 512 타일(overlap 64)로 나누고 **동결된 DINOv3 ViT-L 인코더 + U-Net 디코더**로 직접 분할하는 구조를 실험했습니다.
불균형이 심한 결함 픽셀에 맞춰 Focal Tversky loss(FN 가중)를 쓰고, 절차적 크랙 합성과 원형 영역 배치 제약을 추가했습니다.
자세한 비교는 [experiments/tile_v4/README.md](experiments/tile_v4/README.md) 참고.

## 처리 흐름

```text
원본 이미지 + LabelMe JSON
    ↓
preprocess/preprocess_all.py  (train/val split)
preprocess/preprocess_test.py  (test)
    ↓
preprocess/augment_dataset.py  (이미지당 N장 augmentation)
preprocess/extract_defect_patches.py  (결함 patch)
    ↓
train.py  (AE_high + DiscriminativeSubNetwork)
    ↓
checkpoints/*.pckl
    ↓
test.py  (AUC/AP 지표 + 4분할 시각화)
```

## 단계별 추천 사용 순서

### 1. 데이터 루트 세팅

```bash
export DATA_ROOT=./data/ceramics
export RUNS_ROOT=./runs
```

`$DATA_ROOT` 아래에 원본 이미지와 LabelMe JSON 이 들어가는 폴더들을 배치합니다. 기본 레이아웃 예시:

```text
$DATA_ROOT/
├── train_g2_orig_ok/           # 정상 샘플 (원본)
├── train_g2_orig_ok_new/
├── train_g3_orig_ok/
├── train_g3_orig_ok_new/
├── test_orig/                  # 테스트 원본 + LabelMe JSON
├── train/                      # 전처리 + augmentation 결과 (생성됨)
├── val/                        # 전처리 split 결과 (생성됨)
├── test/                       # 전처리된 테스트 이미지 (생성됨)
└── defect_patches/             # 결함 patch (생성됨)
    ├── crack/
    ├── bow/
    └── other/
```

### 2. 전처리 + split

```bash
python preprocess/preprocess_all.py     # train_g* → train/ + val/
python preprocess/preprocess_test.py    # test_orig → test/
```

환경변수로 오버라이드:

- `INPUT_FOLDERS`: 입력 폴더 목록 (`:` 구분).
- `VAL_RATIO`: val 비율 (기본 0.1).
- `NUM_WORKERS`: 병렬 워커 수.

### 3. 결함 patch 추출

```bash
python preprocess/extract_defect_patches.py
```

LabelMe JSON 의 polygon 을 학습 데이터와 동일한 FFT 전처리로 정렬한 뒤 alpha 채널 마스크로 잘라 저장합니다.
`PATCH_INPUT_DIR`, `PATCH_OUTPUT_DIR`, `PATCH_PADDING` 환경변수로 경로 / 패딩을 조정할 수 있습니다.

### 4. Augmentation (선택)

```bash
python preprocess/augment_dataset.py
```

이미 전처리된 train 폴더에 이미지당 N장 증강본을 추가 저장합니다.
`AUG_INPUT`, `AUG_OUTPUT`, `AUG_COUNT` (기본 14), `AUG_MODE` (`basic` / `full`) 환경변수로 동작을 바꿀 수 있습니다.

### 5. 학습

```bash
python train.py \
    --bs 2 --lr 0.0001 --epochs 700 \
    --data_path "$DATA_ROOT/train/*.jpg" \
    --valid_path "$DATA_ROOT/val/*.jpg" \
    --checkpoint_path ./checkpoints \
    --log_path ./logs \
    --visualize
```

재개 (resume) 는 다음 중 하나를 사용합니다.

```bash
python train.py ... --resume auto                   # 가장 최근 epoch 체크포인트 자동 선택
python train.py ... --resume ./checkpoints/...pckl  # 특정 파일 지정
```

매 epoch 에 다음 파일들이 저장됩니다.

- `{run_name}epoch{N}.pckl`, `{run_name}epoch{N}_seg.pckl`: epoch 단위 가중치.
- `{run_name}epoch{N}_full.pckl`: optimizer / scheduler / best_ssim / n_iter 포함 full checkpoint.
- `{run_name}.pckl`, `{run_name}_seg.pckl`: 최신 latest.
- `{model_class_name}.pckl`, `{model_class_name}_seg.pckl`: 클래스명 기반 latest.
- `{run_name}best.pckl`, `{model_class_name}_best.pckl`: 검증 SSIM loss 가 갱신될 때만 추가 저장.

### 6. 추론 + 시각화

```bash
python test.py \
    --base_model_name DRAEM_AE_high_640_0.0001_700_bs2_ \
    --data_path "$DATA_ROOT/test" \
    --checkpoint_path ./checkpoints \
    --save_dir ./runs/images \
    --csv_path ./runs/DRAEM_result.csv
```

결과:

- `save_dir/{idx}_combined.png`: Input / Reconstructed / Probability heatmap / Overlay 4분할 이미지.
- `csv_path`: run 단위 image AUC, pixel AUC, image AP, pixel AP.

## 가장 먼저 기억할 파일

- `preprocess/pipeline.py`
  - 전체 파이프라인의 FFT + 후처리 공용 모듈 (전 스크립트가 공유).

- `preprocess/extract_defect_patches.py`
  - LabelMe 어노테이션을 학습 친화적인 결함 patch 로 바꾸는 진입점.

- `train.py`
  - AE + U-Net 동시 학습의 메인 엔트리.

- `data_loader_black.py`
  - 실제 결함 patch 와 Perlin noise 를 블렌딩해 `augmented_image` 와 `anomaly_mask` 를 만드는 핵심 Dataset.

- `test.py`
  - Anomaly map 시각화와 AUC / AP 지표 계산.

## 환경 메모

- `requirements.txt` 는 conda list 형식으로 정리되어 있습니다 (`conda create --name <env> --file requirements.txt`). pip 기반 환경이라면 `python=3.8`, `pytorch=1.8`, `torchvision=0.9`, `opencv-python`, `imgaug`, `scikit-learn`, `scikit-image`, `tensorboard`, `imageio`, `numpy`, `pillow` 수준으로 맞추면 됩니다.
- GPU 가 없으면 `train.py` / `test.py` 는 자동으로 MPS (Apple Silicon) 또는 CPU 로 떨어집니다. 하지만 학습 시간은 CUDA 기준으로 설계되어 있습니다.
- TensorBoard 를 쓰려면 `--visualize` 옵션을 주고 `tensorboard --logdir logs/` 로 열면 됩니다.
- 체크포인트 / 로그 / 생성 이미지는 `.gitignore` 로 기본 제외되어 있습니다.

## 메모

- 이 폴더는 전체 파이프라인을 빠르게 이해하기 위한 정리본입니다.
- 원본 코드가 나중에 바뀌어도 이 정리본이 자동 동기화되지는 않습니다.
- 데이터, 체크포인트, one-off 로그는 저장소에 포함하지 않는 구성이 자연스럽습니다.
- 참고 논문: DRAEM - Zavrtanik et al., "DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection", ICCV 2021.
