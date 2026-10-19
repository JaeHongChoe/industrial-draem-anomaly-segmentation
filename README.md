# 반도체 부품 이상 영역 분할 (DRAEM Anomaly Segmentation)

정상 이미지 위주로 학습해, 흑백(grayscale) 반도체 부품 이미지에서 결함이 의심되는 위치를 픽셀 단위 확률맵으로 표시하는 이상 탐지 파이프라인입니다.

## 한눈에 보기

- **문제**: 정상 이미지는 많고 결함 이미지는 매우 적어, 결함 라벨로 학습하는 일반적인 지도학습이 어려움
- **접근**: DRAEM. 정상 이미지에 가짜 결함을 합성하고, "복원 모델 + 결함 위치 분할 모델"을 함께 학습
- **결과물**: 픽셀별 결함 확률맵, 4분할 시각화 PNG (Input / Recon / Prob / Result), image·pixel 단위 AUC/AP CSV
- **기술 스택**: Python, PyTorch, OpenCV, imgaug, scikit-learn, TensorBoard (후속 실험: timm DINOv3)
- **포함**: 전처리 → 결함 패치 추출 → 학습 → 추론/시각화 코드, 후속 실험 `experiments/tile_v4/`
- **미포함**: 실제 이미지·라벨·학습된 체크포인트·성능 수치 (보안상 비공개)

## 왜 필요한가

검사 현장에서는 정상 부품 이미지는 계속 쌓이지만, 결함 이미지는 드물고 모양도 제각각입니다.
결함 사진을 많이 모아 "이것이 결함"이라고 가르치는 지도학습은 샘플 부족과 결함 분포 치우침 때문에 학습이 불안정했습니다.
그래서 모델에게는 "정상이 어떻게 생겼는지"를 배우게 하고, 학습 중에는 정상 이미지 위에 가짜 결함을 직접 만들어 붙여 위치 찾기를 연습시킵니다.
소량의 실제 결함 라벨은 정답 데이터가 아니라 "가짜 결함 재료"로만 사용하므로, 결함 데이터가 적어도 픽셀 단위 분할 모델을 학습할 수 있습니다.

## 동작 방식

```text
1. 원본 grayscale 이미지 → FFT 노이즈 제거 + 대비 보정 + 640x640 리사이즈   (preprocess/)
2. 소량의 결함 라벨 (LabelMe polygon) → 투명 배경 결함 패치 PNG 추출
3. 학습 시 정상 이미지에 가짜 결함 합성: 실제 결함 패치 50% / Perlin noise 50%
4. 재구성 AE (AE_high): 결함이 섞인 입력 → 깨끗한 정상 이미지로 복원
5. 분할 U-Net: [복원 이미지, 입력 이미지] 2채널을 비교 → 결함 마스크 예측
6. 추론: 테스트 이미지 → 결함 확률맵 + 4분할 시각화 + AUC/AP 기록
```

- **왜 두 개의 네트워크인가**: AE는 정상만 복원하도록 배우기 때문에 결함 부위를 "정상처럼" 지워서 복원합니다. 입력과 복원의 차이가 결함 단서가 되고, U-Net이 이 차이를 보고 위치를 마스크로 그립니다.
- **가짜 결함 ① 실제 패치**: 결함 패치 1~3개를 0.5~2.5배 스케일, 회전, flip, elastic/perspective 변형 후 soft-edge 또는 DRAEM 방식으로 블렌딩합니다.
- **가짜 결함 ② Perlin noise**: 가는 crack 50% / 원형 반점 25% / 불규칙 blob 25%로 만들고, 부품의 원형 유효 영역 안에만 배치합니다.
- **정상 샘플 유지**: 학습 샘플의 50%는 결함 없이 그대로 넣어, 정상 이미지에서 결함을 잘못 찾지 않도록 균형을 맞춥니다.
- **손실**: 복원은 MSE + SSIM, 분할은 Focal loss. 두 네트워크를 하나의 optimizer로 동시에 학습합니다.

## 시도와 개선 과정

| 단계 | 방법 | 이유 / 결과 |
|---|---|---|
| 전처리 정리 | 스크립트마다 중복되던 FFT·augmentation 코드를 `pipeline.py`, `augment.py` 공용 모듈로 통합 | 학습·테스트·패치 추출이 동일한 전처리를 공유 |
| DRAEM 기본 (루트) | `AE_high` + `DiscriminativeSubNetwork`, 640x640 입력, 실제 패치 + Perlin 합성 | 결함 샘플 부족을 "정상 + 합성 결함"으로 우회 |
| 재구성 모델 비교 | `model.py`에 `AE`, `AE_half`, `RAE` 계열, `U2NET`, `AE_ultra` 등 변형 구현 | 현재 기본 설정은 `AE_high` 하나만 사용 |
| 학습 중 증강 조정 | Dataset 내부 Zoom/Elastic 증강 | 품질 저하 이슈로 비활성화 (코드에 주석으로 보존) |
| Tile V4 (`experiments/tile_v4/`) | 1024 이미지를 512 타일 5장(overlap 64)으로 분할, 동결 DINOv3 ViT-L/16 인코더 + U-Net 디코더, Focal Tversky loss | 640 전체 이미지에서는 작은 결함이 다운샘플링으로 사라짐 → 고해상도 타일 + 사전학습 표현으로 보강, 미검(FN)에 더 큰 패널티 |

> 저장소에는 단계별 성능 수치가 기록되어 있지 않아 결과 비교 수치는 적지 않았습니다.

## 빠른 시작

```bash
# requirements.txt 는 conda list 형식 (linux-64, Python 3.8 / PyTorch 1.8 / CUDA 10.2)
conda create --name draem --file requirements.txt
conda activate draem
```

데이터와 체크포인트는 포함되어 있지 않습니다. `$DATA_ROOT` 아래에 직접 아래처럼 배치합니다.

```text
$DATA_ROOT/
├── <정상 원본 폴더들>/            # INPUT_FOLDERS 환경변수로 지정
├── test_orig/                     # 테스트 원본 + LabelMe JSON (결함 패치 소스)
├── train/  val/  test/            # 전처리 결과 (생성됨)
└── defect_patches/{crack,bow,other}/   # 결함 패치 (생성됨)
```

```bash
export DATA_ROOT=./data                                   # 기본값: ./data/ceramics
export INPUT_FOLDERS="$DATA_ROOT/normal_a:$DATA_ROOT/normal_b"   # ':' 구분

# 1) 전처리
python preprocess/preprocess_all.py           # 정상 원본 → train/ + val/ (val 10%)
python preprocess/preprocess_test.py          # test_orig/ → test/
python preprocess/extract_defect_patches.py   # LabelMe JSON → defect_patches/
# (선택) 원본 폴더 하나를 전처리 + 이미지당 N장 증강해 train/ 에 추가
AUG_INPUT="$DATA_ROOT/normal_a" AUG_COUNT=14 python preprocess/augment_dataset.py

# 2) 학습 (체크포인트 이름 prefix: DRAEM_AE_high_640_0.0001_700_bs2_)
python train.py --bs 2 --lr 0.0001 --epochs 700 \
  --data_path "$DATA_ROOT/train/*.jpg" --valid_path "$DATA_ROOT/val/*.jpg" \
  --anomaly_source_path "$DATA_ROOT/defect_patches" \
  --checkpoint_path ./checkpoints --log_path ./logs --visualize
# 재개: 같은 인자 + --resume auto (또는 --resume <체크포인트 경로>)

# 3) 추론 + 시각화
python test.py --base_model_name DRAEM_AE_high_640_0.0001_700_bs2_best \
  --data_path "$DATA_ROOT/test" --checkpoint_path ./checkpoints \
  --save_dir ./runs/images --csv_path ./runs/DRAEM_result.csv
```

- `--anomaly_source_path`를 생략하면 실행 위치 기준 `./defect_patches/{crack,bow}`, `../defect_patches/{crack,bow}`만 자동 탐색합니다. 패치가 하나도 없으면 Perlin 합성만 사용합니다.
- AUC/AP 계산에는 MVTec 형식 폴더가 필요합니다: `test/good/`, `test/<결함종류>/`, 그리고 `test/`와 같은 위치의 `ground_truth/<결함종류>/<이름>_mask.png`. 정상/결함 두 클래스가 없으면 시각화만 저장하고 지표 계산은 건너뜁니다.
- GPU가 없으면 MPS(Apple Silicon) 또는 CPU로 자동 전환됩니다. TensorBoard: `tensorboard --logdir logs/`.

## 폴더 구조

```text
.
├── preprocess/                  # 전처리 / augmentation / 결함 패치 추출 (자세한 내용: preprocess/README.md)
│   ├── pipeline.py              # FFT central-crop + 후처리 공용 모듈
│   ├── augment.py               # augmentation (basic / full 모드)
│   ├── preprocess_all.py        # 여러 정상 폴더 → train/val split
│   ├── preprocess_test.py       # 테스트 폴더 전처리
│   ├── augment_dataset.py       # 원본 폴더 전처리 + 이미지당 N장 증강
│   └── extract_defect_patches.py  # LabelMe JSON → alpha 마스크 결함 패치
├── config.py                    # DATA_ROOT / RUNS_ROOT 기반 경로와 기본 하이퍼파라미터
├── train.py                     # 학습 엔트리 (AE + U-Net 동시 학습, resume, 체크포인트)
├── test.py                      # 추론 엔트리 (확률맵, 4분할 시각화, AUC/AP)
├── data.py                      # DataLoader 래퍼
├── data_loader_black.py         # 학습/검증/테스트 Dataset (결함 합성 로직 핵심)
├── model.py                     # AE_high 등 재구성 AE 변형 모음
├── model_unet.py                # DiscriminativeSubNetwork (분할 U-Net)
├── loss.py                      # FocalLoss, SSIM
├── perlin.py                    # Perlin noise 생성
├── tensorboard_visualizer.py    # TensorBoard 이미지 / loss 로깅
├── util.py                      # run name, resume, scheduler 복원, 결과 CSV
├── experiments/tile_v4/         # 후속 실험: 512 타일 + DINOv3 분할 (README 참고)
└── requirements.txt             # conda 환경 정의
```

## 기술 상세

<details>
<summary>전처리 파라미터 (preprocess/pipeline.py, augment.py)</summary>

- Grayscale 변환 → reflection padding(`PAD=8`) 후 FFT, 중심 반경 `CROP_R=170` 영역만 남겨 고주파 노이즈 제거 → 역FFT → 원본 mean/std로 tone matching
- 후처리 순서: Gaussian blur(`sigma=1.0`) → Median(`k=3`) → 국소 Wiener(`ksize=5`, `noise_var=100.0`) → contrast stretch(10~97 percentile) → gamma(`1.4`) → unsharp mask(`amount=2.2`, `sigma=0.5`) → `640x640` 리사이즈
- Augmentation: 항상 flip(상하/좌우 각 50%) + ±180° 회전, 50% 확률 brightness/contrast. `full` 모드에서만 blur / noise / elastic / CLAHE / random erasing (각 30%). `augment_dataset.py` 기본은 `basic`
- 결함 패치: polygon 라벨에 `crack` 포함 → `crack/`, `bow` 포함 → `bow/`, 나머지 → `other/`. 기본 패딩 25px
</details>

<details>
<summary>모델 · 손실 · 학습 설정</summary>

- **재구성**: `AE_high` — encoder 6단계(1→32→64→128→256→512→1024 채널, ResidualBlock + MaxPool), 대칭 decoder(bicubic upsample + ResidualBlock), 출력 Sigmoid
- **분할**: `DiscriminativeSubNetwork` — 입력 2채널(복원 + 입력), 출력 2채널(정상/결함), base channels 64
- **손실**: `MSE(recon, 원본) + SSIM(recon, 원본) + Focal(seg, mask)`
- **기본값 (`config.py`)**: optimizer `adamw`, scheduler `plateau`(검증 SSIM loss 기준, factor 0.5, patience 10), init `xavier`, lr 0.001, batch 6, epochs 1000. `--optimizer_type`, `--scheduler_type`(`step`/`cosine` 가능), `--init_type`, `--model_class_name`으로 변경
- 다른 재구성 모델은 `--model_class_name`으로 선택 (`AE`, `AE_half`, `AE_high_half`, `RAE`, `RAE_high`, `U2NET`, `AE_ultra` 등)
</details>

<details>
<summary>합성 결함 상세 (data_loader_black.py · CustomDRAEMTrainDataset)</summary>

- **실제 패치 모드 (50%)**: RGBA 패치 1~3개 로드 → scale 0.5~2.5x → rotation → flip → elastic(30%) → perspective(20%) → 랜덤 위치 배치. 블렌딩은 50% soft-edge(마스크 Gaussian blur, `beta` 0.6~1.0), 50% 원본 DRAEM 방식(`beta` 0~0.8). 마스크는 패치별 최대값으로 합침
- **Perlin 모드 (50%)**: scale 2^0~2^5 Perlin noise를 ±90° 회전 후 crack(|noise| < 0.02~0.08) 50% / 반지름 5~50px 원 25% / blob(noise > 0.5) 25%. 이미지 중심 기준 반지름 90% 원 밖은 마스크 제거, 결함 영역은 어둡게 블렌딩
- **결함 없음 샘플**: 최종 50% 확률로 원본과 빈 마스크를 반환
- 반환값: `image`(복원 목표), `augmented_image`(합성 입력), `anomaly_mask`, `has_anomaly`
</details>

<details>
<summary>출력물 · 지표 · 체크포인트</summary>

- **확률맵 / 지표**: U-Net 결함 확률을 pixel score로, 21x21 평균 풀링 후 최댓값을 image score로 사용 → image/pixel AUC·AP를 `--csv_path`에 한 줄 추가 (`run_name,img_auc,pixel_auc,img_ap,pixel_ap`)
- **시각화**: `save_dir/{idx}_combined.png` = Input / Recon / Prob / Result. Prob은 `0.5*(1-SSIM) + 0.5*상대 복원오차` heatmap, Result는 이 맵(밝기 비례 임계값)과 U-Net 맵(> 0.1)의 AND 영역에 contour + bbox 표시 (10px 미만 파편 제외)
- **체크포인트 (매 epoch)**: `{run_name}epoch{N}.pckl` / `_seg.pckl`, optimizer·scheduler 포함 `{run_name}epoch{N}_full.pckl`, latest `{run_name}.pckl` / `_seg.pckl`, 클래스명 기반 `AE_high.pckl` / `AE_high_seg.pckl`. 검증 SSIM loss 갱신 시 `{run_name}best.pckl` / `best_seg.pckl`, `AE_high_best.pckl` / `_best_seg.pckl`
- `run_name` 규칙: `DRAEM_{model}_640_{lr}_{epochs}_bs{bs}_` (`util.build_run_name`)
</details>

<details>
<summary>전처리 환경변수</summary>

| 변수 | 기본값 | 사용 스크립트 |
|---|---|---|
| `DATA_ROOT` | `./data/ceramics` | 전체 (`config.py` 포함) |
| `RUNS_ROOT` | `./runs` | `config.py` |
| `INPUT_FOLDERS`, `TRAIN_OUTPUT`, `VAL_OUTPUT`, `VAL_RATIO` | 코드 내 기본 폴더 4종, `$DATA_ROOT/train`, `$DATA_ROOT/val`, `0.1` | `preprocess_all.py` |
| `TEST_INPUT`, `TEST_OUTPUT` | `$DATA_ROOT/test_orig`, `$DATA_ROOT/test` | `preprocess_test.py` |
| `AUG_INPUT`, `AUG_OUTPUT`, `AUG_COUNT`, `AUG_MODE` | 코드 내 기본 폴더, `$DATA_ROOT/train`, `14`, `basic` | `augment_dataset.py` |
| `PATCH_INPUT_DIR`, `PATCH_OUTPUT_DIR`, `PATCH_PADDING` | `$DATA_ROOT/test_orig`, `$DATA_ROOT/defect_patches`, `25` | `extract_defect_patches.py` |
| `NUM_WORKERS` | CPU 코어 수 (`preprocess_all.py`는 최대 8) | 전처리 공통 |
</details>

<details>
<summary>후속 실험: Tile V4 요약</summary>

- 입력: 1024x1024 → 512 타일 5장 (4코너 + 중앙, overlap 64). 추론은 `TileInferenceDataset` + `stitch_tiles`로 원본 해상도 복원
- 모델: `vit_large_patch16_dinov3.lvd1689m` (timm) 인코더 동결(`--freeze_encoder`, 기본 on) + U-Net 스타일 디코더, AdamW + CosineAnnealingLR
- 손실: Focal Tversky (α=0.85, β=0.15, γ=0.75) — 미검(FN)에 더 큰 패널티
- 합성: 실제 패치 + 절차적 crack, 원형 제품 영역 제약, 중첩 방지, epoch별 강도 스케줄(`set_epoch`)
- 검증: 실제 결함 이미지에 대한 IoU / Dice. 실행 예시와 비교표는 [experiments/tile_v4/README.md](experiments/tile_v4/README.md)
</details>

## 한계와 다음 단계

- **결과 수치 비공개**: 데이터·체크포인트·성능 수치가 없어 이 저장소만으로는 결과를 재현할 수 없습니다. 공개 데이터셋(예: MVTec AD)으로 동일 파이프라인을 재현하는 것이 다음 단계입니다.
- **작은 결함 소실**: 640x640 다운샘플링 한계는 tile_v4에서 실험했지만 루트 파이프라인과 통합되지 않았고, tile_v4는 별도 추론 스크립트 없이 학습 중 검증에서만 타일 stitching을 수행합니다. timm·DINOv3 가중치는 `requirements.txt`에 포함되어 있지 않습니다.
- **데이터 특화 설정**: FFT 파라미터(`CROP_R=170`, `PAD=8`), 원형 유효 영역, 시각화 임계값은 원본 촬영 조건에 맞춘 수동 튜닝 값이라 다른 데이터에서는 재조정이 필요합니다.
- **오래된 환경 고정**: `requirements.txt`는 Python 3.8 / PyTorch 1.8 / CUDA 10.2 시점 conda 스냅샷이며, 최신 환경 이식은 검증하지 않았습니다.

## 참고

- Zavrtanik et al., "DRAEM — A discriminatively trained reconstruction embedding for surface anomaly detection", ICCV 2021.
- Focal Loss: Lin et al., [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) (`loss.py` 구현 출처 주석 참고)
