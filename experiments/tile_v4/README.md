# Tile-based DINOv3 Segmentation (Experiment V4)

루트의 DRAEM(AE + Discriminative U-Net) 파이프라인 다음 단계로 진행한 실험입니다.
640x640 전체 이미지 한 장으로 학습하면 작은 결함이 다운샘플링 과정에서 사라지는 문제가 있어,
**고해상도(1024) 이미지를 512 타일로 나눠 학습하고, 사전학습 ViT 인코더로 표현력을 보강**하는 방향을 시험했습니다.

## 루트 DRAEM 대비 달라진 점

| 항목 | 루트 DRAEM | Tile V4 |
|---|---|---|
| 입력 | 640x640 전체 이미지 | 1024x1024 → 512 타일 5장 (4코너 + 중앙, overlap 64) |
| 구조 | 재구성 AE + 분할 U-Net (2-stage) | DINOv3 ViT-L/16 인코더(동결) + U-Net 스타일 디코더 (1-stage 분할) |
| Loss | MSE + SSIM + Focal | Focal Tversky (α=0.85, β=0.15 → 미검(FN)에 더 큰 패널티) |
| 결함 합성 | 실제 결함 패치 50% + Perlin noise 50% | 실제 결함 패치 + 절차적 크랙(직선/곡선/분기) 합성, 원형 제품 영역 제약·중첩 방지·epoch별 강도 스케줄 |
| 검증 | image/pixel AUC, AP | 실제 결함 이미지에 대한 IoU / Dice (타일 추론 후 stitching) |

## 파일 구성

```text
tile_v4/
├── prepare_dataset.py         # FFT central-crop + 후처리 체인 (1024 리사이즈)
├── extract_defect_patches.py  # LabelMe polygon → RGBA 결함 패치
├── data_loader.py             # TileTrainDataset / TileInferenceDataset / stitch_tiles
├── train.py                   # DinoSegModelV3 학습, focal tversky loss, 실결함 IoU/Dice 검증
├── tensorboard_visualizer.py
└── train.slurm                # SLURM 실행 예시 (2 GPU)
```

## 핵심 설계 포인트

- **타일 학습 + stitching 추론**: 학습은 타일 단위, 추론은 `TileInferenceDataset`으로 전 타일을 뽑아
  `stitch_tiles`로 overlap 영역을 평균해 원본 해상도 확률맵으로 복원합니다.
- **동결 인코더**: 정상 데이터는 많지만 결함 라벨은 적은 상황이라, DINOv3 인코더를 동결하고 디코더만 학습해
  과적합을 줄였습니다 (`--freeze_encoder`, 기본 on).
- **불균형 대응 loss**: 결함 픽셀이 극소수라 Dice/Tversky 계열을 쓰고, FN 가중치를 높여 미검을 우선 줄이도록 했습니다.
- **현실적인 합성 결함**: 결함이 제품 원형 영역 밖에 찍히거나 서로 겹치지 않도록 배치 제약을 두고,
  학습 초반엔 약하게·후반엔 강하게 합성 강도를 올립니다 (`set_epoch`).

## 실행 예시

```bash
python3 train.py \
  --train_dirs data/train \
  --real_defect_path data/test_defect \
  --anomaly_source_path data/defect_patches \
  --checkpoint_path ./checkpoints \
  --log_path ./logs \
  --gpus 0,1 --bs 8 --epochs 100 --visualize
```

`timm`에서 `vit_large_patch16_dinov3.lvd1689m` 가중치를 내려받을 수 있는 환경이어야 합니다.
데이터·결함 패치·체크포인트는 저장소에 포함하지 않습니다.
