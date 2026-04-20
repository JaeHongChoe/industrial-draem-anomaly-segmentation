"""학습/추론 공용 설정.

train.py / test.py 는 argparse 기본값으로 이 모듈 값을 참조한다.
경로는 DATA_ROOT / RUNS_ROOT 환경변수로 덮어쓸 수 있도록 f-string 기반으로 구성.
"""

import os
import importlib

# 데이터 루트 / 학습 산출물 루트. 환경변수로 덮어쓸 수 있다.
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/ceramics")
RUNS_ROOT = os.environ.get("RUNS_ROOT", "./runs")

# 학습 관련 경로
train_image_paths = f'{DATA_ROOT}/train/*.jpg'
val_image_paths = f'{DATA_ROOT}/val/*.jpg'
experiment_name = 'DRAEM_AE_high_ceramics'
model_save_path = f'{RUNS_ROOT}/{experiment_name}/model'
log_file = f'{RUNS_ROOT}/{experiment_name}/model/training_log.txt'
loss_plot_path = f'{RUNS_ROOT}/{experiment_name}/model/loss_plot.png'

# 추론 관련 경로
test_image_paths = f"{DATA_ROOT}/test"
best_model_path = f'{RUNS_ROOT}/{experiment_name}/model/AE_high_best.pckl'
save_dir = f"{RUNS_ROOT}/{experiment_name}/images"
csv_path = f"{RUNS_ROOT}/{experiment_name}/DRAEM_result.csv"
threshold = 0.4

# 모델 클래스 선택. model.py 에 정의된 클래스 이름과 매핑된다.
# AE_high 외에도 AE, AE_half, AE_high_half, RAE 계열이 존재하지만
# 현재 기본 파이프라인은 AE_high 한 가지만 사용한다.
model_module = 'model'
model_class_name = 'AE_high'
seg_model_class_name = 'DiscriminativeSubNetwork'

device = 'cuda'

# 학습 세부 설정
model = 'AE_high'
seg_model = 'DiscriminativeSubNetwork'
init_type = 'xavier'
optimizer_type = 'adamw'
scheduler_type = 'plateau'
loss_type = 'mse+ssim+focal'
tv_weight = 0.0000001
noise_robust_loss_type = 'huber'
noise_robust_loss_weight = 0.001
anomaly_source_path = ''
# channels = 1

# 현재 파이프라인에서는 data_loader_black.py 내부 증강을 쓰므로 아래 설정은 참고용.
augmentations = {
    'RandomHorizontalFlip': 0.5,
    'RandomVerticalFlip': 0.5,
    'RandomRotation': 50,
    'ColorJitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2},
    # 'GaussianBlur': {'kernel_size': 3},
}
num_augmentations = 300

image_size = (640, 640)
seed = 8888
num_workers = 8
base_batch_size = 6
num_epochs = 1000
learning_rate = 0.001
early_stopping_patience = 100

# MAE/ViT 계열 하이퍼파라미터 (실험 과정에서 쓰이던 값, 현재 기본 DRAEM 흐름에선 미사용).
img_size=640
patch_size=32
in_channels=1
mae_emb_dim=1024
encoder_depth=12
decoder_depth=8
num_heads=16
masking_ratio=0.75
