"""학습 루프에서 공용으로 쓰는 유틸.

- 학습률 조회, weight 초기화
- run name 생성 / 체크포인트 재개 경로 탐색
- scheduler 로그 기반 복원
- 평가 결과 CSV 기록
"""
import glob
import os
import re

import numpy as np
import torch


def get_learning_rate(optimizer):
    """첫 번째 param group 의 lr 반환 (tqdm 진행 표시 용)."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def init_weights(model, init_type="xavier"):
    """Conv2d / Linear 에 대해 xavier 또는 kaiming 초기화 적용."""
    if init_type == "xavier":
        def init_xavier(module):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        model.apply(init_xavier)
    elif init_type == "kaiming":
        def init_kaiming(module):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        model.apply(init_kaiming)


def ensure_directory(path):
    """디렉토리 생성 (이미 있으면 무시)."""
    os.makedirs(path, exist_ok=True)


def build_run_name(model_class_name, learning_rate, num_epochs, batch_size, prefix="DRAEM"):
    """체크포인트 파일명의 공통 prefix 생성.

    예: DRAEM_AE_high_640_0.0001_700_bs2_
    train.py / test.py 양쪽에서 동일한 규칙을 쓰므로 매칭이 자연스럽다.
    """
    return f"{prefix}_{model_class_name}_640_{learning_rate}_{num_epochs}_bs{batch_size}_"


def resolve_resume_path(checkpoint_path, run_name, resume_path=""):
    """--resume 값에 따라 실제 로드할 체크포인트 경로를 결정.

    - 빈 문자열: 재개하지 않음 → None
    - 실제 파일 경로: 그대로 사용
    - 'auto' / 'latest': 해당 run_name 의 최신 epoch 체크포인트 탐색,
      _full.pckl 이 있으면 그걸 우선 (optimizer/scheduler 상태 포함).
    """
    if not resume_path:
        return None

    if resume_path not in {"auto", "latest"} and os.path.isfile(resume_path):
        return resume_path

    # run_name 패턴에 맞는 epoch 체크포인트만 수집 (seg / full 은 제외).
    pattern = os.path.join(checkpoint_path, f"{run_name}epoch*.pckl")
    checkpoints = glob.glob(pattern)
    checkpoints = [path for path in checkpoints if "_seg" not in path and "_full" not in path]

    if not checkpoints:
        return None

    def get_epoch(path):
        match = re.search(r"epoch(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=get_epoch)
    # _full.pckl 이 존재하면 우선 반환 (optimizer / scheduler 포함 상태).
    full_checkpoint_path = checkpoints[-1].replace(".pckl", "_full.pckl")
    if os.path.exists(full_checkpoint_path):
        return full_checkpoint_path
    return checkpoints[-1]


def restore_scheduler_from_log(scheduler, log_file, start_epoch):
    """구버전 체크포인트(weight only) 재개 시 scheduler 를 로그로 복원.

    training_log.txt 에서 '[Epoch N] Validation SSIM Loss: X' 라인을 파싱해
    start_epoch 까지 순차적으로 scheduler.step(loss) 를 호출.
    """
    if not log_file or not os.path.exists(log_file):
        return

    val_losses = {}
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            match = re.search(r"\[Epoch (\d+)\] Validation SSIM Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss_value = float(match.group(2))
                val_losses[epoch] = loss_value

    for epoch in range(1, start_epoch + 1):
        if epoch in val_losses:
            scheduler.step(val_losses[epoch])


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap, csv_path):
    """테스트 결과를 CSV 한 줄 append. 파일이 없으면 헤더 먼저 기록."""
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        ensure_directory(output_dir)

    header = "run_name,img_auc,pixel_auc,img_ap,pixel_ap\n"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as file:
            file.write(header)

    result_line = (
        f"{run_name},{np.mean(image_auc):.3f},{np.mean(pixel_auc):.3f},"
        f"{np.mean(image_ap):.3f},{np.mean(pixel_ap):.3f}\n"
    )

    with open(csv_path, "a+", encoding="utf-8") as file:
        file.write(result_line)
