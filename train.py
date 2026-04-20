"""DRAEM 학습 엔트리 포인트.

AE (재구성) + DiscriminativeSubNetwork (segmentation) 두 네트워크를 동시에 학습시키고,
매 epoch 마다 epoch / latest / class / best 4종류 체크포인트를 저장한다.
--resume auto 로 마지막 체크포인트 자동 재개가 가능.
"""
import argparse
import importlib
import os

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import config
from data import get_train_dataloader, get_valid_dataloader
from loss import FocalLoss, SSIM
from model import _extract_state_dict, _strip_module_prefix
from tensorboard_visualizer import TensorboardVisualizer
from util import (
    build_run_name,
    ensure_directory,
    get_learning_rate,
    init_weights,
    resolve_resume_path,
    restore_scheduler_from_log,
)


def get_device(gpu_id):
    """CUDA → MPS → CPU 순으로 사용 가능한 디바이스 선택."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_reconstruction_model(model_class_name):
    """config.model_module 에서 AE 계열 클래스를 동적으로 임포트."""
    model_module = importlib.import_module(config.model_module)
    if not hasattr(model_module, model_class_name):
        raise ValueError(f"Unsupported model class: {model_class_name}")
    return getattr(model_module, model_class_name)()


def build_segmentation_model(seg_model_class_name):
    """U-Net segmentation 네트워크 빌드. 입력 2채널(원본+재구성) → 2채널(정상/결함)."""
    model_module = importlib.import_module(config.model_module)
    if not hasattr(model_module, seg_model_class_name):
        raise ValueError(f"Unsupported segmentation model class: {seg_model_class_name}")
    return getattr(model_module, seg_model_class_name)(in_channels=2, out_channels=2)


def build_optimizer(parameters, optimizer_type, learning_rate):
    if optimizer_type == "adam":
        return optim.Adam(parameters, lr=learning_rate)
    if optimizer_type == "adamw":
        return optim.AdamW(parameters, lr=learning_rate)
    if optimizer_type == "sgd":
        return optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if optimizer_type == "rmsprop":
        return optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(optimizer, scheduler_type, learning_rate):
    """scheduler 선택. plateau 는 val SSIM loss 기준, step/cosine 은 epoch 기준."""
    if scheduler_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=0,
            verbose=True,
        )
    if scheduler_type == "step":
        return lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return None


def load_module_state(module, state_dict):
    """DataParallel 저장 / dict 래핑 등 다양한 형태의 state_dict 를 정규화해서 로드."""
    state_dict = _strip_module_prefix(_extract_state_dict(state_dict))
    module.load_state_dict(state_dict)


def load_resume_checkpoint(
    resume_path,
    model,
    model_seg,
    optimizer,
    scheduler,
    train_loader_length,
    history_log,
):
    """체크포인트 재개. full 체크포인트와 weight-only 체크포인트 모두 지원.

    - full (dict with 'model_state_dict'): optimizer/scheduler/epoch/n_iter 까지 복원.
    - weight-only: seg 체크포인트는 파일명 규칙으로 찾고, epoch 는 파일명에서 파싱.
      scheduler 상태는 training_log 를 역으로 재생해서 복원.
    """
    start_epoch = 0
    best_ssim_loss = float("inf")
    n_iter = 0

    if not resume_path or not os.path.exists(resume_path):
        return start_epoch, best_ssim_loss, n_iter

    checkpoint = torch.load(resume_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # full 체크포인트: optimizer / scheduler 까지 같이 들어 있음.
        load_module_state(model, checkpoint["model_state_dict"])
        load_module_state(model_seg, checkpoint["model_seg_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_ssim_loss = checkpoint.get("best_ssim_loss", float("inf"))
        n_iter = checkpoint.get("n_iter", 0)
    else:
        # weight-only 체크포인트: 파일명 규약으로 seg 체크포인트 추정.
        load_module_state(model, checkpoint)

        seg_resume_path = resume_path.replace(".pckl", "_seg.pckl").replace(".pth", "_seg.pth")
        if os.path.exists(seg_resume_path):
            seg_checkpoint = torch.load(seg_resume_path, map_location="cpu")
            load_module_state(model_seg, seg_checkpoint)

        # 파일명이 '...epochN.pckl' 이면 N 을 start_epoch 로 사용.
        epoch_token = os.path.basename(resume_path)
        if "epoch" in epoch_token:
            try:
                start_epoch = int(epoch_token.split("epoch")[1].split(".")[0].split("_")[0])
            except (IndexError, ValueError):
                start_epoch = 0
        n_iter = start_epoch * train_loader_length

        # plateau scheduler 는 log 를 순차 재생해야 state 가 복원됨.
        if scheduler is not None:
            restore_scheduler_from_log(scheduler, history_log, start_epoch)

    return start_epoch, best_ssim_loss, n_iter


def train_model(args):
    """학습 메인 루프: 모델/옵티마이저 구성 → 재개 → epoch 반복."""
    device = get_device(args.gpu_id)
    ensure_directory(args.checkpoint_path)
    ensure_directory(args.log_path)

    # run_name 은 체크포인트/로그 파일의 공통 prefix 로 사용된다.
    run_name = build_run_name(args.model_class_name, args.lr, args.epochs, args.bs)
    history_log = args.history_log or os.path.join(args.log_path, f"{run_name}.log")
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name)) if args.visualize else None

    # AE (재구성) + U-Net (segmentation) 두 네트워크를 같이 학습.
    model = build_reconstruction_model(args.model_class_name).to(device)
    model_seg = build_segmentation_model(args.seg_model_class_name).to(device)
    init_weights(model, init_type=args.init_type)
    init_weights(model_seg, init_type=args.init_type)

    # 두 네트워크의 파라미터를 하나의 optimizer 에서 동시에 업데이트.
    optimizer = build_optimizer(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": model_seg.parameters(), "lr": args.lr},
        ],
        args.optimizer_type,
        args.lr,
    )
    scheduler = build_scheduler(optimizer, args.scheduler_type, args.lr)

    # DRAEM 표준 손실 조합: 재구성 MSE + SSIM, segmentation Focal.
    loss_l2 = torch.nn.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    train_loader = get_train_dataloader(
        args.data_path,
        anomaly_source_path=args.anomaly_source_path,
        batch_size=args.bs,
        image_size=config.image_size,
        num_workers=config.num_workers,
    )

    valid_loader = None
    if args.valid_path:
        valid_loader = get_valid_dataloader(
            args.valid_path,
            batch_size=args.bs,
            image_size=config.image_size,
            num_workers=config.num_workers,
        )

    resume_path = resolve_resume_path(args.checkpoint_path, run_name, args.resume)
    start_epoch, best_ssim_loss, n_iter = load_resume_checkpoint(
        resume_path,
        model,
        model_seg,
        optimizer,
        scheduler,
        len(train_loader),
        history_log,
    )

    if resume_path:
        print(f"[Resume] Loaded checkpoint: {resume_path}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model_seg.train()

        with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
        ) as progress:
            for _, sample in progress:
                image_batch = sample["image"].to(device)
                augmented_batch = sample["augmented_image"].to(device)
                anomaly_mask = sample["anomaly_mask"].to(device)

                # AE 는 augmented 입력을 원본(image_batch) 으로 복원하도록 학습.
                reconstructed_batch = model(augmented_batch)
                # U-Net 입력: 재구성 + 증강 두 채널 concat → anomaly mask 예측.
                segmentation_input = torch.cat((reconstructed_batch, augmented_batch), dim=1)
                segmentation_logits = model_seg(segmentation_input)
                segmentation_prob = torch.softmax(segmentation_logits, dim=1)

                # 재구성 품질(MSE+SSIM) + segmentation 품질(Focal) 합산.
                l2_loss = loss_l2(reconstructed_batch, image_batch)
                ssim_loss = loss_ssim(reconstructed_batch, image_batch)
                segment_loss = loss_focal(segmentation_prob, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.set_postfix(
                    {
                        "Train Loss": loss.item(),
                        "SSIM Loss": ssim_loss.item(),
                        "Seg Loss": segment_loss.item(),
                        "L2 Loss": l2_loss.item(),
                        "LR": get_learning_rate(optimizer),
                    }
                )

                if visualizer is not None and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name="l2_loss")
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name="ssim_loss")
                    visualizer.plot_loss(segment_loss, n_iter, loss_name="segment_loss")
                if visualizer is not None and n_iter % 400 == 0:
                    predicted_mask = segmentation_prob[:, 1:, :, :]
                    visualizer.visualize_image_batch(augmented_batch, n_iter, image_name="batch_augmented")
                    visualizer.visualize_image_batch(image_batch, n_iter, image_name="batch_target")
                    visualizer.visualize_image_batch(reconstructed_batch, n_iter, image_name="batch_reconstructed")
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name="mask_target")
                    visualizer.visualize_image_batch(predicted_mask, n_iter, image_name="mask_output")

                n_iter += 1

        val_ssim_loss = None
        if valid_loader is not None:
            model.eval()
            total_ssim_loss = 0.0
            sample_count = 0

            with torch.no_grad():
                with tqdm(
                    enumerate(valid_loader),
                    total=len(valid_loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs} [Valid]",
                ) as progress:
                    for batch_index, sample in progress:
                        image_batch = sample["image"].to(device)
                        reconstructed_batch = model(image_batch)

                        l2_loss = loss_l2(reconstructed_batch, image_batch)
                        ssim_loss = loss_ssim(reconstructed_batch, image_batch)
                        loss = l2_loss + ssim_loss

                        total_ssim_loss += ssim_loss.item()
                        sample_count += 1
                        progress.set_postfix({"Val Loss": loss.item(), "Val SSIM": ssim_loss.item()})

                        if visualizer is not None and batch_index == 0:
                            visualizer.visualize_image_batch(image_batch, epoch, image_name="val_input")
                            visualizer.visualize_image_batch(reconstructed_batch, epoch, image_name="val_reconstructed")

            val_ssim_loss = total_ssim_loss / max(1, sample_count)
            print(f"[Epoch {epoch + 1}] Validation SSIM Loss: {val_ssim_loss:.6f}")
            with open(history_log, "a", encoding="utf-8") as file:
                file.write(f"[Epoch {epoch + 1}] Validation SSIM Loss: {val_ssim_loss:.6f}\n")

            if visualizer is not None:
                visualizer.plot_loss(torch.tensor(val_ssim_loss), epoch, loss_name="val_ssim_loss")

            if args.scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_ssim_loss)
        elif scheduler is not None and args.scheduler_type != "plateau":
            scheduler.step()

        # 매 epoch 마다 4종 체크포인트 저장:
        #   1) epoch 단위 (백업)
        epoch_model_path = os.path.join(args.checkpoint_path, f"{run_name}epoch{epoch + 1}.pckl")
        epoch_seg_path = os.path.join(args.checkpoint_path, f"{run_name}epoch{epoch + 1}_seg.pckl")
        torch.save(model.state_dict(), epoch_model_path)
        torch.save(model_seg.state_dict(), epoch_seg_path)

        #   2) full 체크포인트 (optimizer/scheduler 포함 — 재개용)
        full_checkpoint_path = os.path.join(args.checkpoint_path, f"{run_name}epoch{epoch + 1}_full.pckl")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "model_seg_state_dict": model_seg.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
                "best_ssim_loss": best_ssim_loss,
                "n_iter": n_iter,
            },
            full_checkpoint_path,
        )

        #   3) run_name 기반 latest (run 별 재개 편의)
        latest_model_path = os.path.join(args.checkpoint_path, f"{run_name}.pckl")
        latest_seg_path = os.path.join(args.checkpoint_path, f"{run_name}_seg.pckl")
        torch.save(model.state_dict(), latest_model_path)
        torch.save(model_seg.state_dict(), latest_seg_path)

        #   4) 클래스명 기반 latest (test.py 기본값과 매칭)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"{args.model_class_name}.pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, f"{args.model_class_name}_seg.pckl"))

        # best 갱신 시에만 best 접미사 파일도 추가 저장.
        if val_ssim_loss is not None and val_ssim_loss < best_ssim_loss:
            best_ssim_loss = val_ssim_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"{run_name}best.pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, f"{run_name}best_seg.pckl"))
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"{args.model_class_name}_best.pckl"))
            torch.save(
                model_seg.state_dict(),
                os.path.join(args.checkpoint_path, f"{args.model_class_name}_best_seg.pckl"),
            )


def build_parser():
    """CLI 인자 파서. 모든 기본값은 config 모듈에서 가져온다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=config.base_batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=config.train_image_paths)
    parser.add_argument("--valid_path", type=str, default=config.val_image_paths)
    parser.add_argument("--anomaly_source_path", type=str, default=config.anomaly_source_path)
    parser.add_argument("--checkpoint_path", type=str, default=config.model_save_path)
    parser.add_argument("--log_path", type=str, default=os.path.dirname(config.log_file))
    parser.add_argument("--history_log", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--model_class_name", type=str, default=config.model_class_name)
    parser.add_argument("--seg_model_class_name", type=str, default=config.seg_model_class_name)
    parser.add_argument("--optimizer_type", type=str, default=config.optimizer_type)
    parser.add_argument("--scheduler_type", type=str, default=config.scheduler_type)
    parser.add_argument("--init_type", type=str, default=config.init_type)
    parser.add_argument("--visualize", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
