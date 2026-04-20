"""DRAEM 추론 + 시각화 엔트리.

학습 완료된 AE + U-Net 체크포인트를 불러와 테스트 이미지에 대해
anomaly probability map 을 계산하고, Input/Recon/Prob/Overlay 4분할
시각화와 AUC/AP 지표를 함께 산출한다.
"""
import argparse
import importlib
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from data import get_test_dataloader
from loss import ssim
from model import load_model
from util import ensure_directory, write_results_to_file


def get_device(gpu_id):
    """CUDA → MPS → CPU 순으로 사용 가능한 디바이스 선택."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_reconstruction_model(model_class_name):
    """AE 계열 재구성 모델을 동적으로 임포트해서 반환."""
    model_module = importlib.import_module(config.model_module)
    if not hasattr(model_module, model_class_name):
        raise ValueError(f"Unsupported model class: {model_class_name}")
    return getattr(model_module, model_class_name)()


def build_segmentation_model(seg_model_class_name):
    """U-Net segmentation 네트워크 빌드 (2ch in / 2ch out)."""
    model_module = importlib.import_module(config.model_module)
    if not hasattr(model_module, seg_model_class_name):
        raise ValueError(f"Unsupported segmentation model class: {seg_model_class_name}")
    return getattr(model_module, seg_model_class_name)(in_channels=2, out_channels=2)


def resolve_model_paths(args):
    """base_model_name 기반 체크포인트 경로를 해석.

    --base_model_name 이 주어지면 checkpoint_path / base_model_name + (.pckl|.pth)
    조합으로 탐색. 못 찾으면 config.best_model_path 로 fallback.
    """
    if args.base_model_name:
        base_path = os.path.join(args.checkpoint_path, args.base_model_name)
        for extension in (".pckl", ".pth"):
            if os.path.exists(base_path + extension):
                model_path = base_path + extension
                seg_path = model_path.replace(extension, f"_seg{extension}")
                return model_path, seg_path
        if os.path.exists(base_path):
            extension = os.path.splitext(base_path)[1]
            seg_path = base_path.replace(extension, f"_seg{extension}")
            return base_path, seg_path
        raise FileNotFoundError(f"Could not find model file for {args.base_model_name}")

    model_path = args.model_path or config.best_model_path
    extension = os.path.splitext(model_path)[1]
    seg_path = model_path.replace(extension, f"_seg{extension}")
    return model_path, seg_path


def detect_circle_mask(image, shape):
    """유효 영역 마스크 placeholder. 현재는 전체 1 (원형 crop 을 쓰고 싶다면 여기서 구현)."""
    return np.ones((shape[0], shape[1]), dtype=np.float32)


def visualize_result(input_img, recon_img, anomaly_map, ssim_map, gt_mask, save_dir, image_index):
    """Input/Recon/Prob/Overlay 4분할 이미지를 PNG 로 저장.

    anomaly probability 는 (1 - SSIM) 과 상대 재구성 오차를 반반 섞어 만든다.
    U-Net segmentation 출력과 AND 로 결합해 최종 마스크를 정리.
    """
    input_img_u8 = np.clip(input_img * 255, 0, 255).astype(np.uint8)
    recon_img_u8 = np.clip(recon_img * 255, 0, 255).astype(np.uint8)

    # grayscale (C=1) 입력은 시각화 단계에서 3채널로 확장.
    if input_img_u8.shape[2] == 1:
        input_img_u8 = cv2.cvtColor(input_img_u8, cv2.COLOR_GRAY2BGR)
    if recon_img_u8.shape[2] == 1:
        recon_img_u8 = cv2.cvtColor(recon_img_u8, cv2.COLOR_GRAY2BGR)

    target_size = tuple(config.image_size)
    input_img_resized = cv2.resize(input_img_u8, target_size)
    recon_img_resized = cv2.resize(recon_img_u8, target_size)

    ssim_map_resized = cv2.resize(ssim_map, target_size)
    input_gray = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    recon_gray = cv2.cvtColor(recon_img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 상대 재구성 오차: 어두운 영역에서 분모가 너무 작아지지 않도록 0.1 로 clamp.
    diff_abs = np.abs(input_gray - recon_gray)
    diff_rel = diff_abs / np.maximum(input_gray, 0.1)
    diff_rel = np.clip(diff_rel, 0, 1.0)
    # 최종 probability = 0.5 * (1 - SSIM) + 0.5 * 상대오차.
    prob = 0.5 * (1.0 - ssim_map_resized) + 0.5 * diff_rel

    valid_mask = detect_circle_mask(input_gray, target_size)
    prob = prob * valid_mask

    # heatmap 시각화. 평균 밝기에 비례해 임계값을 주므로 어두운 장면에서 과검 감소.
    prob_u8 = (np.clip(prob * 3.0, 0, 1.0) * 255).astype(np.uint8)
    threshold_value = (0.1 + 0.2 * np.mean(input_gray)) * 255
    prob_colored = cv2.applyColorMap(prob_u8, cv2.COLORMAP_INFERNO)
    prob_colored[prob_u8 < threshold_value] = 0

    # U-Net anomaly map 과 SSIM 기반 prob 을 AND 로 합쳐 최종 마스크 확정.
    anomaly_map_resized = cv2.resize(anomaly_map, target_size)
    mask_prob = (prob_u8 > threshold_value).astype(np.uint8)
    mask_anomaly = (anomaly_map_resized > 0.1).astype(np.uint8)
    mask_final = cv2.bitwise_and(mask_prob, mask_anomaly)

    overlay_img = input_img_resized.copy()
    overlay_red = np.zeros_like(input_img_resized)
    overlay_red[:, :, 2] = 255
    overlay_img[mask_final > 0] = cv2.addWeighted(
        input_img_resized[mask_final > 0],
        0.7,
        overlay_red[mask_final > 0],
        0.3,
        0,
    )

    # 작은 파편(면적 < 10px) 은 노이즈로 보고 스킵. 나머지는 contour+bbox 표시.
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        cv2.drawContours(overlay_img, [contour], -1, (0, 255, 0), 1)
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(overlay_img, (x, y), (x + width, y + height), (0, 0, 255), 2)

    def put_text(image, text):
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    combined_img = np.hstack(
        (
            put_text(input_img_resized.copy(), "Input"),
            put_text(recon_img_resized.copy(), "Recon"),
            put_text(prob_colored, "Prob"),
            put_text(overlay_img, "Result"),
        )
    )
    cv2.imwrite(os.path.join(save_dir, f"{image_index}_combined.png"), combined_img)


def test_model(args):
    """테스트 루프: 이미지별 재구성 + segmentation + 시각화 + AUC/AP 집계."""
    device = get_device(args.gpu_id)
    model_path, seg_model_path = resolve_model_paths(args)
    run_name = args.base_model_name or os.path.splitext(os.path.basename(model_path))[0]

    model = build_reconstruction_model(args.model_class_name)
    model_seg = build_segmentation_model(args.seg_model_class_name)
    load_model(model, model_path, device)
    load_model(model_seg, seg_model_path, device)

    ensure_directory(args.save_dir)
    dataloader = get_test_dataloader(args.data_path, image_size=config.image_size)
    img_dim = config.image_size[0]

    # pixel-level 지표 계산을 위해 전체 픽셀을 flatten 으로 누적.
    total_pixel_scores = np.zeros(img_dim * img_dim * len(dataloader.dataset))
    total_gt_pixel_scores = np.zeros(img_dim * img_dim * len(dataloader.dataset))
    anomaly_score_gt = []
    anomaly_score_prediction = []
    mask_count = 0

    for batch_index, sample in enumerate(dataloader):
        image_batch = sample["image"].to(device)
        true_mask = sample["mask"]
        gt_mask = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))
        anomaly_score_gt.append(sample["has_anomaly"].detach().cpu().numpy()[0, 0])

        # 학습과 동일한 순서: AE 로 복원 → U-Net 에 [recon, input] 입력.
        reconstructed_batch = model(image_batch)
        segmentation_input = torch.cat((reconstructed_batch.detach(), image_batch), dim=1)
        segmentation_logits = model_seg(segmentation_input)
        segmentation_prob = torch.softmax(segmentation_logits, dim=1)

        pred_mask = segmentation_prob[0, 1, :, :].detach().cpu().numpy()
        # image-level score: 21x21 avg pool 로 국소 최대 anomaly 값을 뽑아 robust 하게 계산.
        pooled_prob = torch.nn.functional.avg_pool2d(
            segmentation_prob[:, 1:, :, :],
            21,
            stride=1,
            padding=21 // 2,
        ).cpu().detach().numpy()
        anomaly_score_prediction.append(np.max(pooled_prob))

        flat_true_mask = gt_mask.flatten()
        flat_pred_mask = pred_mask.flatten()
        offset = mask_count * img_dim * img_dim
        total_pixel_scores[offset:offset + img_dim * img_dim] = flat_pred_mask
        total_gt_pixel_scores[offset:offset + img_dim * img_dim] = flat_true_mask
        mask_count += 1

        if batch_index < 391:
            input_img = image_batch.detach().cpu().numpy()[0].transpose(1, 2, 0)
            recon_img = reconstructed_batch.detach().cpu().numpy()[0].transpose(1, 2, 0)
            _, ssim_map = ssim(image_batch, reconstructed_batch, window_size=11, size_average=False)
            visualize_result(
                input_img,
                recon_img,
                pred_mask,
                ssim_map.detach().cpu().numpy()[0, 0, :, :],
                true_mask.detach().cpu().numpy()[0, 0, :, :],
                args.save_dir,
                batch_index,
            )

    anomaly_score_gt = np.array(anomaly_score_gt)
    anomaly_score_prediction = np.array(anomaly_score_prediction)

    # AUC/AP 는 positive/negative 두 클래스가 모두 있어야 계산 가능.
    if len(np.unique(anomaly_score_gt)) <= 1:
        print("Only one class present in ground truth. Skipping metric calculation.")
        return

    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_count].astype(np.uint8)
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_count]

    image_auc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    image_ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
    pixel_auc = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    pixel_ap = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

    print("AUC Image: ", image_auc)
    print("AP Image: ", image_ap)
    print("AUC Pixel: ", pixel_auc)
    print("AP Pixel: ", pixel_ap)
    write_results_to_file(run_name, [image_auc], [pixel_auc], [image_ap], [pixel_ap], args.csv_path)


def build_parser():
    """CLI 인자 파서. 기본값은 모두 config 에서 가져온다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_model_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default=config.best_model_path)
    parser.add_argument("--data_path", type=str, default=config.test_image_paths)
    parser.add_argument("--checkpoint_path", type=str, default=config.model_save_path)
    parser.add_argument("--csv_path", type=str, default=config.csv_path)
    parser.add_argument("--save_dir", type=str, default=config.save_dir)
    parser.add_argument("--model_class_name", type=str, default=config.model_class_name)
    parser.add_argument("--seg_model_class_name", type=str, default=config.seg_model_class_name)
    return parser


def main():
    args = build_parser().parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
