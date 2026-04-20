"""DRAEM 학습에 사용하는 Loss 모음.

- FocalLoss: Dense object detection 논문에서 제안된 hard-example 집중형 CE.
  segmentation 브랜치의 클래스 불균형 (대부분 배경, 결함이 극소) 완화에 사용.
- SSIM: Structural Similarity. 재구성 브랜치의 시각적 품질을 평가하기 위한 지표
  (1 - SSIM) 을 loss 로 쓴다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class FocalLoss(nn.Module):
    """Focal Loss (https://arxiv.org/abs/1708.02002).

    원본 구현 출처:
        https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py

    수식: Focal_Loss = -alpha * (1 - pt) ** gamma * log(pt)

    Args:
        apply_nonlin: logit 에 적용할 softmax/sigmoid 등 비선형 (None 이면 이미 확률값 가정).
        alpha: 클래스별 가중치 (float / list / None).
        gamma: 잘 분류된 샘플의 loss 기여도를 감쇠하는 지수. 커질수록 hard-example 집중.
        balance_index: alpha 가 float 일 때 해당 클래스에 alpha 를 배정.
        smooth: one-hot 클리핑 및 log 안정화용 작은 값.
        size_average: True 면 mean, False 면 per-sample loss 반환.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        # 필요 시 softmax 등 비선형 적용 (이미 확률이면 생략).
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        # 공간 차원이 있는 경우 (N,C,H,W) → (N*H*W, C) 로 평탄화.
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        # alpha 형식별 분기 (None / list / float).
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            # float 이면 balance_index 클래스에만 해당 값, 나머지는 (1-alpha).
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        # target 을 one-hot 으로 변환.
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # label smoothing: 1 → (1 - smooth), 0 → smooth/(C-1).
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        # pt: 정답 클래스 확률.
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        # sample 별 alpha 조회 및 최종 loss.
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


def gaussian(window_size, sigma):
    """1D Gaussian kernel (합=1)."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    """SSIM 계산용 2D Gaussian window 생성 후 channel 수만큼 복제."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 1D x 1D^T = 2D Gaussian.
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # (C, 1, W, W) 형태로 expand → groups=C depthwise conv 에 바로 사용.
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """Structural Similarity Index Measure.

    local 평균/분산/공분산을 Gaussian window 로 계산해 두 이미지의 구조적 유사도 측정.
    반환: (mean SSIM, ssim_map) 또는 full=True 면 (mean SSIM, contrast_sensitivity).
    """
    # 입력 range 자동 감지 (0~255 vs 0~1, 부호 유무).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size//2
    (_, channel, height, width) = img1.size()
    if window is None:
        # 이미지가 window 보다 작을 경우 대비해 실제 크기 제한.
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 각 채널별 Gaussian 평균 (depthwise conv).
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 분산 / 공분산 = E[X^2] - E[X]^2 형태로 계산.
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # SSIM 안정화용 상수 (Wang et al., 2004).
    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    # contrast sensitivity 항 (SSIM 분해식 중 structure*contrast 부분).
    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)

    # 최종 SSIM map.
    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        # batch 차원만 유지하며 나머지 축 평균.
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class SSIM(torch.nn.Module):
    """SSIM loss 모듈. forward 는 (1 - SSIM) 을 반환해 최소화 가능.

    창(window) 은 cache 하며, 입력 채널/dtype 이 바뀌면 재생성.
    """

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # grayscale 1채널 가정 초기값.
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # 입력 채널/dtype 이 동일하면 캐시 재사용, 아니면 새로 생성.
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        # 1 - SSIM 을 loss 로 반환 (SSIM 이 클수록 loss 가 작음).
        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        return 1.0 - s_score
