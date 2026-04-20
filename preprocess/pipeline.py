"""FFT 중앙 crop + 후처리 체인 기반 공용 전처리 파이프라인.

모든 preprocess_* 스크립트가 이 모듈의 preprocess_file / preprocess_gray 를 공유해
동일한 FFT → 후처리 → 리사이즈 결과를 내도록 통일한다.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# DATA_ROOT: 데이터 루트. 모든 하위 스크립트의 기본 경로가 이 값을 기준으로 계산된다.
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/ceramics")

# 최종 출력 해상도 및 FFT 파라미터.
# CROP_R 은 FFT 중앙 원형 마스크의 반지름. 원본 카메라/해상도 기준으로 튜닝된 값이라
# 다른 해상도에서는 같이 조정해야 한다.
TARGET_SIZE = (640, 640)
PAD = 8
CROP_R = 170

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def to_gray(img: np.ndarray) -> np.ndarray:
    """컬러/알파 포함 입력을 1채널 grayscale 로 변환."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def imread_unicode(path: str) -> np.ndarray | None:
    """한글/특수문자 경로에서도 안전하게 읽히도록 fromfile + imdecode 사용."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def fft2_pad(gray: np.ndarray, pad: int = PAD):
    """경계 반사 패딩 후 2D FFT 수행. 경계 아티팩트를 줄이기 위함."""
    g = gray.astype(np.float32)
    big = cv2.copyMakeBorder(g, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    F = np.fft.fftshift(np.fft.fft2(big))
    return F, big.shape, pad


def ifft2_unpad(F2: np.ndarray, shape: tuple, pad: int) -> np.ndarray:
    """역 FFT 후 패딩을 떼서 원래 크기로 복원."""
    H, W = shape
    out_big = np.fft.ifft2(np.fft.ifftshift(F2))
    out_big = np.real(out_big)
    return out_big[pad:H - pad, pad:W - pad]


def tone_match(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """FFT 필터링으로 어두워진/밝아진 톤을 원본 평균·표준편차에 맞춰 보정."""
    m0, s0 = cv2.meanStdDev(ref)
    m0, s0 = float(m0), float(s0) + 1e-6
    m1, s1 = cv2.meanStdDev(out)
    m1, s1 = float(m1), float(s1) + 1e-6
    out = (out - m1) * (s0 / s1) + m0
    return np.clip(out, 0, 255).astype(np.uint8)


def fft_central_crop(gray: np.ndarray, crop_r: int = CROP_R, pad: int = PAD) -> np.ndarray:
    """주파수 도메인에서 중앙 원형 영역만 남기고 복원 후 톤 매칭.

    고주파(외곽) 성분을 잘라서 scan line noise / 고주파 텍스처를 날리되,
    전체 밝기 분포는 tone_match 로 원본을 따라간다.
    """
    F, shape_big, pad_sz = fft2_pad(gray, pad)
    H, W = F.shape
    cy, cx = H // 2, W // 2
    # 중앙 기준 거리 맵 → 반경 crop_r 이내만 통과시키는 원형 저역 통과 마스크.
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = (R <= crop_r).astype(np.float32)
    F_crop = F * mask
    out = ifft2_unpad(F_crop, shape_big, pad_sz)
    return tone_match(out, gray.astype(np.float32))


def gaussian_pre(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """약한 가우시안 블러로 고주파 잔여 노이즈 감쇠."""
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def median_pre(img: np.ndarray, k: int = 3) -> np.ndarray:
    """소금후추 노이즈 / 튀는 픽셀 제거용 미디안 필터."""
    return cv2.medianBlur(img, k)


def wiener_local(img: np.ndarray, ksize: int = 5, noise_var: float = 100.0) -> np.ndarray:
    """국소 Wiener 필터. 국소 분산이 노이즈 분산보다 낮으면 평균으로 끌어내려 평탄화."""
    f = img.astype(np.float32)
    ones = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    mean = cv2.filter2D(f, -1, ones, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(f * f, -1, ones, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    # gain = max(var - noise_var, 0) / var. 신호 대비 노이즈 비율에 따라 평균 쪽으로 당기는 세기.
    gain = np.maximum(var - noise_var, 0.0) / (var + 1e-6)
    out = mean + gain * (f - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretch(img: np.ndarray, low_perc: int = 10, high_perc: int = 97) -> np.ndarray:
    """하위/상위 percentile 을 [0, 255] 로 스트레칭해 대비를 키운다."""
    low, high = np.percentile(img, (low_perc, high_perc))
    if high - low < 1e-6:
        return img.astype(np.uint8)
    out = np.clip((img - low) * 255.0 / (high - low + 1e-6), 0, 255)
    return out.astype(np.uint8)


def adjust_gamma(img: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """룩업 테이블 기반 감마 보정. gamma > 1 → 어두운 쪽을 밝혀 결함 대비를 키움."""
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def sharpen_unsharp(img: np.ndarray, amount: float = 2.2, sigma: float = 0.5) -> np.ndarray:
    """Unsharp mask: 블러와의 차이를 더해 엣지 강조."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def post_preprocess(img: np.ndarray) -> np.ndarray:
    """Gaussian → Median → Wiener → contrast stretch → gamma → unsharp 순서.

    순서가 중요: 노이즈 먼저 줄이고 → 대비/감마로 분포 확장 → 마지막에 샤프닝.
    """
    out = img.copy()
    out = gaussian_pre(out, sigma=1.0)
    out = median_pre(out, k=3)
    out = wiener_local(out, ksize=5, noise_var=100.0)
    out = contrast_stretch(out, low_perc=10, high_perc=97)
    out = adjust_gamma(out, gamma=1.4)
    out = sharpen_unsharp(out, amount=2.2, sigma=0.5)
    return out


def preprocess_gray(gray: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """풀 파이프라인: FFT 중앙 crop → 후처리 체인 → 리사이즈."""
    denoised = fft_central_crop(gray, crop_r=CROP_R, pad=PAD)
    refined = post_preprocess(denoised)
    return cv2.resize(refined, target_size)


def preprocess_file(img_path: str, target_size: tuple = TARGET_SIZE) -> np.ndarray | None:
    """경로에서 이미지 읽고 preprocess_gray 적용. 실패 시 None."""
    img = imread_unicode(img_path)
    if img is None:
        return None
    gray = to_gray(img)
    return preprocess_gray(gray, target_size=target_size)
