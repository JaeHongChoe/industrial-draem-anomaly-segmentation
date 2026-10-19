"""
FFT + Advanced Post-processing Pipeline for Defect Detection

New preprocessing chain:
1. FFT central circular crop restoration
2. Gaussian + Median denoising
3. Wiener local filter
4. Contrast stretch (10~97%)
5. Gamma correction (1.4)
6. Unsharp masking (sharpen)
7. Resize to target_size (1024)
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# FFT Parameters
FFT_PAD = 8
FFT_CROP_R = 170  # Central circular crop radius

# Post-processing Parameters
GAUSSIAN_SIGMA = 1.0
MEDIAN_K = 3
WIENER_KSIZE = 5
WIENER_NOISE_VAR = 100.0
CONTRAST_LOW_PERC = 10
CONTRAST_HIGH_PERC = 97
GAMMA = 1.4
SHARPEN_AMOUNT = 2.2
SHARPEN_SIGMA = 0.5


def to_gray(img):
    """Convert to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def fft2_pad(gray, pad=FFT_PAD):
    """FFT with padding."""
    g = gray.astype(np.float32)
    big = cv2.copyMakeBorder(g, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    F = np.fft.fftshift(np.fft.fft2(big))
    return F, big.shape, pad


def ifft2_unpad(F2, shape, pad):
    """Inverse FFT and unpad."""
    H, W = shape
    out_big = np.fft.ifft2(np.fft.ifftshift(F2))
    out_big = np.real(out_big)
    return out_big[pad:H-pad, pad:W-pad]


def tone_match(out, ref):
    """Match mean/std of output to reference."""
    m0, s0 = cv2.meanStdDev(ref); m0, s0 = float(m0[0][0]), float(s0[0][0]) + 1e-6
    m1, s1 = cv2.meanStdDev(out); m1, s1 = float(m1[0][0]), float(s1[0][0]) + 1e-6
    out = (out - m1) * (s0 / s1) + m0
    return np.clip(out, 0, 255).astype(np.uint8)


def fft_central_crop(gray, crop_r=FFT_CROP_R, pad=FFT_PAD):
    """
    FFT central circular crop and restoration.
    Keeps only low-frequency central area.
    """
    F, shape_big, pad_sz = fft2_pad(gray, pad)
    H, W = F.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = (R <= crop_r).astype(np.float32)
    F_crop = F * mask
    out = ifft2_unpad(F_crop, shape_big, pad_sz)
    out = tone_match(out, gray.astype(np.float32))
    return out


def gaussian_pre(img, sigma=GAUSSIAN_SIGMA):
    """Gaussian blur."""
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def median_pre(img, k=MEDIAN_K):
    """Median filter."""
    return cv2.medianBlur(img, k)


def wiener_local(img, ksize=WIENER_KSIZE, noise_var=WIENER_NOISE_VAR):
    """Local Wiener filter for noise reduction."""
    f = img.astype(np.float32)
    ones = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    mean = cv2.filter2D(f, -1, ones, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(f * f, -1, ones, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    gain = np.maximum(var - noise_var, 0.0) / (var + 1e-6)
    out = mean + gain * (f - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretch(img, low_perc=CONTRAST_LOW_PERC, high_perc=CONTRAST_HIGH_PERC):
    """Percentile-based contrast stretching."""
    low, high = np.percentile(img, (low_perc, high_perc))
    out = np.clip((img - low) * 255.0 / (high - low + 1e-6), 0, 255)
    return out.astype(np.uint8)


def adjust_gamma(img, gamma=GAMMA):
    """Gamma correction."""
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def sharpen_unsharp(img, amount=SHARPEN_AMOUNT, sigma=SHARPEN_SIGMA):
    """Unsharp masking for edge enhancement."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def post_preprocess(img):
    """
    Post-processing chain:
    1. Gaussian(σ=1.0) → Median(3)
    2. Wiener local filter
    3. Contrast stretch (10~97%)
    4. Gamma (1.4)
    5. Unsharp mask sharpen
    """
    out = img.copy()
    out = gaussian_pre(out, sigma=GAUSSIAN_SIGMA)
    out = median_pre(out, k=MEDIAN_K)
    out = wiener_local(out, ksize=WIENER_KSIZE, noise_var=WIENER_NOISE_VAR)
    out = contrast_stretch(out, low_perc=CONTRAST_LOW_PERC, high_perc=CONTRAST_HIGH_PERC)
    out = adjust_gamma(out, gamma=GAMMA)
    out = sharpen_unsharp(out, amount=SHARPEN_AMOUNT, sigma=SHARPEN_SIGMA)
    return out


def new_preprocess(gray):
    """
    Full preprocessing pipeline:
    1. FFT central circular crop restoration
    2. Post-processing chain
    """
    denoised = fft_central_crop(gray, crop_r=FFT_CROP_R, pad=FFT_PAD)
    refined = post_preprocess(denoised)
    return refined


def apply_fft_filter(image):
    """
    Main preprocessing function (for compatibility).
    Returns float32 normalized to 0-1 range.
    """
    gray = to_gray(image)
    processed = new_preprocess(gray)
    return processed.astype(np.float32) / 255.0


def process_dataset(src_dirs, dst_suffix='_fft_1024', target_size=1024):
    """Process entire dataset with new preprocessing."""
    for src_dir in src_dirs:
        dirname = os.path.dirname(src_dir)
        basename = os.path.basename(src_dir)
        dst_dir = os.path.join(dirname, basename + dst_suffix)
        
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Processing: {src_dir} -> {dst_dir}")
        print(f"  FFT crop_r={FFT_CROP_R}, pad={FFT_PAD}")
        print(f"  Gamma={GAMMA}, Sharpen={SHARPEN_AMOUNT}")
        
        images = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) + 
                       glob.glob(os.path.join(src_dir, "*.png")))
        
        for img_path in tqdm(images):
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Apply new preprocessing
            img_fft = apply_fft_filter(img)
            
            # Resize to target
            img_resized = cv2.resize(img_fft, (target_size, target_size))
            
            # Convert to uint8
            img_save = (img_resized * 255).astype(np.uint8)
            
            # Save
            filename = os.path.basename(img_path)
            dst_path = os.path.join(dst_dir, filename)
            cv2.imwrite(dst_path, img_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dirs', nargs='+', default=[
        'data/train_g2_orig_ok',
        'data/train_g3_orig_ok'
    ])
    parser.add_argument('--suffix', default='_fft_v2_1024')
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()
    
    process_dataset(args.src_dirs, dst_suffix=args.suffix, target_size=args.size)
