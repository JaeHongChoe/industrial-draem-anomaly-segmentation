"""
Extract defect patches (Crack, Bow) from test_orig using LabelMe JSON annotations.
Applies FFT preprocessing before extraction to match training data.
Saves patches as PNG with alpha mask for use in DRAEM training.
"""
import os
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm

# Configuration
INPUT_DIR = '../test_orig'
OUTPUT_DIR = './defect_patches3'
LABELS_TO_EXTRACT = ['Crack', 'Bow', 'crack', 'bow', 'CRACK']  # Case variations
PADDING = 25  # Extra padding around bounding box (increased for more margin)

# ---------------------------
# NEW 전처리 파이프라인 (tile 버전)
# ---------------------------
TARGET_SIZE = (640, 640)
PAD = 8
CROP_R = 170

def to_gray(img):
    """BGR/BGRA 이미지를 그레이스케일로 변환"""
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# FFT pad / unpad
def fft2_pad(gray, pad=64):
    g = gray.astype(np.float32)
    big = cv2.copyMakeBorder(g, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    F = np.fft.fftshift(np.fft.fft2(big))
    return F, big.shape, pad

def ifft2_unpad(F2, shape, pad):
    H, W = shape
    out_big = np.fft.ifft2(np.fft.ifftshift(F2))
    out_big = np.real(out_big)
    return out_big[pad:H-pad, pad:W-pad]

def tone_match(out, ref):
    m0, s0 = cv2.meanStdDev(ref); m0, s0 = float(m0), float(s0)+1e-6
    m1, s1 = cv2.meanStdDev(out); m1, s1 = float(m1), float(s1)+1e-6
    out = (out - m1) * (s0 / s1) + m0
    return np.clip(out, 0, 255).astype(np.uint8)

def fft_central_crop_and_visualize(gray, crop_r=CROP_R, pad=PAD):
    F, shape_big, pad_sz = fft2_pad(gray, pad)
    H, W = F.shape
    cy, cx = H//2, W//2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y-cy)**2 + (X-cx)**2)
    mask = (R <= crop_r).astype(np.float32)
    F_crop = F * mask
    out = ifft2_unpad(F_crop, shape_big, pad_sz)
    out = tone_match(out, gray.astype(np.float32))
    return out

def gaussian_pre(img, sigma=1.0):
    return cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)

def median_pre(img, k=3):
    return cv2.medianBlur(img, k)

def wiener_local(img, ksize=5, noise_var=100.0):
    f = img.astype(np.float32)
    ones = np.ones((ksize, ksize), np.float32) / (ksize*ksize)
    mean = cv2.filter2D(f, -1, ones, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(f*f, -1, ones, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean*mean, 0.0)
    gain = np.maximum(var - noise_var, 0.0) / (var + 1e-6)
    out = mean + gain * (f - mean)
    return np.clip(out, 0, 255).astype(np.uint8)

def contrast_stretch(img, low_perc=10, high_perc=97):
    low, high = np.percentile(img, (low_perc, high_perc))
    out = np.clip((img - low) * 255.0 / (high - low + 1e-6), 0, 255)
    return out.astype(np.uint8)

def adjust_gamma(img, gamma=1.4):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def sharpen_unsharp(img, amount=2.2, sigma=0.5):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    sharp = cv2.addWeighted(img, 1+amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def post_preprocess(img):
    out = img.copy()
    out = gaussian_pre(out, sigma=1.0)
    out = median_pre(out, k=3)
    out = wiener_local(out, ksize=5, noise_var=100.0)
    out = contrast_stretch(out, low_perc=10, high_perc=97)
    out = adjust_gamma(out, gamma=1.4)
    out = sharpen_unsharp(out, amount=2.2, sigma=0.5)
    return out

def new_preprocess(gray):
    """새로운 전처리 파이프라인"""
    denoised = fft_central_crop_and_visualize(gray, crop_r=CROP_R, pad=PAD)
    refined = post_preprocess(denoised)
    out = cv2.resize(refined, TARGET_SIZE)
    return out

def ensure_dirs():
    """Create output directories."""
    os.makedirs(os.path.join(OUTPUT_DIR, 'crack'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'bow'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'other'), exist_ok=True)

def parse_json(json_path):
    """Parse LabelMe JSON file and return shapes."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('shapes', []), data.get('imagePath', ''), data.get('imageHeight', 0), data.get('imageWidth', 0)

def scale_points(points, orig_h, orig_w, target_h, target_w):
    """Scale polygon points from original size to target size."""
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    scaled = [[p[0] * scale_x, p[1] * scale_y] for p in points]
    return scaled

def polygon_to_mask(points, height, width):
    """Convert polygon points to binary mask."""
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def extract_patch(image, mask, points, padding=PADDING):
    """Extract patch with alpha channel from image using polygon mask."""
    pts = np.array(points, dtype=np.int32)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(pts)
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    # Skip if too small
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    
    # Crop image and mask
    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy()
    
    # Create RGBA image (with alpha channel)
    if len(cropped_img.shape) == 2:
        # Grayscale
        rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = cropped_img
        rgba[:, :, 1] = cropped_img
        rgba[:, :, 2] = cropped_img
        rgba[:, :, 3] = cropped_mask
    else:
        # Color
        rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = cropped_img
        rgba[:, :, 3] = cropped_mask
    
    return rgba

def get_label_category(label):
    """Normalize label to category."""
    label_lower = label.lower()
    if 'crack' in label_lower:
        return 'crack'
    elif 'bow' in label_lower:
        return 'bow'
    else:
        return 'other'

def main():
    ensure_dirs()
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    print(f"Found {len(json_files)} JSON files")
    print(f"Applying FFT preprocessing (same as training data)")
    
    patch_counts = {'crack': 0, 'bow': 0, 'other': 0}
    
    for json_path in tqdm(json_files, desc="Extracting patches"):
        try:
            shapes, image_path, orig_height, orig_width = parse_json(json_path)
            
            if not shapes or orig_height == 0 or orig_width == 0:
                continue
            
            # Load corresponding image
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            
            # Try different image extensions
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = os.path.join(INPUT_DIR, base_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            # Load image (handle Unicode paths)
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            # Convert to grayscale
            gray = to_gray(image)
            
            # 새 전처리 적용
            preprocessed = new_preprocess(gray)
            
            # Target size after preprocessing
            target_h, target_w = TARGET_SIZE[1], TARGET_SIZE[0]
            
            # Process each shape
            for i, shape in enumerate(shapes):
                label = shape.get('label', '')
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', '')
                
                if shape_type != 'polygon' or len(points) < 3:
                    continue
                
                category = get_label_category(label)
                
                # Scale polygon points to match preprocessed image size
                scaled_points = scale_points(points, orig_height, orig_width, target_h, target_w)
                
                # Create mask on preprocessed image
                mask = polygon_to_mask(scaled_points, target_h, target_w)
                patch = extract_patch(preprocessed, mask, scaled_points)
                
                if patch is None:
                    continue
                
                # Save patch
                out_name = f"{base_name}_patch{i}.png"
                out_path = os.path.join(OUTPUT_DIR, category, out_name)
                
                # Encode and save (handle Unicode paths)
                success, encoded = cv2.imencode('.png', patch)
                if success:
                    encoded.tofile(out_path)
                    patch_counts[category] += 1
                    
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    print(f"\nExtraction complete!")
    print(f"  Crack patches: {patch_counts['crack']}")
    print(f"  Bow patches: {patch_counts['bow']}")
    print(f"  Other patches: {patch_counts['other']}")
    print(f"\nPatches saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
