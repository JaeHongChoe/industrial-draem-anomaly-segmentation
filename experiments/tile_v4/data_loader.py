"""
Tile-Based Data Loader for Defect Segmentation

Loads original images → Resize to 1024×1024 → Extract 512×512 tiles with overlap
Applies synthetic patch augmentation per tile for training.
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# ... imports ...

def load_single_image(path):
    """Helper for parallel loading"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


class TileTrainDataset(Dataset):
    """
    Tile-based training dataset with synthetic patch augmentation.
    
    - Loads original images from specified folders
    - Resizes to full_size (1024×1024)
    - Extracts tiles of tile_size (512×512) with overlap
    - Applies synthetic patch augmentation per tile
    """
    
    def __init__(self, 
                 image_dirs,
                 anomaly_source_path,
                 full_size=1024,
                 tile_size=512,
                 overlap=64,
                 augment=True,
                 apply_fft=True):  # NEW: Apply FFT preprocessing
        """
        Args:
            image_dirs: List of directories containing training images
            anomaly_source_path: Path to defect patches for augmentation
            full_size: Size to resize original image (1024)
            tile_size: Size of each tile (512)
            overlap: Overlap between tiles in pixels (64)
            augment: Whether to apply synthetic patch augmentation
            apply_fft: Whether to apply FFT high-pass filter preprocessing
        """
        self.full_size = full_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.apply_fft = apply_fft
        self.stride = tile_size - overlap  # 448
        self.current_epoch = 0  # For epoch-based intensity scaling
        self.max_epoch = 100    # Expected max epochs for scaling
        
        # Calculate tile positions: 4 corners + 1 center = 5 tiles
        # For 1024 full_size, 512 tile_size:
        # Corners: (0,0), (0,512), (512,0), (512,512)
        # Center: (256, 256)
        self.tile_positions = [
            (0, 0),                                           # Top-left
            (0, full_size - tile_size),                       # Top-right
            (full_size - tile_size, 0),                       # Bottom-left
            (full_size - tile_size, full_size - tile_size),   # Bottom-right
            ((full_size - tile_size) // 2, (full_size - tile_size) // 2)  # Center
        ]
        
        # User Request: Use 10 random patches for training (augment=True)
        # For validation (augment=False), keep 5 fixed positions
        if self.augment:
            self.tiles_per_image = 10
        else:
            self.tiles_per_image = len(self.tile_positions) # 5
        
        # Load all image paths
        self.image_paths = []
        for img_dir in image_dirs:
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.jpg")))
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.png")))
        self.image_paths = sorted(self.image_paths)
        
        print(f"[TileTrainDataset] Found {len(self.image_paths)} images")
        print(f"  Full size: {full_size}×{full_size}")
        print(f"  Tile size: {tile_size}×{tile_size}")
        print(f"  Tiles per image: {self.tiles_per_image} (4 corners + 1 center)")
        print(f"  Total tiles: {len(self.image_paths) * self.tiles_per_image}")
        print(f"  Apply FFT: {apply_fft}")
        
        # Load anomaly source patches
        self.anomaly_source_paths = []
        if anomaly_source_path and os.path.exists(anomaly_source_path):
            self.anomaly_source_paths = sorted(
                glob.glob(os.path.join(anomaly_source_path, "**", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(anomaly_source_path, "**", "*.png"), recursive=True)
            )
            print(f"  Anomaly patches found: {len(self.anomaly_source_paths)}")
            
            # Pre-load patches into RAM
            print("  Pre-loading anomaly patches into RAM...")
            self.anomaly_patches = []
            for p_path in tqdm(self.anomaly_source_paths, desc="Loading Patches"):
                patch = cv2.imdecode(np.fromfile(p_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if patch is not None:
                    self.anomaly_patches.append(patch)
            print(f"  Successfully loaded {len(self.anomaly_patches)} patches.")



    def set_epoch(self, epoch):
        """Set current epoch for epoch-based intensity scaling."""
        self.current_epoch = epoch
    
    def get_intensity_mode(self):
        """
        Get defect intensity mode based on current epoch.
        
        Fixed probabilities (User Request):
        - Strong: 15%
        - Medium: 15%
        - Subtle: 70%
        
        Returns: 'strong', 'medium', or 'subtle'
        """
        # Fixed probabilities
        subtle_prob = 0.70
        medium_prob = 0.15
        strong_prob = 0.15
        
        rand = np.random.rand()
        if rand < strong_prob:
            return 'strong'
        elif rand < strong_prob + medium_prob:
            return 'medium'
        else:
            return 'subtle'
        
    def __len__(self):
        # Each image produces tiles_per_image tiles
        return len(self.image_paths) * self.tiles_per_image
    
    def cv2_augment(self, image):
        """Apply random augmentation similar to a4.py."""
        # Ensure image is in 0-255 range and uint8
        if image.dtype == np.float32:
            image_aug = (image * 255).astype(np.uint8)
        else:
            image_aug = image.copy()
            
        h, w = image_aug.shape[:2]
        
        # 1. Flip (Vertical / Horizontal)
        if np.random.rand() < 0.5:
            image_aug = cv2.flip(image_aug, 0)  # Top-bottom
        if np.random.rand() < 0.5:
            image_aug = cv2.flip(image_aug, 1)  # Left-right
            
        # 2. Rotation (-180 ~ 180)
        angle = np.random.randint(-180, 181)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image_aug = cv2.warpAffine(image_aug, M, (w, h))
        
        # 3. Brightness (factor 0.7 ~ 1.3)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image_aug = np.clip(image_aug.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
        # 4. Contrast (factor 0.7 ~ 1.5)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.5)
            mean = np.mean(image_aug)
            image_aug = np.clip((image_aug.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
            
        # Return as same type/range as input
        if image.dtype == np.float32:
            return image_aug.astype(np.float32) / 255.0
        return image_aug

    def cv2_rotate(self, image, angle=None):
        """Rotate image by random angle."""
        if angle is None:
            angle = np.random.randint(-180, 181)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def apply_fft_filter(self, image):
        """Apply FFT high-pass filter for edge enhancement."""
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # High-pass filter - remove low frequencies
        mask = np.ones((rows, cols), np.float32)
        r = 30  # Radius of low frequencies to remove
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize to 0-1 range
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
        return img_back.astype(np.float32)
    
    def extract_tile(self, image, tile_idx):
        """Extract a specific tile from the full image using predefined positions."""
        y_start, x_start = self.tile_positions[tile_idx]
        tile = image[y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
        return tile, (y_start, x_start)
    
    def extract_random_tile(self, image):
        """Extract a random tile from the full image (for training)."""
        h, w = image.shape[:2]
        
        # Ensure image is large enough
        if h < self.tile_size or w < self.tile_size:
            # Fallback: Resize to tile size if smaller (should not happen with 1024x1024)
            return cv2.resize(image, (self.tile_size, self.tile_size)), (0, 0)
            
        y_max = h - self.tile_size
        x_max = w - self.tile_size
        
        # Random position
        y_start = np.random.randint(0, y_max + 1)
        x_start = np.random.randint(0, x_max + 1)
        
        tile = image[y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
        return tile, (y_start, x_start)
    
    # ========== NEW: DRAEM-main Style Patch Methods ==========
    
    def load_random_patch(self):
        """Load a random defect patch from RAM."""
        if not hasattr(self, 'anomaly_patches') or not self.anomaly_patches:
            return None, None
        
        # Sample from RAM
        patch_idx = np.random.randint(0, len(self.anomaly_patches))
        patch = self.anomaly_patches[patch_idx]
        
        # Check if it has alpha channel
        if len(patch.shape) == 3 and patch.shape[2] >= 4:
            # Has alpha channel - use it as mask
            patch_img = patch[:, :, :3].mean(axis=2) if patch.shape[2] >= 3 else patch[:, :, 0]
            patch_mask = patch[:, :, 3]
        elif len(patch.shape) == 3 and patch.shape[2] == 3:
            # RGB without alpha - create mask from non-white pixels
            patch_img = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            patch_mask = (patch_img < 250).astype(np.uint8) * 255
        else:
            # Grayscale - create mask from non-white pixels
            patch_img = patch
            patch_mask = (patch_img < 250).astype(np.uint8) * 255
        
        return patch_img.astype(np.float32), patch_mask.astype(np.float32)
    
    def augment_patch(self, patch_img, patch_mask, tiny_mode=False):
        """Apply strong random transformations to patch (DRAEM-main style).
        
        Includes: scale, rotation, flip, elastic, crop, perspective, shear, brightness
        tiny_mode: If True, use small scale (0.3~0.7) for realistic tiny defects
        
        Returns None, None if augmentation results in too small patch (< 25px)
        """
        if patch_img is None:
            return None, None
            
        # Retry loop to ensure valid mask size
        for _ in range(3):
            h, w = patch_img.shape[:2]
            
            # Scale based on mode
            if tiny_mode:
                # Tiny mode: scale 0.30 ~ 0.70 (User Request: min 25px visibility)
                scale = np.random.uniform(0.30, 0.70)
            else:
                # Normal mode: scale 0.3 ~ 3.0
                scale = np.random.uniform(0.3, 3.0)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Minimum size check
            if new_w < 3 or new_h < 3:
                new_w, new_h = max(3, new_w), max(3, new_h)
            
            aug_img = cv2.resize(patch_img, (new_w, new_h))
            aug_mask = cv2.resize(patch_mask, (new_w, new_h))
            
            # Random rotation
            angle = np.random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (new_w, new_h))
            aug_mask = cv2.warpAffine(aug_mask, M, (new_w, new_h))
            
            # Random flip (50% chance)
            if np.random.choice([True, False]):
                aug_img = cv2.flip(aug_img, 1)
                aug_mask = cv2.flip(aug_mask, 1)
                
            if np.random.choice([True, False]):
                aug_img = cv2.flip(aug_img, 0)
                aug_mask = cv2.flip(aug_mask, 0)
            
            # Random Elastic deformation (30% chance)
            if np.random.rand() < 0.3:
                aug_img, aug_mask = self._elastic_transform(aug_img, aug_mask)
            
            # Random Crop (40% chance)
            if np.random.rand() < 0.4:
                aug_img, aug_mask = self._random_crop_patch(aug_img, aug_mask)
            
            # Random Perspective (30% chance)
            if np.random.rand() < 0.3:
                try:
                    aug_img, aug_mask = self._perspective_transform(aug_img, aug_mask)
                except:
                    pass
            
            # Random Shear (30% chance)
            if np.random.rand() < 0.3:
                try:
                    aug_img, aug_mask = self._shear_transform(aug_img, aug_mask)
                except:
                    pass
            
            # Random Aspect Ratio Stretch (50% chance)
            if np.random.rand() < 0.5:
                stretch = np.random.uniform(0.5, 1.5)
                if np.random.choice([True, False]): # Width
                    aug_img = cv2.resize(aug_img, (int(aug_img.shape[1] * stretch), aug_img.shape[0]))
                    aug_mask = cv2.resize(aug_mask, (int(aug_mask.shape[1] * stretch), aug_mask.shape[0]))
                else: # Height
                    aug_img = cv2.resize(aug_img, (aug_img.shape[1], int(aug_img.shape[0] * stretch)))
                    aug_mask = cv2.resize(aug_mask, (aug_mask.shape[1], int(aug_mask.shape[0] * stretch)))
            
            # Check minimum pixel count (User Request: > 25px)
            pixel_count = np.count_nonzero(aug_mask > 0)
            if pixel_count >= 25:
                # Brightness/Contrast adjust
                alpha = np.random.uniform(0.5, 1.5)
                beta = np.random.uniform(0.7, 1.3)
                aug_img = cv2.multiply(aug_img, np.array([alpha]))
                aug_img = cv2.add(aug_img, np.array([beta]))
                aug_img = np.clip(aug_img, 0, 255)
                
                return aug_img, aug_mask
        
        # If failed 3 times
        return None, None
    
    def _elastic_transform(self, img, mask, alpha=30, sigma=5):
        """Apply elastic deformation to patch."""
        shape = img.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        return img, mask
    
    def _random_crop_patch(self, img, mask):
        """Randomly crop edges of patch to create more diverse shapes."""
        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return img, mask
        
        # Random crop from each side (0-30%)
        crop_top = int(h * np.random.uniform(0, 0.3))
        crop_bottom = int(h * np.random.uniform(0, 0.3))
        crop_left = int(w * np.random.uniform(0, 0.3))
        crop_right = int(w * np.random.uniform(0, 0.3))
        
        new_h = h - crop_top - crop_bottom
        new_w = w - crop_left - crop_right
        
        if new_h < 5 or new_w < 5:
            return img, mask
        
        img = img[crop_top:h-crop_bottom, crop_left:w-crop_right]
        mask = mask[crop_top:h-crop_bottom, crop_left:w-crop_right]
        return img, mask
    
    def _perspective_transform(self, img, mask):
        """Apply random perspective warp."""
        h, w = img.shape[:2]
        offset = min(h, w) * 0.2
        
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_pts = np.float32([
            [np.random.uniform(0, offset), np.random.uniform(0, offset)],
            [w - np.random.uniform(0, offset), np.random.uniform(0, offset)],
            [np.random.uniform(0, offset), h - np.random.uniform(0, offset)],
            [w - np.random.uniform(0, offset), h - np.random.uniform(0, offset)]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h))
        mask = cv2.warpPerspective(mask, M, (w, h))
        return img, mask
    
    def _shear_transform(self, img, mask):
        """Apply random shear transformation."""
        h, w = img.shape[:2]
        shear_x = np.random.uniform(-0.3, 0.3)
        shear_y = np.random.uniform(-0.3, 0.3)
        
        M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        new_w = int(w + abs(shear_x) * h)
        new_h = int(h + abs(shear_y) * w)
        
        img = cv2.warpAffine(img, M, (new_w, new_h))
        mask = cv2.warpAffine(mask, M, (new_w, new_h))
        return img, mask
    
    def place_patch_on_tile(self, tile, patch_img, patch_mask, placed_regions=None, tile_start=None):
        """Place augmented patch on tile with circular 98% constraint (full image coords) and overlap checking.
        
        Args:
            tile_start: (y_start, x_start) position of this tile in full image
        """
        if patch_img is None or patch_mask is None:
            return tile, np.zeros((self.tile_size, self.tile_size), dtype=np.float32), None
        
        tile_h, tile_w = tile.shape[:2]
        patch_h, patch_w = patch_img.shape[:2]
        
        # Ensure patch fits in tile
        if patch_h > tile_h or patch_w > tile_w:
            scale = min(tile_h / patch_h, tile_w / patch_w) * 0.9
            patch_h, patch_w = int(patch_h * scale), int(patch_w * scale)
            patch_img = cv2.resize(patch_img, (patch_w, patch_h))
            patch_mask = cv2.resize(patch_mask, (patch_w, patch_h))
        
        # === 98% Circular Area Constraint (FULL IMAGE coordinates) ===
        # Full image center and radius
        full_center_x, full_center_y = self.full_size // 2, self.full_size // 2
        max_radius = int(self.full_size / 2 * 0.98)
        
        # Get tile offset in full image
        if tile_start is not None:
            tile_y_off, tile_x_off = tile_start
        else:
            tile_y_off, tile_x_off = 0, 0
        
        # Try up to 20 times to find valid position
        margin = 5
        found = False
        for _ in range(20):
            x1 = np.random.randint(margin, max(margin+1, tile_w - patch_w - margin))
            y1 = np.random.randint(margin, max(margin+1, tile_h - patch_h - margin))
            
            # Convert patch center to FULL IMAGE coordinates
            patch_cx_full = tile_x_off + x1 + patch_w // 2
            patch_cy_full = tile_y_off + y1 + patch_h // 2
            dist = np.sqrt((patch_cx_full - full_center_x)**2 + (patch_cy_full - full_center_y)**2)
            
            if dist > max_radius:
                continue
            
            x2, y2 = x1 + patch_w, y1 + patch_h
            
            # Check overlap with existing regions
            if placed_regions is not None:
                overlap = False
                for (py1, px1, py2, px2) in placed_regions:
                    ix1 = max(x1, px1)
                    iy1 = max(y1, py1)
                    ix2 = min(x2, px2)
                    iy2 = min(y2, py2)
                    if ix1 < ix2 and iy1 < iy2:
                        overlap = True
                        break
                if overlap:
                    continue
            
            found = True
            break
        
        if not found:
            return tile, np.zeros((tile_h, tile_w), dtype=np.float32), None
        
        x2, y2 = x1 + patch_w, y1 + patch_h
        
        # Normalize mask to 0-1
        norm_mask = patch_mask / 255.0 if patch_mask.max() > 1 else patch_mask
        
        # Create output
        result = tile.copy()
        full_mask = np.zeros((tile_h, tile_w), dtype=np.float32)
        
        # === Unified Blending Logic (same as Linear Crack) ===
        
        # 2. Get Intensity Mode (epoch-based)
        intensity_mode = self.get_intensity_mode()
        
        # 3. Opacity
        opacity = np.random.uniform(0.3, 1.0)
        
        # 4. Get ROI with safe clipping
        y1_safe = max(0, y1)
        y2_safe = min(tile_h, y2)
        x1_safe = max(0, x1)
        x2_safe = min(tile_w, x2)
        
        if y2_safe <= y1_safe or x2_safe <= x1_safe:
             return result, full_mask, None
             
        roi = result[y1_safe:y2_safe, x1_safe:x2_safe]
        
        # Calculate source offsets for patch
        py1 = y1_safe - y1
        py2 = py1 + (y2_safe - y1_safe)
        px1 = x1_safe - x1
        px2 = px1 + (x2_safe - x1_safe)
        
        # Crop blending source
        norm_mask_crop = norm_mask[py1:py2, px1:px2]
        
        # Soft/Hard Edge (50%/50%) - must be defined before intensity mode block
        is_soft_edge = np.random.rand() < 0.5
        
        # 5. Determine defect color based on Intensity Mode
        if intensity_mode == 'strong':
            # Strong: 70% darker, 30% black
            if np.random.rand() < 0.7:
                dark_ratio = 1.0 - np.random.uniform(0.5, 1.0)
                defect = roi * dark_ratio
            else:
                defect = np.zeros_like(roi)
        elif intensity_mode == 'medium':
            # Medium: 20-50% darker
            dark_ratio = np.random.uniform(0.5, 0.8)
            defect = roi * dark_ratio
        else:  # subtle
            # Subtle: absolute pixel difference
            if is_soft_edge:
                diff = np.random.uniform(0.12, 0.24)  # 30-60px
            else:
                diff = np.random.uniform(0.04, 0.24)  # 10-60px
            defect = np.maximum(roi - diff, 0)
            
        # 6. Apply blending
        # Apply blur to cropped mask if needed
        blend_mask = norm_mask_crop.copy()
        if is_soft_edge:
            blur_size = max(5, int(min(patch_h, patch_w) * 0.1))
            if blur_size % 2 == 0:
                blur_size += 1
            blend_mask = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
            
        final_alpha = blend_mask * opacity
        
        if len(roi.shape) == 2:
            blended = roi * (1 - final_alpha) + defect * final_alpha
        else:
            final_alpha_3d = final_alpha[:, :, np.newaxis]
            blended = roi * (1 - final_alpha_3d) + defect * final_alpha_3d
        
        result[y1_safe:y2_safe, x1_safe:x2_safe] = np.clip(blended, 0, 1)
        full_mask[y1_safe:y2_safe, x1_safe:x2_safe] = norm_mask_crop
        
        return result, full_mask, (y1_safe, x1_safe, y2_safe, x2_safe)
    
    def augment_tile(self, tile):
        """Apply synthetic patch augmentation to a tile (DRAEM-main style).
        
        - 95% probability of adding defect patches, 5% no anomaly
        - Mixed Mode: 65% tiny patches, 35% normal patches
        - Multiple patches per tile
        """
        if len(self.anomaly_source_paths) == 0:
            return tile, np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        # 5% chance of no anomaly
        if np.random.rand() > 0.95:
            return tile, np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        # Mixed Tiny Mode: 65% tiny, 35% normal
        tiny_mode = np.random.rand() < 0.65
        
        if tiny_mode:
            num_patches = np.random.randint(3, 16)  # 3-15 tiny patches
        else:
            num_patches = np.random.randint(5, 31)  # 5-30 patches for dense coverage
        
        result_tile = tile.copy()
        combined_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        for _ in range(num_patches):
            patch_img, patch_mask = self.load_random_patch()
            if patch_img is not None:
                patch_img, patch_mask = self.augment_patch(patch_img, patch_mask, tiny_mode=tiny_mode)
                if patch_img is not None and patch_mask is not None:
                    result_tile, single_mask = self.place_patch_on_tile(result_tile, patch_img, patch_mask)
                    combined_mask = np.maximum(combined_mask, single_mask)
        
        return result_tile, combined_mask
    
    def augment_tile_v4(self, tile, tile_start=None):
        """V4: Apply comprehensive augmentations directly to 512x512 tile.
        
        Includes all V3 augmentation logic adapted for tile-level processing:
        - Defect patches (5-30 per tile)
        - Linear crack with adaptive opacity (30% chance)
        - Shift augmentation with inpainting (5% chance)
        
        Probability distribution:
        - Normal (no defect): 5%
        - Patches only: 55%
        - Patches + linear crack: 30%
        - Linear crack only: 5%
        - Shift augmentation: 5% (applied on top of above)
        """
        tile_size = tile.shape[0]
        
        if len(self.anomaly_source_paths) == 0:
            return tile, np.zeros((tile_size, tile_size), dtype=np.float32)
        
        # Determine augmentation mode
        rand_val = np.random.rand()
        if rand_val < 0.05:
            # 5% - No anomaly
            return tile, np.zeros((tile_size, tile_size), dtype=np.float32)
        elif rand_val < 0.60:
            # 55% - Patches only
            apply_patches = True
            apply_crack = False
        elif rand_val < 0.90:
            # 30% - Patches + linear crack
            apply_patches = True
            apply_crack = True
        else:
            # 10% - Linear crack only
            apply_patches = False
            apply_crack = True
        
        result_tile = tile.copy()
        combined_mask = np.zeros((tile_size, tile_size), dtype=np.float32)
        
        # === Apply Patches ===
        if apply_patches:
            tiny_mode = np.random.rand() < 0.80
            
            if tiny_mode:
                num_patches = np.random.randint(5, 16)  # 5-15 tiny patches
            else:
                num_patches = np.random.randint(5, 31)  # 5-30 patches
            
            placed_regions = []
            for _ in range(num_patches):
                patch_img, patch_mask = self.load_random_patch()
                if patch_img is not None:
                    patch_img, patch_mask = self.augment_patch(patch_img, patch_mask, tiny_mode=tiny_mode)
                    if patch_img is not None and patch_mask is not None:
                        result_tile, single_mask, new_region = self.place_patch_on_tile(
                            result_tile, patch_img, patch_mask, placed_regions, tile_start
                        )
                        if new_region is not None:
                            placed_regions.append(new_region)
                            combined_mask = np.maximum(combined_mask, single_mask)
        
        # === Apply Linear Crack (with adaptive opacity) ===
        if apply_crack:
            crack_mask = self.generate_linear_crack(tile_size)
            
            # Mask out background (User Request)
            # Use original tile to determine valid foreground area (> 0.05)
            if len(tile.shape) == 3:
                valid_mask = (np.mean(tile, axis=2) > 0.05).astype(np.float32)
            else:
                valid_mask = (tile > 0.05).astype(np.float32)
            
            crack_mask = crack_mask * valid_mask
            
            result_tile, crack_applied_mask = self.apply_crack_to_image(result_tile, crack_mask)
            combined_mask = np.maximum(combined_mask, crack_applied_mask)
        
        # === Apply Shift Augmentation (5% chance, with inpainting) ===
        if np.random.rand() < 0.00:
            result_tile, shift_mask = self.shift_region_augmentation(result_tile)
            combined_mask = np.maximum(combined_mask, shift_mask)
        
        return result_tile, combined_mask
    
    def augment_full_image(self, image):
        """Apply synthetic patch and/or linear crack augmentation to full 1024x1024 image.
        
        Probability distribution:
        - Normal (no defect): 5%
        - Patches only: 45%
        - Patches + linear crack: 40%
        - Linear crack only: 10%
        """
        if len(self.anomaly_source_paths) == 0:
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        # Determine augmentation mode
        rand_val = np.random.rand()
        if rand_val < 0.05:
            # 5% - No anomaly
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32)
        elif rand_val < 0.50:
            # 45% - Patches only
            apply_patches = True
            apply_crack = False
        elif rand_val < 0.90:
            # 40% - Patches + linear crack
            apply_patches = True
            apply_crack = True
        else:
            # 10% - Linear crack only
            apply_patches = False
            apply_crack = True
        
        result_image = image.copy()
        combined_mask = np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        # Apply patches if needed
        if apply_patches:
            scale_factor = self.full_size / 640.0
            tiny_mode = np.random.rand() < 0.80
            
            if tiny_mode:
                num_patches = np.random.randint(5, 25)
            else:
                num_patches = np.random.randint(5, 31)
            
            placed_regions = []
            
            for _ in range(num_patches):
                patch_img, patch_mask = self.load_random_patch()
                if patch_img is not None:
                    patch_img, patch_mask = self.augment_patch_for_full(patch_img, patch_mask, tiny_mode, scale_factor)
                    if patch_img is not None and patch_mask is not None:
                        result_image, single_mask, new_region = self.place_patch_on_full_no_overlap(
                            result_image, patch_img, patch_mask, placed_regions
                        )
                        if new_region is not None:
                            placed_regions.append(new_region)
                            combined_mask = np.maximum(combined_mask, single_mask)
        
        # Apply linear crack if needed
        if apply_crack:
            crack_mask = self.generate_linear_crack(self.full_size)
            result_image, crack_applied_mask = self.apply_crack_to_image(result_image, crack_mask)
            combined_mask = np.maximum(combined_mask, crack_applied_mask)
        
        return result_image, combined_mask
    
    def augment_full_image_v3(self, image, tile_positions):
        """V3: Apply 5-30 patches PER TILE REGION on full image.
        
        This combines V1's dense coverage (guaranteed 5-30 patches per tile) with
        V2's spatial consistency (patches applied before tiling).
        
        Args:
            image: Full image (1024x1024)
            tile_positions: List of 5 (y, x) tile positions
            
        Returns:
            augmented_image, full_mask
        """
        if len(self.anomaly_source_paths) == 0:
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        # 5% chance of no anomaly
        if np.random.rand() < 0.05:
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        result_image = image.copy()
        combined_mask = np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        scale_factor = self.full_size / 640.0
        
        # Process each tile region
        for (region_y, region_x) in tile_positions:
            # Mixed Tiny Mode: 80% tiny, 20% normal
            tiny_mode = np.random.rand() < 0.80
            
            if tiny_mode:
                num_patches = np.random.randint(5, 16)  # 5-15 tiny patches per region
            else:
                num_patches = np.random.randint(5, 31)  # 5-30 patches per region
            
            placed_regions_local = []
            
            for _ in range(num_patches):
                patch_img, patch_mask = self.load_random_patch()
                if patch_img is not None:
                    patch_img, patch_mask = self.augment_patch_for_full(
                        patch_img, patch_mask, tiny_mode, scale_factor
                    )
                    if patch_img is not None and patch_mask is not None:
                        # Place patch within this tile region
                        result_image, single_mask, new_region = self.place_patch_in_region(
                            result_image, patch_img, patch_mask,
                            region_y, region_x, self.tile_size,
                            placed_regions_local
                        )
                        if new_region is not None:
                            placed_regions_local.append(new_region)
                            combined_mask = np.maximum(combined_mask, single_mask)
        
        # Optionally add linear crack (30% chance)
        if np.random.rand() < 0.30:
            crack_mask = self.generate_linear_crack(self.full_size)
            result_image, crack_applied_mask = self.apply_crack_to_image(result_image, crack_mask)
            combined_mask = np.maximum(combined_mask, crack_applied_mask)
        
        # Optionally add shift augmentation for shape defect (DISABLED for now)
        if np.random.rand() < 0.05:
            result_image, shift_mask = self.shift_region_augmentation(result_image)
            combined_mask = np.maximum(combined_mask, shift_mask)
        
        return result_image, combined_mask
    
    def shift_region_augmentation(self, image, bg_value=None):
        """
        Shift a random region of the image to simulate shape/position defects.
        
        New Logic (User Request):
        1. Use Inpainting to fill the hole naturally (instead of solid color)
        2. Apply subtle Shift (±5px) + Rotation (±3deg) for realistic defects
        
        Args:
            image: Input image (H, W) normalized to 0-1
        """
        h, w = image.shape[:2]
        
        # Prepare patches
        # Image is float 0-1, convert to uint8 for inpainting
        img_u8 = (image * 255).astype(np.uint8)
        
        # Random circular region selection
        region_size = np.random.randint(60, 150)
        margin = region_size + 30
        
        # Ensure we stay within bounds
        if w <= 2 * margin or h <= 2 * margin:
            return image, np.zeros((h, w), dtype=np.float32)
        
        # Select center point away from edges
        cx = np.random.randint(margin, w - margin)
        cy = np.random.randint(margin, h - margin)
        
        # === 1. Create Mask for the hole ===
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(hole_mask, (cx, cy), region_size, 255, -1)
        
        # === 2. Inpainting (Fill the hole naturally) ===
        # Using TELEA algo (radius 3)
        inpainted_u8 = cv2.inpaint(img_u8, hole_mask, 3, cv2.INPAINT_TELEA)
        inpainted_f = inpainted_u8.astype(np.float32) / 255.0
        
        # === 3. Prepare Shifted Content ===
        # Random slight shift & rotation
        shift_x = np.random.uniform(-5, 5)
        shift_y = np.random.uniform(-5, 5)
        angle = np.random.uniform(-3, 3)
        
        # Rotation matrix around the region center
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        
        # Apply transformation to original image
        shifted_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Apply transformation to soft mask
        # Create soft mask for smoother blending
        norm_mask = hole_mask.astype(np.float32) / 255.0
        soft_mask = cv2.GaussianBlur(norm_mask, (21, 21), 0)
        shifted_mask = cv2.warpAffine(soft_mask, M, (w, h))
        
        # === 4. Composite ===
        # Base is the Inpainted image (which looks like "empty background")
        # Overlay the Shifted content on top
        result = inpainted_f * (1 - shifted_mask) + shifted_image * shifted_mask
        
        # Defect mask: The region where the shifted content is visible
        # (Technically, the "difference" is the defect, but simpler to mark the shifted area)
        defect_mask = shifted_mask
        
        return np.clip(result, 0, 1).astype(np.float32), defect_mask
    
    def place_patch_in_region(self, image, patch_img, patch_mask, region_y, region_x, region_size, placed_regions):
        """Place patch within a specific region (tile area) of the full image.
        
        Args:
            image: Full image
            patch_img: Patch to place
            patch_mask: Mask for the patch
            region_y, region_x: Top-left corner of the region
            region_size: Size of the region (tile_size)
            placed_regions: List of already placed regions for overlap checking
            
        Returns:
            modified_image, mask, placed_region (or None if failed)
        """
        ph, pw = patch_img.shape[:2]
        
        # Calculate valid placement range within the region
        max_y = region_y + region_size - ph
        max_x = region_x + region_size - pw
        
        if max_y <= region_y or max_x <= region_x:
            # Patch too large for region, try to shrink it
            max_dim = min(region_size - 10, max(ph, pw))
            if max_dim < 10:
                return image, np.zeros_like(image), None
            scale = max_dim / max(ph, pw)
            new_h, new_w = int(ph * scale), int(pw * scale)
            if new_h < 5 or new_w < 5:
                return image, np.zeros_like(image), None
            patch_img = cv2.resize(patch_img, (new_w, new_h))
            patch_mask = cv2.resize(patch_mask, (new_w, new_h))
            ph, pw = new_h, new_w
            max_y = region_y + region_size - ph
            max_x = region_x + region_size - pw
        
        # Try to find non-overlapping position within 97% circular area
        max_attempts = 30
        CENTER_RATIO = 0.97
        center_x, center_y = self.full_size // 2, self.full_size // 2
        max_radius = int(self.full_size / 2 * CENTER_RATIO)
        
        found = False
        for _ in range(max_attempts):
            py = np.random.randint(region_y, max(region_y + 1, max_y))
            px = np.random.randint(region_x, max(region_x + 1, max_x))
            
            # Check if patch CENTER is within 97% circular area
            patch_center_x = px + pw // 2
            patch_center_y = py + ph // 2
            dist_from_center = np.sqrt((patch_center_x - center_x)**2 + (patch_center_y - center_y)**2)
            
            if dist_from_center > max_radius:
                continue  # Outside circular area
            
            new_region = (py, px, py + ph, px + pw)
            
            # Check overlap with existing patches
            overlap = False
            for (ey1, ex1, ey2, ex2) in placed_regions:
                if not (new_region[2] <= ey1 or new_region[0] >= ey2 or
                        new_region[3] <= ex1 or new_region[1] >= ex2):
                    overlap = True
                    break
            
            if not overlap:
                found = True
                break
        
        if not found:
            # If no valid position found, skip this patch
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32), None
        
        # Create mask for this patch
        single_mask = np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        # Ensure patch fits within image bounds
        y_end = min(py + ph, self.full_size)
        x_end = min(px + pw, self.full_size)
        patch_h = y_end - py
        patch_w = x_end - px
        
        if patch_h <= 0 or patch_w <= 0:
            return image, single_mask, None
        
        # Normalize patch mask to 0-1
        patch_mask_crop = patch_mask[:patch_h, :patch_w].copy()
        if patch_mask_crop.max() > 1:
            patch_mask_crop = patch_mask_crop / 255.0
        
        gt_mask = patch_mask_crop.copy()  # Keep original for GT (without blur)
        
        # Soft edge blending (blur for smooth blending only, not GT)
        blend_mask = patch_mask_crop.copy()
        blur_size = max(3, min(patch_h, patch_w) // 4)
        if blur_size % 2 == 0:
            blur_size += 1
        blend_mask = cv2.GaussianBlur(blend_mask.astype(np.float32), (blur_size, blur_size), 0)
        
        # Normalize blend mask
        if blend_mask.max() > 0:
            blend_mask = blend_mask / blend_mask.max()
        
        # Apply patch with alpha blending
        result = image.copy()
        
        # Beta blending strength (0.3~1.0)
        beta = np.random.uniform(0.3, 1.0)
        
        # 50% darker mode, 50% original texture mode
        use_darker_mode = np.random.rand() < 0.5
        
        if use_darker_mode:
            # Darker mode: darken background by 50-100%
            dark_ratio = 1.0 - np.random.uniform(0.5, 1.0)  # 0~0.5 brightness
            patch_region = result[py:y_end, px:x_end] * dark_ratio
        else:
            # Original mode: use patch texture
            patch_region = patch_img[:patch_h, :patch_w]
            # Normalize patch to 0-1 range if needed
            if patch_region.max() > 1.0:
                patch_region = patch_region.astype(np.float32) / 255.0
        
        if len(result.shape) == 2:
            roi = result[py:y_end, px:x_end]
            blended = roi * (1 - blend_mask * beta) + patch_region * blend_mask * beta
            result[py:y_end, px:x_end] = np.clip(blended, 0, 1)
        
        # Use original GT mask (without blur)
        single_mask[py:y_end, px:x_end] = gt_mask
        
        return result, single_mask, new_region
    
    def generate_linear_crack(self, size):
        """Generate random linear crack mask (straight, smooth_curve, or branching)."""
        crack_type = np.random.choice(['straight', 'smooth_curve', 'branching'])
        
        if crack_type == 'straight':
            return self._generate_straight_crack(size)
        elif crack_type == 'smooth_curve':
            return self._generate_smooth_curve_crack(size)
        else:
            return self._generate_branching_crack(size)
    
    def _generate_straight_crack(self, size):
        """Generate straight line crack mask."""
        mask = np.zeros((size, size), dtype=np.float32)
        thickness = np.random.randint(2, 10)
        length_ratio = np.random.uniform(0.3, 0.8)
        length = int(size * length_ratio)
        angle = np.random.uniform(0, 360)
        
        cx = np.random.randint(size//4, 3*size//4)
        cy = np.random.randint(size//4, 3*size//4)
        dx = int(length/2 * np.cos(np.radians(angle)))
        dy = int(length/2 * np.sin(np.radians(angle)))
        
        x1, y1 = int(np.clip(cx - dx, 10, size-10)), int(np.clip(cy - dy, 10, size-10))
        x2, y2 = int(np.clip(cx + dx, 10, size-10)), int(np.clip(cy + dy, 10, size-10))
        
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness)
        return mask
    
    def _generate_smooth_curve_crack(self, size):
        """Generate smooth curved crack mask using spline interpolation."""
        mask = np.zeros((size, size), dtype=np.float32)
        thickness = np.random.randint(2, 8)
        num_points = np.random.randint(4, 7)
        
        if np.random.rand() > 0.5:
            x_pts = np.linspace(100, size-100, num_points)
            y_center = np.random.randint(size//3, 2*size//3)
            y_pts = y_center + np.random.randint(-120, 120, num_points)
        else:
            y_pts = np.linspace(100, size-100, num_points)
            x_center = np.random.randint(size//3, 2*size//3)
            x_pts = x_center + np.random.randint(-120, 120, num_points)
        
        t = np.linspace(0, 1, num_points)
        t_new = np.linspace(0, 1, 150)
        
        try:
            cs_x = interpolate.CubicSpline(t, x_pts)
            cs_y = interpolate.CubicSpline(t, y_pts)
            x_smooth = np.clip(cs_x(t_new), 10, size-10).astype(np.int32)
            y_smooth = np.clip(cs_y(t_new), 10, size-10).astype(np.int32)
            points = np.column_stack([x_smooth, y_smooth])
            cv2.polylines(mask, [points], False, 1.0, thickness)
        except:
            # Fallback to straight line
            return self._generate_straight_crack(size)
        
        return mask

    
    def _generate_branching_crack(self, size):
        """Generate branching crack mask."""
        mask = np.zeros((size, size), dtype=np.float32)
        thickness = np.random.randint(2, 7)
        
        x1, y1 = np.random.randint(50, 150), np.random.randint(size//3, 2*size//3)
        x2, y2 = np.random.randint(size-150, size-50), np.random.randint(size//3, 2*size//3)
        
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, thickness)
        
        for _ in range(np.random.randint(2, 5)):
            t = np.random.uniform(0.2, 0.8)
            px, py = int(x1 + t*(x2-x1)), int(y1 + t*(y2-y1))
            bx = int(np.clip(px + np.random.randint(-200, 200), 50, size-50))
            by = int(np.clip(py + np.random.randint(-250, 250), 50, size-50))
            cv2.line(mask, (px, py), (bx, by), 1.0, max(1, thickness-1))
        
        return mask
    
    def apply_crack_to_image(self, image, mask):
        """Apply crack mask to image using epoch-based intensity modes.
        
        New Logic (User Request):
        1. Adaptive Opacity (0.3 ~ 1.0) to simulate Thread-like cracks
        2. Dark Area handling: force pure black defect if ROI is dark
        """
        result = image.copy()
        
        # Soft edge (50% chance)
        blend_mask = mask.copy()
        is_soft_edge = np.random.rand() < 0.5
        if is_soft_edge:
            blur_size = np.random.choice([3, 5, 7])
            blend_mask = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
        
        # Get intensity mode based on epoch
        intensity_mode = self.get_intensity_mode()
        
        # Select base opacity
        opacity = np.random.uniform(0.3, 1.0)
        
        # === Adaptive Defect Color Strategy ===
        # Calculate ROI brightness to handle dark areas
        pts = np.argwhere(mask > 0)
        if len(pts) > 0:
            y1, x1 = pts.min(axis=0)
            y2, x2 = pts.max(axis=0)
            roi_slice = image[y1:y2+1, x1:x2+1]
            if roi_slice.size > 0:
                roi_mean = np.mean(roi_slice) * 255.0
            else:
                roi_mean = 128
        else:
            roi_mean = 128
            
        if roi_mean < 80: # Dark area logic
            # Force defect to be pure black for maximum contrast
            defect = np.zeros_like(result)
            # Ensure minimum opacity in dark areas so it's visible
            final_opacity = max(opacity, 0.4)
        else:
            # Bright area logic
            beta = np.random.uniform(0.5, 1.0) # Original beta logic mostly merged into opacity
            
            if intensity_mode == 'strong':
                # Strong: 70% darker
                if np.random.rand() < 0.7:
                    dark_ratio = 1.0 - np.random.uniform(0.5, 1.0)
                    defect = result * dark_ratio
                else:
                    defect = np.zeros_like(result)
            elif intensity_mode == 'medium':
                 # Medium: darker by 20-50%
                dark_ratio = np.random.uniform(0.5, 0.8)
                defect = result * dark_ratio
            else:
                # Subtle: absolute pixel difference
                # Soft edge needs stronger diff (30-60px), Hard edge can be lighter (10-60px)
                if is_soft_edge:
                    diff = np.random.uniform(0.12, 0.24)  # 30-60px in 0-255 range
                else:
                    diff = np.random.uniform(0.04, 0.24)  # 10-60px in 0-255 range
                defect = np.maximum(result - diff, 0)
            
            final_opacity = opacity
            
        # Apply blend
        # result = result * (1 - alpha) + defect * alpha
        final_alpha = blend_mask * final_opacity
        
        result = result * (1 - final_alpha) + defect * final_alpha
        return np.clip(result, 0, 1), mask
    
    def augment_patch_for_full(self, patch_img, patch_mask, tiny_mode, scale_factor):
        """Augment patch for full 1024 image with scale adjustment."""
        if patch_img is None:
            return None, None
        
        h, w = patch_img.shape[:2]
        
        # Minimum patch size in pixels
        MIN_SIZE_TINY = 50   # Minimum 50x50 for tiny mode
        MIN_SIZE_NORMAL = 80  # Minimum 80x80 for normal mode
        
        # Scale based on mode, adjusted for 1024
        if tiny_mode:
            scale = np.random.uniform(0.15, 0.5) * scale_factor
            min_size = MIN_SIZE_TINY
        else:
            scale = np.random.uniform(0.3, 3.0) * scale_factor
            min_size = MIN_SIZE_NORMAL
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Enforce minimum size
        if new_w < min_size or new_h < min_size:
            # Scale up to meet minimum
            scale_up = max(min_size / max(new_w, 1), min_size / max(new_h, 1))
            new_w = int(new_w * scale_up)
            new_h = int(new_h * scale_up)
        
        patch_img = cv2.resize(patch_img, (new_w, new_h))
        patch_mask = cv2.resize(patch_mask, (new_w, new_h))
        
        # Random rotation
        angle = np.random.uniform(0, 360)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2
        patch_img = cv2.warpAffine(patch_img, M, (rot_w, rot_h))
        patch_mask = cv2.warpAffine(patch_mask, M, (rot_w, rot_h))
        
        # Random flip
        if np.random.rand() > 0.5:
            patch_img = np.fliplr(patch_img).copy()
            patch_mask = np.fliplr(patch_mask).copy()
        if np.random.rand() > 0.5:
            patch_img = np.flipud(patch_img).copy()
            patch_mask = np.flipud(patch_mask).copy()
        
        # Brightness/contrast
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.7, 1.3)
        patch_img = np.clip((patch_img - 128) * contrast + 128 * brightness, 0, 255)
        
        return patch_img, patch_mask
    
    def place_patch_on_full(self, image, patch_img, patch_mask):
        """Place patch on full 1024x1024 image (legacy, no overlap check)."""
        result, mask, _ = self.place_patch_on_full_no_overlap(image, patch_img, patch_mask, [])
        return result, mask
    
    def place_patch_on_full_no_overlap(self, image, patch_img, patch_mask, placed_regions):
        """Place patch on full 1024x1024 image, avoiding overlap with placed regions.
        
        Patches are placed within 97% radius CIRCULAR area. If patch overflows, it's clipped.
        """
        if patch_img is None or patch_mask is None:
            return image, np.zeros((self.full_size, self.full_size), dtype=np.float32), None
        
        img_h, img_w = image.shape[:2]
        patch_h, patch_w = patch_img.shape[:2]
        
        # 97% circular area placement
        CENTER_RATIO = 0.97
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = int(min(img_w, img_h) / 2 * CENTER_RATIO)  # 97% of half-width
        
        # Ensure patch fits in valid region
        max_dim = int(max_radius * 1.5)  # Max patch size relative to radius
        if patch_h > max_dim or patch_w > max_dim:
            scale = min(max_dim / patch_h, max_dim / patch_w) * 0.9
            patch_h, patch_w = int(patch_h * scale), int(patch_w * scale)
            patch_img = cv2.resize(patch_img, (patch_w, patch_h))
            patch_mask = cv2.resize(patch_mask, (patch_w, patch_h))
        
        # Try up to 30 times to find valid position in circular area
        margin = 10
        found = False
        for _ in range(30):
            # Random position
            x1 = np.random.randint(margin, img_w - patch_w - margin)
            y1 = np.random.randint(margin, img_h - patch_h - margin)
            
            # Check if patch CENTER is within circular 97% area
            patch_center_x = x1 + patch_w // 2
            patch_center_y = y1 + patch_h // 2
            dist_from_center = np.sqrt((patch_center_x - center_x)**2 + (patch_center_y - center_y)**2)
            
            if dist_from_center > max_radius:
                continue  # Outside circular area
            
            x2, y2 = x1 + patch_w, y1 + patch_h
            
            # Check overlap with existing regions
            overlap = False
            for (py1, px1, py2, px2) in placed_regions:
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                
                if ix1 < ix2 and iy1 < iy2:
                    overlap = True
                    break
            
            if not overlap:
                found = True
                break
        
        if not found:
            return image, np.zeros((img_h, img_w), dtype=np.float32), None
        
        x2, y2 = x1 + patch_w, y1 + patch_h
        
        # Clip if patch overflows (safety check)
        if x2 > img_w:
            patch_w = img_w - x1
            patch_img = patch_img[:, :patch_w]
            patch_mask = patch_mask[:, :patch_w]
            x2 = img_w
        if y2 > img_h:
            patch_h = img_h - y1
            patch_img = patch_img[:patch_h, :]
            patch_mask = patch_mask[:patch_h, :]
            y2 = img_h
        
        # Normalize mask
        norm_mask = patch_mask / 255.0 if patch_mask.max() > 1 else patch_mask
        gt_mask = norm_mask.copy()  # Keep original for ground truth
        
        # Soft edge blending (50% chance) - apply Gaussian blur ONLY for blending, not GT
        blend_mask = norm_mask.copy()
        if np.random.rand() < 0.5 and norm_mask.shape[0] > 5 and norm_mask.shape[1] > 5:
            blur_size = max(3, min(norm_mask.shape[0] // 4, 15))
            if blur_size % 2 == 0:
                blur_size += 1
            blend_mask = cv2.GaussianBlur(blend_mask.astype(np.float32), (blur_size, blur_size), 0)
        
        result = image.copy()
        full_mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Alpha blending using blurred mask
        beta = np.random.uniform(0.3, 1.0)
        roi = result[y1:y2, x1:x2]
        
        # 50% original texture, 50% darker mode
        use_darker_mode = np.random.rand() < 0.5
        
        if use_darker_mode:
            # Darker mode: darken background by 50-100%
            dark_ratio = 1.0 - np.random.uniform(0.5, 1.0)  # 0~0.5 brightness
            patch_region = roi * dark_ratio
        else:
            # Original mode: use patch texture
            patch_region = patch_img / 255.0 if patch_img.max() > 1 else patch_img
        
        if len(roi.shape) == 2:
            mask_region = blend_mask
            blended = roi * (1 - mask_region * beta) + patch_region * mask_region * beta
        else:
            mask_region = blend_mask[:, :, np.newaxis]
            patch_region = patch_region[:, :, np.newaxis] if len(patch_region.shape) == 2 else patch_region
            blended = roi * (1 - mask_region * beta) + patch_region * mask_region * beta
        
        result[y1:y2, x1:x2] = np.clip(blended, 0, 1)
        full_mask[y1:y2, x1:x2] = gt_mask  # Use original mask for GT
        
        return result, full_mask, (y1, x1, y2, x2)
    
    def __getitem__(self, idx):
        # Determine which image and which tile
        image_idx = idx // self.tiles_per_image
        tile_idx = idx % self.tiles_per_image
        
        # Load full image from RAM (Optimized)
        if hasattr(self, 'loaded_images') and self.loaded_images:
            image = self.loaded_images[image_idx]
        else:
            # Fallback for compatibility
            image_path = self.image_paths[image_idx]
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.full_size, self.full_size), dtype=np.uint8)
        
        # Apply FFT preprocessing FIRST (on original size)
        if self.apply_fft:
            image = self.apply_fft_filter(image)  # Returns float32 0-1
            image = cv2.resize(image, (self.full_size, self.full_size))
        else:
            image = cv2.resize(image, (self.full_size, self.full_size))
            image = image.astype(np.float32) / 255.0
        
        # Extract tile FIRST (before augmentation)
        if self.augment:
            # Random crop for training (Much better generalization)
            tile, (y_start, x_start) = self.extract_random_tile(image)
        else:
            # Fixed crop for validation
            tile, (y_start, x_start) = self.extract_tile(image, tile_idx)
        
        # Apply V4 augmentation at TILE level (512x512)
        if self.augment:
            augmented_tile, mask = self.augment_tile_v4(tile, tile_start=(y_start, x_start))
        else:
            augmented_tile = tile
            mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        
        # Add channel dimension
        tile = np.expand_dims(tile, axis=0)  # (1, H, W)
        augmented_tile = np.expand_dims(augmented_tile, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return {
            'image': tile.astype(np.float32),
            'augmented_image': augmented_tile.astype(np.float32),
            'anomaly_mask': mask.astype(np.float32),
            'has_anomaly': np.array([1.0 if mask.sum() > 0 else 0.0], dtype=np.float32),
            'idx': idx,
            'tile_pos': (y_start, x_start)
        }


class TileInferenceDataset(Dataset):
    """
    Dataset for tile-based inference with overlap.
    Returns all tiles for a single image for stitching.
    """
    
    def __init__(self, 
                 test_dir,
                 full_size=1024,
                 tile_size=512,
                 overlap=64,
                 apply_fft=True):
        self.test_dir = test_dir
        self.full_size = full_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.apply_fft = apply_fft
        
        self.tiles_per_dim = (full_size - tile_size) // self.stride + 1
        self.tiles_per_image = self.tiles_per_dim ** 2
        
        # Load image paths
        self.image_paths = sorted(
            glob.glob(os.path.join(test_dir, "*.jpg")) +
            glob.glob(os.path.join(test_dir, "*.png"))
        )
        
        print(f"[TileInferenceDataset] Found {len(self.image_paths)} images")
        print(f"  Tiles per image: {self.tiles_per_image}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def apply_fft_filter(self, image):
        """Apply FFT high-pass filter."""
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # High-pass filter
        mask = np.ones((rows, cols), np.float32)
        r = 30  # Radius of low frequencies to remove
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
        return img_back.astype(np.float32)
    
    def extract_all_tiles(self, image):
        """Extract all tiles from the image."""
        tiles = []
        positions = []
        
        for row in range(self.tiles_per_dim):
            for col in range(self.tiles_per_dim):
                y_start = row * self.stride
                x_start = col * self.stride
                
                tile = image[y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
                tiles.append(tile)
                positions.append((y_start, x_start))
        
        return tiles, positions
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.full_size, self.full_size), dtype=np.uint8)
        
        original_size = image.shape[:2]
        
        # Resize to full_size
        image = cv2.resize(image, (self.full_size, self.full_size))
        
        # Apply FFT if enabled
        if self.apply_fft:
            image = self.apply_fft_filter(image)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Extract all tiles
        tiles, positions = self.extract_all_tiles(image)
        
        # Convert to tensor format
        tiles_tensor = []
        for tile in tiles:
            tile = np.expand_dims(tile, axis=0)  # (1, H, W)
            tiles_tensor.append(tile)
        
        tiles_tensor = np.stack(tiles_tensor, axis=0)  # (num_tiles, 1, H, W)
        
        return {
            'tiles': tiles_tensor.astype(np.float32),
            'positions': positions,
            'image_path': img_path,
            'original_size': original_size,
            'full_image': image
        }


def stitch_tiles(tiles, positions, full_size, tile_size, overlap):
    """
    Stitch tiles back together with overlap blending.
    
    Args:
        tiles: List of tile predictions (numpy arrays)
        positions: List of (y, x) positions
        full_size: Size of full image
        tile_size: Size of each tile
        overlap: Overlap in pixels
    
    Returns:
        Stitched image
    """
    # Create output and weight arrays
    stitched = np.zeros((full_size, full_size), dtype=np.float32)
    weights = np.zeros((full_size, full_size), dtype=np.float32)
    
    # Create blending weight for each tile (higher in center, lower at edges)
    weight_tile = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Apply linear ramp at edges for smooth blending
    ramp_size = overlap
    for i in range(ramp_size):
        weight = (i + 1) / ramp_size
        weight_tile[i, :] *= weight
        weight_tile[-(i+1), :] *= weight
        weight_tile[:, i] *= weight
        weight_tile[:, -(i+1)] *= weight
    
    # Add each tile
    for tile, (y, x) in zip(tiles, positions):
        stitched[y:y+tile_size, x:x+tile_size] += tile * weight_tile
        weights[y:y+tile_size, x:x+tile_size] += weight_tile
    
    # Normalize by weights
    stitched = stitched / (weights + 1e-8)
    
    return stitched


if __name__ == "__main__":
    # Test the dataset
    # Test the dataset
    # Use relative paths for portability or correct absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    dataset = TileTrainDataset(
        image_dirs=[
            'data/train'
        ],
        anomaly_source_path=os.path.join(parent_dir, 'defect_patches3'),
        full_size=1024,
        tile_size=512,
        overlap=64
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Augmented shape: {sample['augmented_image'].shape}")
    print(f"Mask shape: {sample['anomaly_mask'].shape}")


class FullImageDatasetV2(Dataset):
    """
    V2 Dataset that returns all 6 inputs per image for end-to-end training.
    
    Returns:
        - 5 tiles (512x512) from corners + center
        - 1 full image resized to 512x512
        - Full 1024x1024 ground truth mask for stitched loss
    """
    
    def __init__(self, 
                 image_dirs,
                 anomaly_source_path,
                 full_size=1024,
                 tile_size=512,
                 overlap=64,
                 augment=True,
                 apply_fft=False):
        
        self.full_size = full_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.apply_fft = apply_fft
        
        # Fixed 5 tile positions
        self.tile_positions = [
            (0, 0),                                           # Top-left
            (0, full_size - tile_size),                       # Top-right
            (full_size - tile_size, 0),                       # Bottom-left
            (full_size - tile_size, full_size - tile_size),   # Bottom-right
            ((full_size - tile_size) // 2, (full_size - tile_size) // 2)  # Center
        ]
        
        # Load image paths
        self.image_paths = []
        for img_dir in image_dirs:
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.jpg")))
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.png")))
        self.image_paths = sorted(self.image_paths)
        
        # Load anomaly patches (reuse from TileTrainDataset logic)
        self.anomaly_source_paths = []
        if anomaly_source_path and os.path.exists(anomaly_source_path):
            self.anomaly_source_paths = sorted(
                glob.glob(os.path.join(anomaly_source_path, "**", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(anomaly_source_path, "**", "*.png"), recursive=True)
            )
        
        print(f"[FullImageDatasetV2] Found {len(self.image_paths)} images")
        print(f"  Full size: {full_size}×{full_size}")
        print(f"  Tile size: {tile_size}×{tile_size}")
        print(f"  Outputs per image: 6 (5 tiles + 1 full)")
        print(f"  Anomaly patches: {len(self.anomaly_source_paths)}")
        
        # Create a helper instance of TileTrainDataset for augmentation methods
        self._helper = TileTrainDataset(
            image_dirs=image_dirs,
            anomaly_source_path=anomaly_source_path,
            full_size=full_size,
            tile_size=tile_size,
            overlap=overlap,
            augment=augment,
            apply_fft=apply_fft
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return all 6 inputs for one image."""
        
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            # Return zeros if image can't be loaded
            return self._get_dummy_sample()
        
        # Resize to full_size
        image = cv2.resize(image, (self.full_size, self.full_size))
        image = image.astype(np.float32) / 255.0
        
        # Apply augmentation to full image (this creates defects on full 1024x1024)
        augmented_full, full_mask = self._helper.augment_full_image(image)
        
        # Extract 5 tiles + their masks
        tiles = []
        tile_masks = []
        positions = []
        
        for (y, x) in self.tile_positions:
            tile = augmented_full[y:y+self.tile_size, x:x+self.tile_size].copy()
            tile_mask = full_mask[y:y+self.tile_size, x:x+self.tile_size].copy()
            
            # Add channel dimension (1, H, W)
            tiles.append(np.expand_dims(tile, axis=0).astype(np.float32))
            tile_masks.append(np.expand_dims(tile_mask, axis=0).astype(np.float32))
            positions.append((y, x))
        
        # Full image resized to 512x512 (6th input)
        full_resized = cv2.resize(augmented_full, (self.tile_size, self.tile_size))
        full_resized = np.expand_dims(full_resized, axis=0).astype(np.float32)
        
        # Full mask resized to 512x512 (for the 6th input's local loss if needed)
        full_mask_resized = cv2.resize(full_mask, (self.tile_size, self.tile_size))
        full_mask_resized = np.expand_dims(full_mask_resized, axis=0).astype(np.float32)
        
        # Full mask at 1024x1024 (for stitched loss)
        full_mask_1024 = np.expand_dims(full_mask, axis=0).astype(np.float32)
        
        return {
            'tiles': np.stack(tiles, axis=0),  # (5, 1, 512, 512)
            'tile_masks': np.stack(tile_masks, axis=0),  # (5, 1, 512, 512)
            'positions': positions,  # list of 5 (y, x) tuples
            'full_resized': full_resized,  # (1, 512, 512)
            'full_mask_resized': full_mask_resized,  # (1, 512, 512)
            'full_mask_1024': full_mask_1024,  # (1, 1024, 1024) - for stitched loss
            'idx': idx
        }
    
    def _get_dummy_sample(self):
        """Return dummy sample for failed image loads."""
        dummy_tile = np.zeros((1, self.tile_size, self.tile_size), dtype=np.float32)
        return {
            'tiles': np.stack([dummy_tile] * 5, axis=0),
            'tile_masks': np.stack([dummy_tile] * 5, axis=0),
            'positions': self.tile_positions,
            'full_resized': dummy_tile,
            'full_mask_resized': dummy_tile,
            'full_mask_1024': np.zeros((1, self.full_size, self.full_size), dtype=np.float32),
            'idx': 0
        }


class FullImageDatasetV3(Dataset):
    """V3 Dataset: Per-tile-region augmentation.
    
    Applies 5-30 patches to EACH of the 5 tile regions on the full image,
    then crops. This guarantees dense coverage per tile while maintaining
    spatial consistency at tile boundaries.
    
    Expected patches per image: 25-150 (5 regions × 5-30 patches)
    """
    
    def __init__(self, 
                 image_dirs,
                 anomaly_source_path,
                 full_size=1024,
                 tile_size=512,
                 overlap=64,
                 augment=True,
                 apply_fft=False):
        
        self.full_size = full_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.apply_fft = apply_fft
        
        # Fixed 5 tile positions
        self.tile_positions = [
            (0, 0),                                           # Top-left
            (0, full_size - tile_size),                       # Top-right
            (full_size - tile_size, 0),                       # Bottom-left
            (full_size - tile_size, full_size - tile_size),   # Bottom-right
            ((full_size - tile_size) // 2, (full_size - tile_size) // 2)  # Center
        ]
        
        # Load image paths
        self.image_paths = []
        for img_dir in image_dirs:
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.jpg")))
            self.image_paths.extend(glob.glob(os.path.join(img_dir, "*.png")))
        self.image_paths = sorted(self.image_paths)
        
        # Load anomaly patches
        self.anomaly_source_paths = []
        if anomaly_source_path and os.path.exists(anomaly_source_path):
            self.anomaly_source_paths = sorted(
                glob.glob(os.path.join(anomaly_source_path, "**", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(anomaly_source_path, "**", "*.png"), recursive=True)
            )
        
        print(f"[FullImageDatasetV3] Found {len(self.image_paths)} images")
        print(f"  Full size: {full_size}×{full_size}")
        print(f"  Tile size: {tile_size}×{tile_size}")
        print(f"  Outputs per image: 6 (5 tiles + 1 full)")
        print(f"  Augmentation: V3 (5-30 patches PER tile region)")
        print(f"  Anomaly patches: {len(self.anomaly_source_paths)}")
        
        # Create helper for augmentation methods
        self._helper = TileTrainDataset(
            image_dirs=image_dirs,
            anomaly_source_path=anomaly_source_path,
            full_size=full_size,
            tile_size=tile_size,
            overlap=overlap,
            augment=augment,
            apply_fft=apply_fft
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return self._get_dummy_sample()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self._get_dummy_sample()
        
        # Resize to full_size
        image = cv2.resize(image, (self.full_size, self.full_size))
        image = image.astype(np.float32) / 255.0
        
        # V3 Augmentation: Apply 5-30 patches PER TILE REGION
        if self.augment:
            # Pass only the 4 corner tiles for augmentation placement to avoid over-saturating the center
            augmented_full, full_mask = self._helper.augment_full_image_v3(image, self.tile_positions[:-1])
        else:
            augmented_full = image
            full_mask = np.zeros((self.full_size, self.full_size), dtype=np.float32)
        
        # Extract 5 tiles + their masks
        tiles = []
        tile_masks = []
        positions = []
        
        for (y, x) in self.tile_positions:
            tile = augmented_full[y:y+self.tile_size, x:x+self.tile_size].copy()
            tile_mask = full_mask[y:y+self.tile_size, x:x+self.tile_size].copy()
            
            tiles.append(np.expand_dims(tile, axis=0).astype(np.float32))
            tile_masks.append(np.expand_dims(tile_mask, axis=0).astype(np.float32))
            positions.append((y, x))
        
        # Full image resized to tile_size
        full_resized = cv2.resize(augmented_full, (self.tile_size, self.tile_size))
        full_resized = np.expand_dims(full_resized, axis=0).astype(np.float32)
        
        # Full mask resized to tile_size
        full_mask_resized = cv2.resize(full_mask, (self.tile_size, self.tile_size))
        full_mask_resized = np.expand_dims(full_mask_resized, axis=0).astype(np.float32)
        
        # Full mask at original size
        full_mask_1024 = np.expand_dims(full_mask, axis=0).astype(np.float32)
        
        return {
            'tiles': np.stack(tiles, axis=0),  # (5, 1, 512, 512)
            'tile_masks': np.stack(tile_masks, axis=0),  # (5, 1, 512, 512)
            'positions': positions,
            'full_resized': full_resized,  # (1, 512, 512)
            'full_mask_resized': full_mask_resized,  # (1, 512, 512)
            'full_mask_1024': full_mask_1024,  # (1, 1024, 1024)
            'idx': idx
        }
    
    def _get_dummy_sample(self):
        dummy_tile = np.zeros((1, self.tile_size, self.tile_size), dtype=np.float32)
        return {
            'tiles': np.stack([dummy_tile] * 5, axis=0),
            'tile_masks': np.stack([dummy_tile] * 5, axis=0),
            'positions': self.tile_positions,
            'full_resized': dummy_tile,
            'full_mask_resized': dummy_tile,
            'full_mask_1024': np.zeros((1, self.full_size, self.full_size), dtype=np.float32),
            'idx': 0
        }


def differentiable_stitch(tile_preds, full_pred, positions, full_size, tile_size):
    """
    Differentiable stitching for backpropagation.
    
    Args:
        tile_preds: Tensor (5, 1, 512, 512) - predictions from 5 tiles
        full_pred: Tensor (1, 512, 512) - prediction from full resized image
        positions: List of (y, x) positions for each tile
        full_size: 1024
        tile_size: 512
    
    Returns:
        stitched: Tensor (1, 1024, 1024) - combined prediction
    """
    import torch
    import torch.nn.functional as F
    
    device = tile_preds.device
    batch_size = 1  # Single image processing
    
    # Initialize stitched output and weight map
    stitched = torch.zeros((batch_size, 1, full_size, full_size), device=device)
    weights = torch.zeros((batch_size, 1, full_size, full_size), device=device)
    
    # Place each tile prediction
    for i, (y, x) in enumerate(positions):
        stitched[:, :, y:y+tile_size, x:x+tile_size] += tile_preds[i:i+1]
        weights[:, :, y:y+tile_size, x:x+tile_size] += 1
    
    # Upscale full_pred to 1024x1024 and add
    full_upscaled = F.interpolate(full_pred.unsqueeze(0), size=(full_size, full_size), 
                                   mode='bilinear', align_corners=False)
    stitched += full_upscaled
    weights += 1
    
    # Average
    stitched = stitched / (weights + 1e-8)
    
    return stitched.squeeze(0)  # (1, 1024, 1024)
