"""
Tile-Based Training Script V4 (V1-style loop + V4 Augmentation)

Uses data_loader.py (TileTrainDataset) for training with 1024→512 tiles.
Features:
- Individual tile training (5 patches per image)
- V4 Augmentation (Unified blending, circular constraint, overlap check)
- DINOv3 Large + Focal Tversky Loss
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

# Local directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from tensorboard_visualizer import TensorboardVisualizer
from data_loader import TileTrainDataset, TileInferenceDataset, stitch_tiles


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--train_dirs', type=str, nargs='+', 
                       default=['data/train'],
                       help='Directories containing training images')
    parser.add_argument('--anomaly_source_path', type=str, 
                       default=os.path.join(current_dir, 'defect_patches3'),
                       help='Path to anomaly source patches')
    parser.add_argument('--real_defect_path', type=str, 
                       default='data/test',
                       help='Path to real defect images for evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to save checkpoints')
    parser.add_argument('--log_path', type=str, required=True, 
                       help='Path for tensorboard logs')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--dino_model', type=str, 
                       default='vit_large_patch16_dinov3.lvd1689m',
                       help='DINOv3 model name from timm')
    parser.add_argument('--full_size', type=int, default=1024, 
                       help='Full image size before tiling')
    parser.add_argument('--tile_size', type=int, default=512, 
                       help='Tile size')
    parser.add_argument('--overlap', type=int, default=64, 
                       help='Overlap between tiles')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                       help='Freeze DINOv3 encoder')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., tile_seg_epoch_35.pckl)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', 
                       help='Comma-separated GPU IDs to use')
    parser.add_argument('--num_workers', type=int, default=32,
                       help='Number of data loading workers') 
                       
    return parser.parse_args()


class DinoSegModelV3(torch.nn.Module):
    """DINOv3 encoder + U-Net style decoder for segmentation."""
    
    def __init__(self, dino_model='vit_large_patch16_dinov3.lvd1689m', 
                 freeze_encoder=True, input_size=512):
        super().__init__()
        import timm
        
        self.input_size = input_size
        self.patch_size = 16  # DINOv3 always uses patch16
        
        # Load DINOv3 encoder
        self.encoder = timm.create_model(
            dino_model,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Get embed dimension
        self.embed_dim = self.encoder.embed_dim  # 1024 for large, 768 for base
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # U-Net style decoder
        self.decoder = torch.nn.Sequential(
            # From (B, embed_dim, H/16, W/16) to (B, 256, H/8, W/8)
            torch.nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            # (B, 512, H/8, W/8) to (B, 256, H/4, W/4)
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            # (B, 256, H/4, W/4) to (B, 128, H/2, W/2)
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            # (B, 128, H/2, W/2) to (B, 64, H, W)
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            # Final output
            torch.nn.Conv2d(64, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Repeat grayscale to 3 channels if needed
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Get encoder features
        features = self.encoder.forward_features(x)
        
        # Handle different output formats (with/without registers)
        if len(features.shape) == 3:  # (B, num_tokens, embed_dim)
            # Remove CLS and register tokens
            # For DINOv3, typically first token is CLS
            num_patches = (H // self.patch_size) * (W // self.patch_size)
            features = features[:, -num_patches:, :]
            
            # Reshape to spatial format
            h = H // self.patch_size
            w = W // self.patch_size
            features = features.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        
        # Decode
        output = self.decoder(features)
        
        # Ensure output matches input size
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output


def focal_tversky_loss(pred, target, alpha=0.85, beta=0.15, gamma=0.75, smooth=1e-6):
    """
    Focal Tversky Loss for heavy penalty on missing small defects.
    alpha > beta means False Negatives are penalized more.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    true_pos = (pred * target).sum()
    false_neg = ((1 - pred) * target).sum()
    false_pos = (pred * (1 - target)).sum()
    
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    focal_tversky = (1 - tversky) ** gamma
    return focal_tversky


def stitch_tiles_for_viz(tiles, positions, full_size, tile_size, overlap):
    """
    Stitch tiles using MAX operation (Max projection) - Matches Inference Logic.
    Preserves the strongest defect signal in overlapping regions.
    """
    stitched = np.zeros((full_size, full_size), dtype=np.float32)
    
    for i, (y, x) in enumerate(positions):
        if isinstance(tiles, torch.Tensor):
            tile = tiles[i].squeeze().cpu().numpy()
        else:
            tile = tiles[i].squeeze()
            
        if isinstance(tile, torch.Tensor):
            tile = tile.numpy()
            
        # Current region in stitched map
        current_region = stitched[y:y+tile_size, x:x+tile_size]
        
        # Take element-wise maximum
        stitched[y:y+tile_size, x:x+tile_size] = np.maximum(current_region, tile)
        
    return stitched


def train_on_device(args):
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPUs: {args.gpus})")
    
    # Create directories
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # Model
    print(f"Loading DINO model: {args.dino_model}")
    print(f"Tile size: {args.tile_size}×{args.tile_size}")
    
    model = DinoSegModelV3(
        dino_model=args.dino_model,
        freeze_encoder=args.freeze_encoder,
        input_size=args.tile_size
    ).to(device)
    
    # Multi-GPU DataParallel
    if num_gpus > 1:
        print(f"Wrapping model in DataParallel for {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    # Print model info
    # Handle DataParallel wrapper for parameter counting
    model_to_inspect = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_params = sum(p.numel() for p in model_to_inspect.parameters())
    trainable_params = sum(p.numel() for p in model_to_inspect.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Data loader
    # Use pre-processed FFT data
    train_dirs = args.train_dirs
    print(f"Loading training data from: {train_dirs}")
    
    train_dataset = TileTrainDataset(
        image_dirs=train_dirs,
        anomaly_source_path=args.anomaly_source_path,
        full_size=args.full_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        augment=True,
        apply_fft=False  # Disable online FFT since data is pre-processed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,      # Reverted to aggressive prefetch (User Request)
        persistent_workers=True 
    )
    print(f"Training tiles: {len(train_dataset)}")
    
    # Visualizer
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, 'TileSeg'))
    
    # Training loop
    best_loss = float('inf')
    n_iter = 0
    start_epoch = 0
    
    # Resume logic
    if args.resume:
        resume_path = args.resume
        
        # Auto-resume latest
        if args.resume == 'latest':
            import glob
            # Find all pckl files in checkpoint path
            checkpoints = glob.glob(os.path.join(args.checkpoint_path, '*.pckl'))
            if not checkpoints:
                print(f"No checkpoints found in {args.checkpoint_path}. Starting from scratch.")
                resume_path = None
            else:
                # Sort by modification time (or epoch number if possible)
                # Ideally sort by epoch number to be safe
                def extract_epoch(f):
                    import re
                    match = re.search(r'epoch_(\d+)', f)
                    return int(match.group(1)) if match else -1
                
                latest_ckpt = max(checkpoints, key=extract_epoch)
                print(f"Auto-resume: Found latest checkpoint: {latest_ckpt}")
                resume_path = latest_ckpt
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Load model weights
            # Check if checkpoint is full dict or just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Fix 'module.' prefix mismatch (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load into unwrapped model
            model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_load.load_state_dict(new_state_dict)
            
            # Try to infer epoch from filename
            import re
            match = re.search(r'epoch_(\d+)', resume_path)
            if match:
                start_epoch = int(match.group(1))
                print(f"Resumed at epoch {start_epoch} (next epoch will be {start_epoch+1})")
                
                # Fast-forward scheduler to match the epoch
                print(f"Fast-forwarding scheduler by {start_epoch} steps...")
                for _ in range(start_epoch):
                    scheduler.step()
            else:
                print("Could not infer epoch from filename. Starting from epoch 0 but with loaded weights.")
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for intensity scaling
        train_dataset.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in pbar:
            aug_gray = batch["augmented_image"].to(device)
            anomaly_mask = batch["anomaly_mask"].to(device)
            
            # Forward
            pred_mask = model(aug_gray)
            
            # Loss
            loss = focal_tversky_loss(pred_mask, anomaly_mask, alpha=0.85, beta=0.15)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_iter += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.6f}'})
            
            # Visualization
            if args.visualize and n_iter % 100 == 0:
                visualizer.plot_loss(loss, n_iter, loss_name='train_loss')
            
            if args.visualize and n_iter % 500 == 0:
                visualizer.visualize_image_batch(
                    batch["image"][:4], n_iter, "train/original")
                visualizer.visualize_image_batch(
                    aug_gray[:4].cpu(), n_iter, "train/augmented")
                visualizer.visualize_image_batch(
                    anomaly_mask[:4].cpu(), n_iter, "train/gt_mask")
                visualizer.visualize_image_batch(
                    pred_mask[:4].detach().cpu(), n_iter, "train/pred_mask")
                # Overlay: Green=GT, Red=Pred, Yellow=Overlap
                visualizer.visualize_overlay(
                    aug_gray[:4].cpu(), 
                    anomaly_mask[:4].cpu(), 
                    pred_mask[:4].detach().cpu(), 
                    n_iter, "train/overlay_gt_pred")
                # Heatmap for predictions
                pred_heatmap = pred_mask[:4].detach().cpu()
                heatmap_colored = torch.zeros(4, 3, pred_heatmap.shape[2], pred_heatmap.shape[3])
                for j in range(min(4, pred_heatmap.shape[0])):
                    pm = (pred_heatmap[j, 0].numpy() * 255).astype(np.uint8)
                    hm = cv2.applyColorMap(pm, cv2.COLORMAP_JET)
                    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                    heatmap_colored[j] = torch.from_numpy(hm).permute(2, 0, 1).float() / 255
                visualizer.visualize_image_batch(heatmap_colored, n_iter, "train/pred_heatmap")
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # Real defect evaluation (every epoch)
        if args.real_defect_path and os.path.exists(args.real_defect_path):
            model.eval()
            total_iou = 0.0
            total_dice = 0.0
            num_samples = 0
            
            # Get list of test images
            import glob
            import json
            test_images = sorted(glob.glob(os.path.join(args.real_defect_path, "*.jpg")) +
                                glob.glob(os.path.join(args.real_defect_path, "*.png")))
            
            # Pick random indices for visualization
            import random
            viz_indices = random.sample(range(len(test_images)), min(3, len(test_images)))
            
            with torch.no_grad():
                for img_path in test_images:
                    # Load image
                    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    
                    # Convert to grayscale if needed
                    if image.ndim == 3:
                        if image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                        else:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # === Preprocessing (Matches all.py: High-Pass + Brightness/Contrast) ===
                    # Constants
                    FFT_PAD = 16
                    FFT_CUTOFF_RATIO = 0.12
                    FFT_GAUSS_SHARPNESS = 1.0
                    FFT_MATCH_MEAN_STD = True
                    TARGET_SIZE = (args.full_size, args.full_size) # 1024x1024

                    # 1. Uint8 to Float conversion (Handle potential 16-bit inputs)
                    # Note: Image loading above must be IMREAD_UNCHANGED!
                    if image.dtype == np.uint16:
                        g = (image.astype(np.float32) / 65535.0) * 255.0
                    else:
                        g = image.astype(np.float32)

                    # 2. Padding
                    big = cv2.copyMakeBorder(g, FFT_PAD, FFT_PAD, FFT_PAD, FFT_PAD, 
                                            borderType=cv2.BORDER_REFLECT_101)
                    H, W = big.shape

                    # 3. FFT
                    F = np.fft.fftshift(np.fft.fft2(big))
                    
                    # Gaussian Mask
                    cutoff_px = int(min(H, W) * FFT_CUTOFF_RATIO)
                    cy, cx = H // 2, W // 2
                    Y, X = np.ogrid[:H, :W]
                    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)
                    sigma = max(1.0, cutoff_px / max(0.1, FFT_GAUSS_SHARPNESS))
                    mask = np.exp(-(R * R) / (2.0 * sigma * sigma)).astype(np.float32)

                    # Apply filter & inverse FFT
                    out_big = np.fft.ifft2(np.fft.ifftshift(F * mask))
                    out_big = np.real(out_big).astype(np.float32)
                    out = out_big[FFT_PAD:H - FFT_PAD, FFT_PAD:W - FFT_PAD]

                    # Match mean/std
                    if FFT_MATCH_MEAN_STD:
                        m0, s0 = cv2.meanStdDev(g); m0, s0 = float(m0[0][0]), float(s0[0][0]) + 1e-6
                        m1, s1 = cv2.meanStdDev(out); m1, s1 = float(m1[0][0]), float(s1[0][0]) + 1e-6
                        out = (out - m1) * (s0 / s1) + m0
                    
                    out = np.clip(out, 0, 255).astype(np.uint8)

                    # 5. Brightness & Contrast Adjustment
                    # Percentile Stretching (10~97%)
                    low, high = np.percentile(out, (10, 97))
                    out = np.clip((out - low) * 255.0 / (high - low + 1e-6), 0, 255).astype(np.uint8)

                    # Gamma Correction (Gamma=1.4)
                    gamma = 1.4
                    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
                    out = cv2.LUT(out, table)

                    # 6. Resize and Normalize
                    img_resized = cv2.resize(out, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                    
                    # DEBUG: Save first preprocessed image to disk
                    if num_samples == 0:
                        cv2.imwrite("debug_val_preprocessed.jpg", img_resized)
                        print("Saved debug_val_preprocessed.jpg")
                    
                    img_fft = img_resized.astype(np.float32) / 255.0
                    
                    # Fixed 5 tile positions: 4 corners + 1 center (matching training)
                    tile_positions = [
                        (0, 0),                                                         # Top-left
                        (0, args.full_size - args.tile_size),                          # Top-right
                        (args.full_size - args.tile_size, 0),                          # Bottom-left
                        (args.full_size - args.tile_size, args.full_size - args.tile_size),  # Bottom-right
                        ((args.full_size - args.tile_size) // 2, (args.full_size - args.tile_size) // 2)  # Center
                    ]
                    
                    tile_preds = []
                    positions = []
                    
                    for (y, x) in tile_positions:
                        # Use img_fft (float 0-1) for model input
                        tile = img_fft[y:y+args.tile_size, x:x+args.tile_size]
                        tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
                        pred = model(tile_tensor).squeeze().cpu().numpy()
                        tile_preds.append(pred)
                        positions.append((y, x))
                    
                    # Stitch predictions
                    stitched_pred = stitch_tiles_for_viz(tile_preds, positions, 
                                                         args.full_size, args.tile_size, args.overlap)
                    
                    # Load ground truth mask from JSON
                    json_path = img_path.rsplit('.', 1)[0] + '.json'
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            gt_data = json.load(f)
                        
                        original_h = gt_data.get('imageHeight', image.shape[0])
                        original_w = gt_data.get('imageWidth', image.shape[1])
                        
                        gt_mask = np.zeros((args.full_size, args.full_size), dtype=np.float32)
                        for shape in gt_data.get('shapes', []):
                            pts = np.array(shape['points'], dtype=np.float32)
                            pts[:, 0] *= args.full_size / original_w
                            pts[:, 1] *= args.full_size / original_h
                            cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 1.0)
                        
                        # Calculate IoU
                        pred_binary = (stitched_pred > 0.5).astype(np.float32)
                        intersection = (pred_binary * gt_mask).sum()
                        union = pred_binary.sum() + gt_mask.sum() - intersection
                        iou = intersection / (union + 1e-8)
                        dice = 2 * intersection / (pred_binary.sum() + gt_mask.sum() + 1e-8)
                        
                        total_iou += iou
                        total_dice += dice
                        num_samples += 1
                        
                        # Visualize random validation samples every epoch
                        if args.visualize and (num_samples - 1) in viz_indices:
                            # Create overlay: Green=GT only, Red=Pred only, Yellow=overlap
                            # img_resized is already uint8 (0-255), do NOT multiply by 255 again
                            img_vis = img_resized.copy()
                            img_rgb = np.stack([img_vis, img_vis, img_vis], axis=-1)
                            
                            overlay = img_rgb.copy().astype(np.float32)
                            gt_bin = (gt_mask > 0.5)
                            pred_bin = (stitched_pred > 0.5)
                            
                            # GT only (green)
                            gt_only = gt_bin & ~pred_bin
                            overlay[gt_only, 1] = 255
                            
                            # Pred only (red)
                            pred_only = pred_bin & ~gt_bin
                            overlay[pred_only, 2] = 255
                            
                            # Overlap (yellow)
                            overlap_area = gt_bin & pred_bin
                            overlay[overlap_area, 0] = 255
                            overlay[overlap_area, 1] = 255
                            
                            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                            
                            # Create heatmap
                            heatmap = cv2.applyColorMap((stitched_pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            
                            # Log to TensorBoard
                            visualizer.visualize_image_batch(
                                torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255,
                                epoch, f'valid/input_{num_samples}'
                            )
                            visualizer.visualize_image_batch(
                                torch.from_numpy(overlay).permute(2, 0, 1).unsqueeze(0).float() / 255,
                                epoch, f'valid/overlay_{num_samples}'
                            )
                            visualizer.visualize_image_batch(
                                torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).float() / 255,
                                epoch, f'valid/heatmap_{num_samples}'
                            )
            
            if num_samples > 0:
                avg_iou = total_iou / num_samples
                avg_dice = total_dice / num_samples
                print(f"Epoch {epoch+1} - Real Defect IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
                
                if args.visualize:
                    visualizer.plot_loss(torch.tensor(avg_iou), epoch, loss_name='real_defect_iou')
                    visualizer.plot_loss(torch.tensor(avg_dice), epoch, loss_name='real_defect_dice')
            
            model.train()
        
        # Save checkpoint every epoch
        if True:
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_path, f'tile_seg_epoch_{epoch+1}.pckl'))
            print(f"Saved checkpoint: epoch_{epoch+1}")
            
        # Visualize stitched full image EVERY epoch (User Request)
        if args.visualize:
            model.eval()
            with torch.no_grad():
                # Show 3 different synthesized full images
                for k in range(3):
                    sample_idx = k  # 0, 1, 2
                    if sample_idx * train_dataset.tiles_per_image >= len(train_dataset):
                        break
                        
                    tiles = []
                    positions = []
                    gt_masks = []
                    pred_masks = []
                    
                    for tile_idx in range(train_dataset.tiles_per_image):
                        sample = train_dataset[sample_idx * train_dataset.tiles_per_image + tile_idx]
                        tiles.append(sample['augmented_image'])
                        gt_masks.append(sample['anomaly_mask'])
                        positions.append(sample['tile_pos'])
                    
                    # Stack and run inference
                    tiles_tensor = torch.stack([torch.from_numpy(t) for t in tiles]).to(device)
                    pred_tensor = model(tiles_tensor)
                    
                    # Stitch all components
                    stitched_input = stitch_tiles_for_viz(
                        [t for t in tiles], positions, 
                        args.full_size, args.tile_size, args.overlap)
                    stitched_gt = stitch_tiles_for_viz(
                        [m for m in gt_masks], positions,
                        args.full_size, args.tile_size, args.overlap)
                    stitched_pred = stitch_tiles_for_viz(
                        pred_tensor.detach().cpu(), positions,
                        args.full_size, args.tile_size, args.overlap)
                    
                    # Log to tensorboard (indexed by k)
                    # Overlay for stitched: Green=GT, Red=Pred, Yellow=Overlap
                    visualizer.visualize_overlay(
                        torch.from_numpy(stitched_input).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(stitched_gt).unsqueeze(0).unsqueeze(0),
                        torch.from_numpy(stitched_pred).unsqueeze(0).unsqueeze(0),
                        epoch, f"stitched/overlay_gt_pred_1024_sample_{k}")
            
            model.train()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.checkpoint_path, 'tile_seg_best.pckl'))
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(args.checkpoint_path, 'tile_seg_final.pckl'))
    print("Training complete!")


if __name__ == "__main__":
    args = get_args()
    train_on_device(args)
