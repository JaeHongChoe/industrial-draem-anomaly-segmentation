import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
# Writer will output to ./runs/ directory by default

class TensorboardVisualizer():

    def __init__(self,log_dir='./logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def visualize_image_batch(self,image_batch,n_iter,image_name='Image_batch'):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name,grid,n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        self.writer.add_scalar(loss_name, loss_val, n_iter)

    def visualize_overlay(self, images, masks, preds, n_iter, image_name='Overlay'):
        """
        Create overlay image:
        - Background: Gray image
        - Green: Ground Truth Mask
        - Red: Prediction Mask
        - Yellow: Overlap (TP)
        """
        # Ensure inputs are on CPU and numpy
        if isinstance(images, torch.Tensor): images = images.detach().cpu()
        if isinstance(masks, torch.Tensor): masks = masks.detach().cpu()
        if isinstance(preds, torch.Tensor): preds = preds.detach().cpu()
        
        batch_size = images.shape[0]
        overlays = []
        
        for i in range(batch_size):
            img = images[i]
            mask = masks[i]
            pred = preds[i]
            
            # Handle shapes
            if img.dim() == 3: img = img[0]
            if mask.dim() == 3: mask = mask[0]
            if pred.dim() == 3: pred = pred[0]
            
            # Normalize image to 0-1
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Create RGB canvas
            canvas = torch.stack([img, img, img], dim=0)  # (3, H, W)
            
            # Green for GT
            canvas[1] = torch.max(canvas[1], mask * 0.8) # Green channel
            
            # Red for Pred
            canvas[0] = torch.max(canvas[0], pred * 0.8) # Red channel
            
            overlays.append(canvas)
            
        grid = torchvision.utils.make_grid(torch.stack(overlays))
        self.writer.add_image(image_name, grid, n_iter)

