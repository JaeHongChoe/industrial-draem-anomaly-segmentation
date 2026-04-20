"""Tensorboard 로깅 헬퍼.

학습 루프에서 이미지 grid / loss scalar 를 기록하기 위한 얇은 wrapper.
현재 train.py 에서는 사용하지 않지만, 확장 시 활용할 수 있도록 보존.
"""
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class TensorboardVisualizer():
    """SummaryWriter 래퍼. log_dir 이 없으면 자동 생성."""

    def __init__(self, log_dir='./logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def visualize_image_batch(self, image_batch, n_iter, image_name='Image_batch'):
        """배치 이미지를 grid 로 묶어 Tensorboard 에 기록."""
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name, grid, n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        """scalar loss 값 기록."""
        self.writer.add_scalar(loss_name, loss_val, n_iter)
