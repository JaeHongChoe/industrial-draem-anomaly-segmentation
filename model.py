"""DRAEM 재구성(Autoencoder) 모델 정의 모음.

이 파일에는 실제 학습/추론에 쓰이는 AE_high 외에도,
실험 과정에서 만든 여러 변형(AE, AE_half, RAE*, U2NET 계열, AE_ultra 등) 이 함께 남아있다.
현재 config.ae_type 기본값은 'AE_high' 이며, 나머지는 과거 실험 히스토리 보존용이다.

클래스 요약:
    - BottleneckBlock / ResidualBlock: 재사용 블록.
    - AE / AE_high / AE_half / AE_high_half: 인코더-디코더 계열 변형.
    - RAE*: Residual 버전 AE (스킵 연결 유무 차이).
    - U2NET / U2NETP_AE / RSU*_NoSkip: 실험용 U^2-Net 기반 재구성 모델.
    - AE_ultra: 최신 실험 변형.
segmentation 브랜치는 model_unet.py 의 DiscriminativeSubNetwork 를 사용.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_unet import DiscriminativeSubNetwork, ReconstructiveSubNetwork


def _extract_state_dict(loaded):
    """torch.load 결과에서 실제 state_dict dict 를 꺼낸다.

    체크포인트 저장 방식이 섞여있어 {state_dict, model_state_dict} 키가 있거나
    곧장 state_dict 가 오는 경우 모두 대응.
    """
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            return loaded["model_state_dict"]
    return loaded


def _strip_module_prefix(state_dict):
    """DataParallel 저장본의 'module.' 접두사 제거."""
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def load_model(model, model_path, device):
    """체크포인트 로드 헬퍼. state_dict 정제 → device 이동 → eval 모드."""
    state_dict = torch.load(model_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(state_dict))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class BottleneckBlock(nn.Module):
    """ResNet-style bottleneck (1x1 → 3x3 → 1x1) 블록."""

    def __init__(self, in_channels, out_channels, expansion=4):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        return self.relu(out)
    
class ResidualBlock(nn.Module):
    """기본 Residual block (3x3 Conv → BN → ReLU → 3x3 Conv → BN + shortcut)."""

    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out

class AE(nn.Module):
    """기본 ResidualBlock 기반 Autoencoder. 입력 1채널 → 5단계 다운샘플 → 복원.

    현재 실험에서는 AE_high 를 기본 채택하고 있고, AE 는 baseline 비교용이다.
    """

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, stride=2, padding=0), # 512 => 256
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0), # 256 => 128
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0), # 128 => 64
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0), # 64 => 32
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0) # 32 => 16
            # 16 x 16 x  512 => 131,072
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE_high(nn.Module):
    """**본 프로젝트의 메인 재구성 네트워크.**

    AE 대비 각 stage 앞에 'Conv-ReLU' 를 한 층 더 끼운 고용량 버전.
    encoder 는 6단계 (1→32→64→128→256→512→1024 채널) 로 다운샘플,
    decoder 는 대칭 구조로 bicubic upsample + ResidualBlock 구성.
    최종 Sigmoid 로 0~1 범위 grayscale 이미지 복원.
    """

    def __init__(self):
        super(AE_high, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(512, 1024),
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(1024, 512),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# class AE_high(nn.Module):
#     def __init__(self):
#         super(AE_high, self).__init__()
#         self.encoder = nn.Sequential(
#             ResidualBlock(1, 32),
#             nn.MaxPool2d(2, stride=2, padding=0),

#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.ReLU(),
#             ResidualBlock(32, 64),
#             nn.MaxPool2d(2, stride=2, padding=0),

#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.ReLU(),
#             ResidualBlock(64, 128),
#             nn.MaxPool2d(2, stride=2, padding=0),

#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128, 256),
#             nn.MaxPool2d(2, stride=2, padding=0),
            
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.ReLU(),
#             ResidualBlock(256, 512),
#             nn.MaxPool2d(2, stride=2, padding=0),

#             nn.Conv2d(512, 512, 3, stride=1, padding=1),
#             nn.ReLU(),
#             ResidualBlock(512, 1024),
#             nn.MaxPool2d(2, stride=2, padding=0),
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             ResidualBlock(1024, 512),
            
#             nn.Conv2d(512, 512, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             ResidualBlock(512, 256),
            
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             ResidualBlock(256, 128),

#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.ReLU(),            
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             ResidualBlock(128, 64),

#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.ReLU(),            
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             ResidualBlock(64, 32),

#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.ReLU(),            
#             nn.Upsample(scale_factor=2, mode='bicubic'),
#             nn.Conv2d(32, 1, 3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class AE_half(nn.Module):
    """AE 대비 채널 수를 절반으로 줄인 경량 변형 (실험용)."""

    def __init__(self):
        super(AE_half, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, stride=2, padding=0), # 512 => 256
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, stride=2, padding=0), # 256 => 128
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0), # 128 => 64
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0), # 64 => 32
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0) # 32 => 16
            # 16 x 16 x 256 => 65,536
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 16),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE_high_half(nn.Module):
    """AE_high 의 채널 절반 + 한 단계 더 깊은 다운샘플 변형 (실험용)."""

    def __init__(self):
        super(AE_high_half, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, stride=2, padding=0), # 512 => 256
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, stride=2, padding=0), # 256 => 128
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0), # 128 => 64
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0), # 64 => 32
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0), # 32 => 16
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0) # 16 => 8
            # 8 x 8 x 512 = 32,768
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 16),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RAE(nn.Module):
    """Residual AE 계열 baseline. 디코더 마지막도 ResidualBlock → Sigmoid (실험용)."""

    def __init__(self):
        super(RAE, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RAE_high(nn.Module):
    """RAE 의 더 깊은 버전 (6단계 다운샘플, 실험용)."""

    def __init__(self):
        super(RAE_high, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(512, 1024),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(1024, 1024),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(1024, 512),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RAE_half(nn.Module):
    """RAE 의 채널 절반 경량 버전 (실험용)."""

    def __init__(self):
        super(RAE_half, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 16),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class RAE_high_half(nn.Module):
    """RAE_high 의 채널 절반 경량 버전 (실험용)."""

    def __init__(self):
        super(RAE_high_half, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(16, 32),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(32, 16),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# class AttentionGate(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionGate, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         identity = self.skip(x)
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += identity
#         return self.relu(out)

# class AE_high(nn.Module):
#     def __init__(self):
#         super(AE_high, self).__init__()

#         # Encoder
#         self.enc1 = ResidualBlock(1, 32)
#         self.pool1 = nn.MaxPool2d(2)  # 640 → 320

#         self.enc2 = ResidualBlock(32, 64)
#         self.pool2 = nn.MaxPool2d(2)  # 320 → 160

#         self.enc3 = ResidualBlock(64, 128)
#         self.pool3 = nn.MaxPool2d(2)  # 160 → 80

#         self.enc4 = ResidualBlock(128, 256)
#         self.pool4 = nn.MaxPool2d(2)  # 80 → 40

#         self.enc5 = ResidualBlock(256, 512)
#         self.pool5 = nn.MaxPool2d(2)  # 40 → 20

#         self.bottleneck = ResidualBlock(512, 1024)  # 20 → latent

#         # Decoder
#         self.up5 = nn.Upsample(scale_factor=2, mode='bicubic')
#         self.att5 = AttentionGate(1024, 512, 256)
#         self.dec5 = ResidualBlock(1024 + 512, 512)

#         self.up4 = nn.Upsample(scale_factor=2, mode='bicubic')
#         self.att4 = AttentionGate(512, 256, 128)
#         self.dec4 = ResidualBlock(512 + 256, 256)

#         self.up3 = nn.Upsample(scale_factor=2, mode='bicubic')
#         self.att3 = AttentionGate(256, 128, 64)
#         self.dec3 = ResidualBlock(256 + 128, 128)

#         self.up2 = nn.Upsample(scale_factor=2, mode='bicubic')
#         self.att2 = AttentionGate(128, 64, 32)
#         self.dec2 = ResidualBlock(128 + 64, 64)

#         self.up1 = nn.Upsample(scale_factor=2, mode='bicubic')
#         self.att1 = AttentionGate(64, 32, 16)
#         self.dec1 = ResidualBlock(64 + 32, 32)

#         self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
#         self.activation = nn.Sigmoid()

#     def forward(self, x):
#         # Encoder
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool1(x1))
#         x3 = self.enc3(self.pool2(x2))
#         x4 = self.enc4(self.pool3(x3))
#         x5 = self.enc5(self.pool4(x4))
#         x6 = self.bottleneck(self.pool5(x5))

#         # Decoder
#         d5 = self.up5(x6)
#         x5_att = self.att5(d5, x5)
#         d5 = self.dec5(torch.cat([d5, x5_att], dim=1))

#         d4 = self.up4(d5)
#         x4_att = self.att4(d4, x4)
#         d4 = self.dec4(torch.cat([d4, x4_att], dim=1))

#         d3 = self.up3(d4)
#         x3_att = self.att3(d3, x3)
#         d3 = self.dec3(torch.cat([d3, x3_att], dim=1))

#         d2 = self.up2(d3)
#         x2_att = self.att2(d2, x2)
#         d2 = self.dec2(torch.cat([d2, x2_att], dim=1))

#         d1 = self.up1(d2)
#         x1_att = self.att1(d1, x1)
#         d1 = self.dec1(torch.cat([d1, x1_att], dim=1))

#         out = self.activation(self.final_conv(d1))
#         return out



## 이건 외부 u-net끼리 스킵커넥션이 있는 버전 ##
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class REBNCONV(nn.Module):
#     def __init__(self,in_ch=3,out_ch=3,dirate=1):
#         super(REBNCONV,self).__init__()

#         self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
#         self.bn_s1 = nn.BatchNorm2d(out_ch)
#         self.relu_s1 = nn.ReLU(inplace=True)

#     def forward(self,x):

#         hx = x
#         xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

#         return xout

# ## upsample tensor 'src' to have the same spatial size with tensor 'tar'
# def _upsample_like(src,tar):

#     src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

#     return src


# ### RSU-7 (No Skip Connection) ###
# class RSU7(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU7,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

#         self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

#         self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

#     def forward(self,x):

#         hx = x
#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)

#         hx5 = self.rebnconv5(hx)
#         hx = self.pool5(hx5)

#         hx6 = self.rebnconv6(hx)

#         hx7 = self.rebnconv7(hx6)

#         hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
#         hx6dup = _upsample_like(hx6d,hx5)

#         hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         # Removed residual connection: return hx1d + hxin
#         return hx1d

# ### RSU-6 (No Skip Connection) ###
# class RSU6(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU6,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)

#         hx5 = self.rebnconv5(hx)

#         hx6 = self.rebnconv6(hx5)


#         hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         # Removed residual connection: return hx1d + hxin
#         return hx1d

# ### RSU-5 (No Skip Connection) ###
# class RSU5(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU5,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)

#         hx4 = self.rebnconv4(hx)

#         hx5 = self.rebnconv5(hx4)

#         hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         # Removed residual connection: return hx1d + hxin
#         return hx1d

# ### RSU-4 (No Skip Connection) ###
# class RSU4(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU4,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)

#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)

#         hx3 = self.rebnconv3(hx)

#         hx4 = self.rebnconv4(hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

#         # Removed residual connection: return hx1d + hxin
#         return hx1d

# ### RSU-4F (No Skip Connection) ###
# class RSU4F(nn.Module):

#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU4F,self).__init__()

#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

#     def forward(self,x):

#         hx = x

#         hxin = self.rebnconvin(hx)

#         hx1 = self.rebnconv1(hxin)
#         hx2 = self.rebnconv2(hx1)
#         hx3 = self.rebnconv3(hx2)

#         hx4 = self.rebnconv4(hx3)

#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
#         hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

#         # Removed residual connection: return hx1d + hxin
#         return hx1d


# ##### U^2-Net (No Encoder-Decoder Skip Connections) ####
# class U2NET(nn.Module):

#     def __init__(self,in_ch=1,out_ch=1):
#         super(U2NET,self).__init__()

#         self.stage1 = RSU7(in_ch,32,64)
#         self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage2 = RSU6(64,32,128)
#         self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage3 = RSU5(128,64,256)
#         self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage4 = RSU4(256,128,512)
#         self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage5 = RSU4F(512,256,512)
#         self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage6 = RSU4F(512,256,512)

#         # decoder - modified to not use encoder features
#         self.stage5d = RSU4F(512,256,512)  # Changed from 1024 to 512
#         self.stage4d = RSU4(512,128,256)   # Changed from 1024 to 512
#         self.stage3d = RSU5(256,64,128)    # Changed from 512 to 256
#         self.stage2d = RSU6(128,32,64)     # Changed from 256 to 128
#         self.stage1d = RSU7(64,16,64)      # Changed from 128 to 64

#         self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
#         self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
#         self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
#         self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

#         self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

#     def forward(self,x):

#         hx = x

#         #stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)

#         #stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)

#         #stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)

#         #stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)

#         #stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)

#         #stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6,hx5)

#         #-------------------- decoder (No skip connections from encoder) --------------------
#         hx5d = self.stage5d(hx6up)  # Removed: torch.cat((hx6up,hx5),1)
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.stage4d(hx5dup)  # Removed: torch.cat((hx5dup,hx4),1)
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.stage3d(hx4dup)  # Removed: torch.cat((hx4dup,hx3),1)
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.stage2d(hx3dup)  # Removed: torch.cat((hx3dup,hx2),1)
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.stage1d(hx2dup)  # Removed: torch.cat((hx2dup,hx1),1)


#         #side output
#         d1 = self.side1(hx1d)

#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2,d1)

#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3,d1)

#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4,d1)

#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5,d1)

#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6,d1)

#         d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

#         return F.sigmoid(d0)

# ### U^2-Net small (No Encoder-Decoder Skip Connections) ###
# class U2NETP(nn.Module):

#     def __init__(self,in_ch=3,out_ch=1):
#         super(U2NETP,self).__init__()

#         self.stage1 = RSU7(in_ch,16,64)
#         self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage2 = RSU6(64,16,64)
#         self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage3 = RSU5(64,16,64)
#         self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage4 = RSU4(64,16,64)
#         self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage5 = RSU4F(64,16,64)
#         self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

#         self.stage6 = RSU4F(64,16,64)

#         # decoder - modified to not use encoder features
#         self.stage5d = RSU4F(64,16,64)  # Changed from 128 to 64
#         self.stage4d = RSU4(64,16,64)   # Changed from 128 to 64
#         self.stage3d = RSU5(64,16,64)   # Changed from 128 to 64
#         self.stage2d = RSU6(64,16,64)   # Changed from 128 to 64
#         self.stage1d = RSU7(64,16,64)   # Changed from 128 to 64

#         self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

#         self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

#     def forward(self,x):

#         hx = x

#         #stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)

#         #stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)

#         #stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)

#         #stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)

#         #stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)

#         #stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6,hx5)

#         #decoder (No skip connections from encoder)
#         hx5d = self.stage5d(hx6up)  # Removed: torch.cat((hx6up,hx5),1)
#         hx5dup = _upsample_like(hx5d,hx4)

#         hx4d = self.stage4d(hx5dup)  # Removed: torch.cat((hx5dup,hx4),1)
#         hx4dup = _upsample_like(hx4d,hx3)

#         hx3d = self.stage3d(hx4dup)  # Removed: torch.cat((hx4dup,hx3),1)
#         hx3dup = _upsample_like(hx3d,hx2)

#         hx2d = self.stage2d(hx3dup)  # Removed: torch.cat((hx3dup,hx2),1)
#         hx2dup = _upsample_like(hx2d,hx1)

#         hx1d = self.stage1d(hx2dup)  # Removed: torch.cat((hx2dup,hx1),1)


#         #side output
#         d1 = self.side1(hx1d)

#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2,d1)

#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3,d1)

#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4,d1)

#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5,d1)

#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6,d1)

#         d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

#         return F.sigmoid(d0)





## 이건 모든 스킵 커넥션을 제거한 버전##
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------------
# # Small building block
# # -----------------------------
# class REBNCONV(nn.Module):
#     def __init__(self, in_ch=3, out_ch=3, dirate=1, norm='bn'):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate, bias=False)
#         if norm == 'bn':
#             self.norm = nn.BatchNorm2d(out_ch)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm2d(out_ch, affine=True)
#         else:
#             self.norm = nn.Identity()
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.act(self.norm(self.conv(x)))


# def _upsample_like(src, tar):
#     # F.upsample deprecated → use interpolate
#     return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


# # =========================================================
# # RSU Blocks (NO internal skip / NO concat in decoder path)
# # =========================================================
# class RSU7_NoSkip(nn.Module):
#     """
#     Original RSU7 has internal U-shape with concat skips.
#     This version removes all concats → decoder takes only its own stream.
#     """
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
#         super().__init__()
#         self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, norm=norm)

#         self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, norm=norm); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

#         self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)

#         # decoder (no concat → all inputs are mid_ch)
#         self.rebnconv6d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv5d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch, out_ch, dirate=1, norm=norm)

#     def forward(self, x):
#         hxin = self.rebnconvin(x)

#         h1 = self.rebnconv1(hxin); d = self.pool1(h1)
#         h2 = self.rebnconv2(d);     d = self.pool2(h2)
#         h3 = self.rebnconv3(d);     d = self.pool3(h3)
#         h4 = self.rebnconv4(d);     d = self.pool4(h4)
#         h5 = self.rebnconv5(d);     d = self.pool5(h5)
#         h6 = self.rebnconv6(d)
#         h7 = self.rebnconv7(h6)

#         u6 = _upsample_like(self.rebnconv6d(h7), h5)
#         u5 = _upsample_like(self.rebnconv5d(u6), h4)
#         u4 = _upsample_like(self.rebnconv4d(u5), h3)
#         u3 = _upsample_like(self.rebnconv3d(u4), h2)
#         u2 = _upsample_like(self.rebnconv2d(u3), h1)
#         u1 = self.rebnconv1d(u2)

#         # no residual add with hxin (pure AE style)
#         return u1


# class RSU6_NoSkip(nn.Module):
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
#         super().__init__()
#         self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, norm=norm)

#         self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, norm=norm); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)

#         self.rebnconv5d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch, out_ch, dirate=1, norm=norm)

#     def forward(self, x):
#         hxin = self.rebnconvin(x)
#         h1 = self.rebnconv1(hxin); d = self.pool1(h1)
#         h2 = self.rebnconv2(d);     d = self.pool2(h2)
#         h3 = self.rebnconv3(d);     d = self.pool3(h3)
#         h4 = self.rebnconv4(d);     d = self.pool4(h4)
#         h5 = self.rebnconv5(d)
#         h6 = self.rebnconv6(h5)

#         u5 = _upsample_like(self.rebnconv5d(h6), h4)
#         u4 = _upsample_like(self.rebnconv4d(u5), h3)
#         u3 = _upsample_like(self.rebnconv3d(u4), h2)
#         u2 = _upsample_like(self.rebnconv2d(u3), h1)
#         u1 = self.rebnconv1d(u2)
#         return u1


# class RSU5_NoSkip(nn.Module):
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
#         super().__init__()
#         self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, norm=norm)
#         self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, norm=norm); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)

#         self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch, out_ch, dirate=1, norm=norm)

#     def forward(self, x):
#         hxin = self.rebnconvin(x)
#         h1 = self.rebnconv1(hxin); d = self.pool1(h1)
#         h2 = self.rebnconv2(d);     d = self.pool2(h2)
#         h3 = self.rebnconv3(d)
#         h4 = self.rebnconv4(h3)

#         u3 = _upsample_like(self.rebnconv3d(h4), h2)
#         u2 = _upsample_like(self.rebnconv2d(u3), h1)
#         u1 = self.rebnconv1d(u2)
#         return u1


# class RSU4_NoSkip(nn.Module):
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
#         super().__init__()
#         self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, norm=norm)
#         self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, norm=norm); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)

#         self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch, out_ch, dirate=1, norm=norm)

#     def forward(self, x):
#         hxin = self.rebnconvin(x)
#         h1 = self.rebnconv1(hxin); d = self.pool1(h1)
#         h2 = self.rebnconv2(d)
#         h3 = self.rebnconv3(h2)

#         u2 = _upsample_like(self.rebnconv2d(h3), h1)
#         u1 = self.rebnconv1d(u2)
#         return u1


# class RSU4F_NoSkip(nn.Module):
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
#         super().__init__()
#         self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1, norm=norm)
#         self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1, norm=norm)
#         self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)
#         self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4, norm=norm)
#         self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8, norm=norm)

#         self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=4, norm=norm)
#         self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm)
#         self.rebnconv1d = REBNCONV(mid_ch, out_ch, dirate=1, norm=norm)

#     def forward(self, x):
#         hxin = self.rebnconvin(x)
#         h1 = self.rebnconv1(hxin)
#         h2 = self.rebnconv2(h1)
#         h3 = self.rebnconv3(h2)
#         h4 = self.rebnconv4(h3)

#         d3 = self.rebnconv3d(h4)
#         d2 = self.rebnconv2d(d3)
#         d1 = self.rebnconv1d(d2)
#         return d1


# # =========================================================
# # U2NET / U2NETP (AE2Net style, NO skips anywhere)
# # =========================================================
# class U2NET(nn.Module):
#     """
#     AE-style U²-Net: no encoder↔decoder skips and no RSU internal concats.
#     Keeps side outputs (from decoder features) for deep supervision / fusion.
#     """
#     def __init__(self, in_ch=1, out_ch=1, norm='bn'):
#         super().__init__()

#         self.stage1 = RSU7_NoSkip(in_ch, 32, 64, norm=norm);  self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage2 = RSU6_NoSkip(64, 32, 128, norm=norm);    self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage3 = RSU5_NoSkip(128, 64, 256, norm=norm);   self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage4 = RSU4_NoSkip(256, 128, 512, norm=norm);  self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage5 = RSU4F_NoSkip(512, 256, 512, norm=norm); self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage6 = RSU4F_NoSkip(512, 256, 512, norm=norm)

#         # decoder (no top-level skips)
#         self.stage5d = RSU4F_NoSkip(512, 256, 512, norm=norm)
#         self.stage4d = RSU4_NoSkip(512, 128, 256, norm=norm)
#         self.stage3d = RSU5_NoSkip(256, 64, 128, norm=norm)
#         self.stage2d = RSU6_NoSkip(128, 32, 64, norm=norm)
#         self.stage1d = RSU7_NoSkip(64, 16, 64, norm=norm)

#         # side outputs from decoder features only
#         self.side1 = nn.Conv2d(64,  out_ch, 3, padding=1)
#         self.side2 = nn.Conv2d(64,  out_ch, 3, padding=1)
#         self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
#         self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
#         self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
#         self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

#         self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

#     def forward(self, x):
#         # encoder
#         h1 = self.stage1(x);   d = self.pool12(h1)
#         h2 = self.stage2(d);   d = self.pool23(h2)
#         h3 = self.stage3(d);   d = self.pool34(h3)
#         h4 = self.stage4(d);   d = self.pool45(h4)
#         h5 = self.stage5(d);   d = self.pool56(h5)
#         h6 = self.stage6(d)

#         # decoder (no top-level concat)
#         u5 = _upsample_like(self.stage5d(h6), h4)
#         u4 = _upsample_like(self.stage4d(u5), h3)
#         u3 = _upsample_like(self.stage3d(u4), h2)
#         u2 = _upsample_like(self.stage2d(u3), h1)
#         u1 = self.stage1d(u2)

#         # side outputs (from decoder states only)
#         d1 = self.side1(u1)
#         d2 = _upsample_like(self.side2(u2), d1)
#         d3 = _upsample_like(self.side3(u3), d1)
#         d4 = _upsample_like(self.side4(u4), d1)
#         d5 = _upsample_like(self.side5(u5), d1)
#         d6 = _upsample_like(self.side6(h6), d1)  # bottleneck supervision

#         d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

#         # If you plan to use BCEWithLogitsLoss, return raw d0 (no sigmoid).
#         return torch.sigmoid(d0)  # <- change to `return d0` for BCEWithLogitsLoss


# class U2NETP_AE(nn.Module):
#     """Lightweight AE-style U²-Net"""
#     def __init__(self, in_ch=1, out_ch=1, norm='bn'):
#         super().__init__()
#         self.stage1 = RSU7_NoSkip(in_ch, 16, 64, norm=norm);  self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage2 = RSU6_NoSkip(64, 16, 64, norm=norm);     self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage3 = RSU5_NoSkip(64, 16, 64, norm=norm);     self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage4 = RSU4_NoSkip(64, 16, 64, norm=norm);     self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage5 = RSU4F_NoSkip(64, 16, 64, norm=norm);    self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.stage6 = RSU4F_NoSkip(64, 16, 64, norm=norm)

#         self.stage5d = RSU4F_NoSkip(64, 16, 64, norm=norm)
#         self.stage4d = RSU4_NoSkip(64, 16, 64, norm=norm)
#         self.stage3d = RSU5_NoSkip(64, 16, 64, norm=norm)
#         self.stage2d = RSU6_NoSkip(64, 16, 64, norm=norm)
#         self.stage1d = RSU7_NoSkip(64, 16, 64, norm=norm)

#         self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

#         self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

#     def forward(self, x):
#         h1 = self.stage1(x);  d = self.pool12(h1)
#         h2 = self.stage2(d);  d = self.pool23(h2)
#         h3 = self.stage3(d);  d = self.pool34(h3)
#         h4 = self.stage4(d);  d = self.pool45(h4)
#         h5 = self.stage5(d);  d = self.pool56(h5)
#         h6 = self.stage6(d)

#         u5 = _upsample_like(self.stage5d(h6), h4)
#         u4 = _upsample_like(self.stage4d(u5), h3)
#         u3 = _upsample_like(self.stage3d(u4), h2)
#         u2 = _upsample_like(self.stage2d(u3), h1)
#         u1 = self.stage1d(u2)

#         d1 = self.side1(u1)
#         d2 = _upsample_like(self.side2(u2), d1)
#         d3 = _upsample_like(self.side3(u3), d1)
#         d4 = _upsample_like(self.side4(u4), d1)
#         d5 = _upsample_like(self.side5(u5), d1)
#         d6 = _upsample_like(self.side6(h6), d1)

#         d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
#         return torch.sigmoid(d0)  # or return d0 for BCEWithLogitsLoss




## 이 아래는 U^2-Net 을 AE 처럼 쓰기 위한 실험 세트 (RSU*_NoSkip → U2NET / U2NETP_AE / AE_ultra).
## 내부에 residual 만 추가해 AE 와 비슷한 동작을 시도한 변형이라 별도 import/block 정의가 이어짐.
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Residual / 기본 convolution 블록
# -----------------------------
class ResidualWrapper(nn.Module):
    """임의 block 을 감싸 residual 연결을 추가.

    out = ReLU( block(x) + Proj(x) )
    - Proj: 입출력 채널이 같으면 identity, 다르면 1x1 conv 로 매칭 (BN/IN 은 생략).
    """
    def __init__(self, block: nn.Module, in_ch: int, out_ch: int):
        super().__init__()
        self.block = block
        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.block(x)
        return self.act(y + self.proj(x))

class REBNCONV(nn.Module):
    """U^2-Net 의 기본 conv 블록: 3x3 conv (optional dilation) → Norm → ReLU."""

    def __init__(self, in_ch=3, out_ch=3, dirate=1, norm='bn'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate, bias=False)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def _upsample_like(src, tar):
    """src 를 tar 의 공간 크기 (H, W) 로 bilinear upsample."""
    # F.upsample 은 deprecated → F.interpolate 사용.
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


# =========================================================
# RSU Blocks (NO internal skip / NO concat in decoder path)
# =========================================================
class RSU7_NoSkip(nn.Module):
    """RSU-7 (Residual U-Block) without internal skip connections.

    원본 U^2-Net 과 달리 decoder 에서 encoder feature 와의 concat 을 생략해
    AE 처럼 단방향 재구성 흐름만 남긴 변형.
    """

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
        super().__init__()
        self.rebnconvin = ResidualWrapper(REBNCONV(in_ch, out_ch, dirate=1, norm=norm), in_ch, out_ch)

        self.rebnconv1 = ResidualWrapper(REBNCONV(out_ch, mid_ch, dirate=1, norm=norm), out_ch, mid_ch); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv4 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv5 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.rebnconv6 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv7 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)

        self.rebnconv6d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv5d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv4d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv3d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv2d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv1d = ResidualWrapper(REBNCONV(mid_ch, out_ch, dirate=1, norm=norm), mid_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); d = self.pool1(h1)
        h2 = self.rebnconv2(d);     d = self.pool2(h2)
        h3 = self.rebnconv3(d);     d = self.pool3(h3)
        h4 = self.rebnconv4(d);     d = self.pool4(h4)
        h5 = self.rebnconv5(d);     d = self.pool5(h5)
        h6 = self.rebnconv6(d)
        h7 = self.rebnconv7(h6)

        u6 = _upsample_like(self.rebnconv6d(h7), h5)
        u5 = _upsample_like(self.rebnconv5d(u6), h4)
        u4 = _upsample_like(self.rebnconv4d(u5), h3)
        u3 = _upsample_like(self.rebnconv3d(u4), h2)
        u2 = _upsample_like(self.rebnconv2d(u3), h1)
        u1 = self.rebnconv1d(u2)
        return u1


class RSU6_NoSkip(nn.Module):
    """RSU-6 (6 단계 다운샘플), skip 제거 변형."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
        super().__init__()
        self.rebnconvin = ResidualWrapper(REBNCONV(in_ch, out_ch, dirate=1, norm=norm), in_ch, out_ch)

        self.rebnconv1 = ResidualWrapper(REBNCONV(out_ch, mid_ch, dirate=1, norm=norm), out_ch, mid_ch); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv4 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv5 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv6 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)

        self.rebnconv5d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv4d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv3d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv2d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv1d = ResidualWrapper(REBNCONV(mid_ch, out_ch, dirate=1, norm=norm), mid_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); d = self.pool1(h1)
        h2 = self.rebnconv2(d);     d = self.pool2(h2)
        h3 = self.rebnconv3(d);     d = self.pool3(h3)
        h4 = self.rebnconv4(d);     d = self.pool4(h4)
        h5 = self.rebnconv5(d)
        h6 = self.rebnconv6(h5)

        u5 = _upsample_like(self.rebnconv5d(h6), h4)
        u4 = _upsample_like(self.rebnconv4d(u5), h3)
        u3 = _upsample_like(self.rebnconv3d(u4), h2)
        u2 = _upsample_like(self.rebnconv2d(u3), h1)
        u1 = self.rebnconv1d(u2)
        return u1


class RSU5_NoSkip(nn.Module):
    """RSU-5 (5 단계 다운샘플), skip 제거 변형."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
        super().__init__()
        self.rebnconvin = ResidualWrapper(REBNCONV(in_ch, out_ch, dirate=1, norm=norm), in_ch, out_ch)
        self.rebnconv1 = ResidualWrapper(REBNCONV(out_ch, mid_ch, dirate=1, norm=norm), out_ch, mid_ch); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv4 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)

        self.rebnconv3d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv2d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv1d = ResidualWrapper(REBNCONV(mid_ch, out_ch, dirate=1, norm=norm), mid_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); d = self.pool1(h1)
        h2 = self.rebnconv2(d);     d = self.pool2(h2)
        h3 = self.rebnconv3(d)
        h4 = self.rebnconv4(h3)

        u3 = _upsample_like(self.rebnconv3d(h4), h2)
        u2 = _upsample_like(self.rebnconv2d(u3), h1)
        u1 = self.rebnconv1d(u2)
        return u1


class RSU4_NoSkip(nn.Module):
    """RSU-4 (4 단계 다운샘플), skip 제거 변형."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
        super().__init__()
        self.rebnconvin = ResidualWrapper(REBNCONV(in_ch, out_ch, dirate=1, norm=norm), in_ch, out_ch)
        self.rebnconv1 = ResidualWrapper(REBNCONV(out_ch, mid_ch, dirate=1, norm=norm), out_ch, mid_ch); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv3 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)

        self.rebnconv2d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=1, norm=norm), mid_ch, mid_ch)
        self.rebnconv1d = ResidualWrapper(REBNCONV(mid_ch, out_ch, dirate=1, norm=norm), mid_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); d = self.pool1(h1)
        h2 = self.rebnconv2(d)
        h3 = self.rebnconv3(h2)

        u2 = _upsample_like(self.rebnconv2d(h3), h1)
        u1 = self.rebnconv1d(u2)
        return u1


class RSU4F_NoSkip(nn.Module):
    """RSU-4F: dilation 을 늘려 downsampling 없이 receptive field 확대 (skip 제거)."""

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, norm='bn'):
        super().__init__()
        self.rebnconvin = ResidualWrapper(REBNCONV(in_ch, out_ch, dirate=1, norm=norm), in_ch, out_ch)
        self.rebnconv1 = ResidualWrapper(REBNCONV(out_ch, mid_ch, dirate=1, norm=norm), out_ch, mid_ch)
        self.rebnconv2 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)
        self.rebnconv3 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=4, norm=norm), mid_ch, mid_ch)
        self.rebnconv4 = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=8, norm=norm), mid_ch, mid_ch)

        self.rebnconv3d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=4, norm=norm), mid_ch, mid_ch)
        self.rebnconv2d = ResidualWrapper(REBNCONV(mid_ch, mid_ch, dirate=2, norm=norm), mid_ch, mid_ch)
        self.rebnconv1d = ResidualWrapper(REBNCONV(mid_ch, out_ch, dirate=1, norm=norm), mid_ch, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin)
        h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2)
        h4 = self.rebnconv4(h3)

        d3 = self.rebnconv3d(h4)
        d2 = self.rebnconv2d(d3)
        d1 = self.rebnconv1d(d2)
        return d1



# =========================================================
# U2NET / U2NETP (AE2Net style, NO skips anywhere)
# =========================================================
class U2NET(nn.Module):
    """AE 스타일 U^2-Net.

    - encoder ↔ decoder 사이 skip 과 RSU 내부 concat 모두 제거.
    - 각 decoder feature 에서 side output 을 뽑아 마지막에 concat → 1x1 conv 로 융합.
    - deep supervision 효과를 유지하면서 AE 처럼 재구성만 수행.
    (실험용. 현재 config.ae_type 기본은 AE_high.)
    """

    def __init__(self, in_ch=1, out_ch=1, norm='bn'):
        super().__init__()

        self.stage1 = RSU7_NoSkip(in_ch, 32, 64, norm=norm);  self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6_NoSkip(64, 32, 128, norm=norm);    self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5_NoSkip(128, 64, 256, norm=norm);   self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4_NoSkip(256, 128, 512, norm=norm);  self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F_NoSkip(512, 256, 512, norm=norm); self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F_NoSkip(512, 256, 512, norm=norm)

        # decoder (no top-level skips)
        self.stage5d = RSU4F_NoSkip(512, 256, 512, norm=norm)
        self.stage4d = RSU4_NoSkip(512, 128, 256, norm=norm)
        self.stage3d = RSU5_NoSkip(256, 64, 128, norm=norm)
        self.stage2d = RSU6_NoSkip(128, 32, 64, norm=norm)
        self.stage1d = RSU7_NoSkip(64, 16, 64, norm=norm)

        # side outputs from decoder features only
        self.side1 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        # encoder
        h1 = self.stage1(x);   d = self.pool12(h1)
        h2 = self.stage2(d);   d = self.pool23(h2)
        h3 = self.stage3(d);   d = self.pool34(h3)
        h4 = self.stage4(d);   d = self.pool45(h4)
        h5 = self.stage5(d);   d = self.pool56(h5)
        h6 = self.stage6(d)

        # decoder (no top-level concat)
        u5 = _upsample_like(self.stage5d(h6), h4)
        u4 = _upsample_like(self.stage4d(u5), h3)
        u3 = _upsample_like(self.stage3d(u4), h2)
        u2 = _upsample_like(self.stage2d(u3), h1)
        u1 = self.stage1d(u2)

        # side outputs (from decoder states only)
        d1 = self.side1(u1)
        d2 = _upsample_like(self.side2(u2), d1)
        d3 = _upsample_like(self.side3(u3), d1)
        d4 = _upsample_like(self.side4(u4), d1)
        d5 = _upsample_like(self.side5(u5), d1)
        d6 = _upsample_like(self.side6(h6), d1)  # bottleneck supervision

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        # If you plan to use BCEWithLogitsLoss, return raw d0 (no sigmoid).
        return torch.sigmoid(d0)  # <- change to `return d0` for BCEWithLogitsLoss


class U2NETP_AE(nn.Module):
    """경량 AE 스타일 U^2-Net (모든 stage 를 동일한 64 채널로 통일)."""

    def __init__(self, in_ch=1, out_ch=1, norm='bn'):
        super().__init__()
        self.stage1 = RSU7_NoSkip(in_ch, 16, 64, norm=norm);  self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6_NoSkip(64, 16, 64, norm=norm);     self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5_NoSkip(64, 16, 64, norm=norm);     self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4_NoSkip(64, 16, 64, norm=norm);     self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F_NoSkip(64, 16, 64, norm=norm);    self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F_NoSkip(64, 16, 64, norm=norm)

        self.stage5d = RSU4F_NoSkip(64, 16, 64, norm=norm)
        self.stage4d = RSU4_NoSkip(64, 16, 64, norm=norm)
        self.stage3d = RSU5_NoSkip(64, 16, 64, norm=norm)
        self.stage2d = RSU6_NoSkip(64, 16, 64, norm=norm)
        self.stage1d = RSU7_NoSkip(64, 16, 64, norm=norm)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        h1 = self.stage1(x);  d = self.pool12(h1)
        h2 = self.stage2(d);  d = self.pool23(h2)
        h3 = self.stage3(d);  d = self.pool34(h3)
        h4 = self.stage4(d);  d = self.pool45(h4)
        h5 = self.stage5(d);  d = self.pool56(h5)
        h6 = self.stage6(d)

        u5 = _upsample_like(self.stage5d(h6), h4)
        u4 = _upsample_like(self.stage4d(u5), h3)
        u3 = _upsample_like(self.stage3d(u4), h2)
        u2 = _upsample_like(self.stage2d(u3), h1)
        u1 = self.stage1d(u2)

        d1 = self.side1(u1)
        d2 = _upsample_like(self.side2(u2), d1)
        d3 = _upsample_like(self.side3(u3), d1)
        d4 = _upsample_like(self.side4(u4), d1)
        d5 = _upsample_like(self.side5(u5), d1)
        d6 = _upsample_like(self.side6(h6), d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
        return torch.sigmoid(d0)  # or return d0 for BCEWithLogitsLoss

class AE_ultra(nn.Module):
    """AE_high 에서 각 stage 마다 ResidualBlock 을 하나 더 쌓은 extra-deep 변형 (실험용)."""

    def __init__(self):
        super(AE_ultra, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            ResidualBlock(32, 32), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(512, 1024),
            ResidualBlock(1024, 1024), # Added depth
            nn.MaxPool2d(2, stride=2, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(1024, 512),
            ResidualBlock(512, 512), # Added depth
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256), # Added depth
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128), # Added depth

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64), # Added depth

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            ResidualBlock(64, 32),
            ResidualBlock(32, 32), # Added depth

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
