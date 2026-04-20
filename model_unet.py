"""DRAEM 의 U-Net 서브네트워크 정의.

- ReconstructiveSubNetwork: 원본 DRAEM 논문의 재구성용 U-Net (현재 프로젝트에선
  model.py 의 AE_high 를 대신 사용하므로 이 파일 버전은 참고/fallback 용).
- DiscriminativeSubNetwork: anomaly segmentation U-Net. 학습시 "원본 + 재구성" 2채널을
  concat 해서 입력, 출력은 2채널 logit 으로 softmax → anomaly 확률 map.

EncoderDiscriminative 는 6-stage (b1~b6), DecoderDiscriminative 는 대칭 5-stage
스킵 연결을 사용하는 표준 U-Net 구조.
"""
import torch
import torch.nn as nn


class ReconstructiveSubNetwork(nn.Module):
    """DRAEM 재구성 U-Net (Encoder - Decoder). out_channels 은 입력과 동일해야 한다."""

    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        b5 = self.encoder(x)
        output = self.decoder(b5)
        return output


class DiscriminativeSubNetwork(nn.Module):
    """DRAEM segmentation 브랜치.

    Args:
        in_channels: 보통 2 (원본 1 + 재구성 1 을 concat).
        out_channels: 2 (정상 / 결함 logit).
        base_channels: encoder/decoder 채널 폭 (기본 64).
        out_features: True 면 중간 feature (b2..b6) 도 함께 반환.
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        # segment_act 는 외부에서 softmax 를 적용하므로 필요 없음.
        self.out_features = out_features

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment


class EncoderDiscriminative(nn.Module):
    """6단계 다운샘플 encoder. 각 block = Conv-BN-ReLU x2, 그 뒤 MaxPool."""

    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1,b2,b3,b4,b5,b6

class DecoderDiscriminative(nn.Module):
    """대칭 U-Net decoder. 각 단계에서 bilinear upsample → encoder skip 과 concat → Conv 블록."""

    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )



        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1,b2,b3,b4,b5,b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b,b5),dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1,b4),dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2,b3),dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3,b2),dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4,b1),dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out



class EncoderReconstructive(nn.Module):
    """5단계 다운샘플 encoder. b5 bottleneck 만 반환 (skip 연결 없음)."""

    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5


class DecoderReconstructive(nn.Module):
    """4단계 upsample decoder. bottleneck 에서 입력 해상도까지 복원."""

    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out
