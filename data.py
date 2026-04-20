"""DataLoader 래퍼.

train/valid/test 각각에 대해 data_loader_black.CustomDRAEM* Dataset 을
래핑해서 torch DataLoader 객체를 반환한다.
"""
from torch.utils.data import DataLoader

import config
from data_loader_black import (
    CustomDRAEMTestDataset,
    CustomDRAEMTrainDataset,
    CustomDRAEMValidDataset,
)


# 프로젝트 외부에서는 Anomaly* 이름으로만 참조하도록 alias 제공.
AnomalyTrainDataset = CustomDRAEMTrainDataset
AnomalyValidDataset = CustomDRAEMValidDataset
AnomalyTestDataset = CustomDRAEMTestDataset


def get_train_dataloader(
    image_paths,
    anomaly_source_path=None,
    batch_size=None,
    image_size=None,
    num_workers=None,
):
    """학습용 DataLoader. shuffle=True, pin_memory=True."""
    dataset = AnomalyTrainDataset(
        image_paths,
        anomaly_source_path=anomaly_source_path or config.anomaly_source_path,
        resize_shape=list(image_size or config.image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or config.base_batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers is not None else config.num_workers,
        pin_memory=True,
    )


def get_valid_dataloader(image_paths, batch_size=None, image_size=None, num_workers=None):
    """검증용 DataLoader. 재현성 확보를 위해 shuffle=False."""
    dataset = AnomalyValidDataset(
        image_paths,
        resize_shape=list(image_size or config.image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or config.base_batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else config.num_workers,
        pin_memory=True,
    )


def get_test_dataloader(image_paths, batch_size=1, image_size=None, num_workers=0):
    """추론용 DataLoader. 이미지당 결과 시각화를 위해 batch_size=1 기본."""
    dataset = AnomalyTestDataset(
        image_paths,
        resize_shape=list(image_size or config.image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
