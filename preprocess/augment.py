"""PIL/NumPy 기반 공용 증강 유틸.

a1~a5 스크립트가 각자 갖고 있던 증강 코드를 한곳에 모아둔 모듈이다.
apply_augmentation(image, mode) 한 함수로 호출하며, mode 값에 따라
'가벼운(basic)' 조합과 '전체(full)' 조합을 선택한다.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def add_gaussian_noise(img: Image.Image, mean: int = 0, std: int = 10) -> Image.Image:
    """가우시안 노이즈 주입. 노이즈 강건성을 위한 학습 augmentation."""
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def elastic_transform(img: Image.Image, alpha: int = 5, sigma: int = 10) -> Image.Image:
    """Elastic 변형. 결함 모양이 약간 뒤틀려도 일반화되도록 좌표 필드를 흔든다."""
    img_np = np.array(img)
    shape = img_np.shape
    # 난수 필드를 가우시안 블러로 부드럽게 만들어야 자연스러운 왜곡이 나옴.
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(
        img_np, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return Image.fromarray(distorted)


def adjust_contrast(img: Image.Image, factor: float | None = None) -> Image.Image:
    """대비 조정. factor 가 None 이면 0.7~1.5 범위 랜덤."""
    if factor is None:
        factor = random.uniform(0.7, 1.5)
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_blur(img: Image.Image) -> Image.Image:
    """가벼운 5x5 가우시안 블러. 포커스 미세 편차에 강건해지도록 한다."""
    np_img = np.array(img)
    blurred = cv2.GaussianBlur(np_img, (5, 5), 0)
    return Image.fromarray(blurred)


def random_erasing(img: Image.Image, erase_size: tuple = (30, 50)) -> Image.Image:
    """임의 사각 영역을 흰색으로 가리는 Random Erasing. Occlusion 시나리오 대비."""
    np_img = np.array(img)
    h, w = np_img.shape[:2]
    erase_w = random.randint(*erase_size)
    erase_h = random.randint(*erase_size)
    x = random.randint(0, max(0, w - erase_w))
    y = random.randint(0, max(0, h - erase_h))
    np_img[y:y + erase_h, x:x + erase_w] = 255
    return Image.fromarray(np_img)


def apply_clahe(img: Image.Image) -> Image.Image:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 로 국소 대비 강화."""
    np_img = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(np_img))


def adjust_brightness(img: Image.Image) -> Image.Image:
    """밝기 ±30% 범위 랜덤 조정."""
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_augmentation(image, mode: str = "full") -> Image.Image:
    """확률 기반 증강 파이프라인.

    mode:
      - "basic": flip, rotate, brightness, contrast 까지만 (a4/a5 스타일, 과도한 변형 방지).
      - "full":  blur, noise, elastic, clahe, erasing 까지 전부 (a1~a3 스타일).

    입력은 PIL.Image 또는 numpy.ndarray 모두 허용. 내부적으로 grayscale(L) 로 정규화.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("L")

    # 기하 변환: flip 2종 + ±180° 랜덤 회전.
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.randint(-180, 180)
    image = image.rotate(angle)

    # 밝기/대비는 basic 에도 포함 (과하지 않게 ±30% 수준).
    if random.random() < 0.5:
        image = adjust_brightness(image)
    if random.random() < 0.5:
        image = adjust_contrast(image)

    # full 모드에서만 추가되는 강한 변형들.
    if mode != "basic":
        if random.random() < 0.3:
            image = apply_blur(image)
        if random.random() < 0.3:
            image = add_gaussian_noise(image)
        if random.random() < 0.3:
            image = elastic_transform(image)
        if random.random() < 0.3:
            image = apply_clahe(image)
        if random.random() < 0.3:
            image = random_erasing(image)

    return image
