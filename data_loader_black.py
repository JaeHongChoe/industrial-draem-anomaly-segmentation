"""DRAEM Dataset 구현 모음.

- MVTecDRAEM* : MVTec 원본 코드 형태 (컬러 3채널, 원본 DRAEM 로직). 참고/실험용.
- CustomDRAEM* : 이 프로젝트에서 실제로 사용하는 grayscale(1채널) 버전.

CustomDRAEMTrainDataset 이 핵심으로, 실제 결함 patch (defect_patches/) 와
Perlin noise 기반 합성 결함을 각각 50% 확률로 섞어 '재구성 목표 이미지' 와
'증강된 입력 이미지' + 'anomaly mask' 3종을 반환한다.
"""
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class MVTecDRAEMTestDataset(Dataset):
    """MVTec 원본 DRAEM 테스트 Dataset (3채널 / good·bad 하위 폴더 구조)."""

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):
    """MVTec 원본 DRAEM 학습 Dataset. Perlin noise + 외부 anomaly 이미지로 결함 합성.

    현재 파이프라인에서는 CustomDRAEMTrainDataset 을 사용하므로 참고용이며,
    원본 DRAEM 과의 호환을 위해 남겨둔다.
    """

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path, cv2.IMREAD_GRAYSCALE)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_source_img = np.expand_dims(anomaly_source_img, axis=2)

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.expand_dims(image, axis=2)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class CustomDRAEMTestDataset(Dataset):
    """프로젝트 전용 grayscale 테스트 Dataset.

    root_dir 에 와일드카드(*) 가 있으면 직접 glob, 아니면 두 단계 (category/file)
    구조를 우선 시도하고 없으면 플랫 구조로 fallback.
    """

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        if '*' in root_dir:
            self.images = sorted(glob.glob(root_dir))
        else:
            # 우선 root/category/file 구조 시도, 비어 있으면 플랫 구조로 fallback.
            self.images = sorted(glob.glob(root_dir+"/*/*.png") + glob.glob(root_dir+"/*/*.jpg"))
            if len(self.images) == 0:
                 self.images = sorted(glob.glob(root_dir+"/*.png") + glob.glob(root_dir+"/*.jpg"))

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
            
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 1)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            
            if not os.path.exists(mask_path):
                 mask_path = img_path.replace("test", "ground_truth").replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            
            image, mask = self.transform_image(img_path, mask_path)
            
            if np.sum(mask) == 0:
                mask = np.ones_like(mask)
                
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

class CustomDRAEMTrainDataset(Dataset):
    """프로젝트 전용 grayscale 학습 Dataset.

    핵심 동작:
      1) root_dir 에서 정상 이미지를 수집.
      2) anomaly_source_path 또는 defect_patches/ 하위에서 실제 결함 patch (alpha 채널) 수집.
      3) __getitem__ 에서 50% 는 실제 결함 patch 를 블렌딩, 50% 는 Perlin noise 기반 합성.
      4) 'image' (원본), 'augmented_image' (결함 합성), 'anomaly_mask' 반환.
    """

    def __init__(self, root_dir, anomaly_source_path=None, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            anomaly_source_path: Path to defect_patches folder (optional, auto-detected if None)
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        if '*' in root_dir:
            self.image_paths = sorted(glob.glob(root_dir))
        else:
            self.image_paths = sorted(glob.glob(root_dir+"/*.png") + glob.glob(root_dir+"/*.jpg"))

        # 결함 patch 수집: anomaly_source_path 가 주어지면 우선 시도,
        # 실패 시 defect_patches/ 표준 위치를 자동 탐색.
        self.defect_patches = []
        patch_patterns = []

        if anomaly_source_path:
            if '*' in anomaly_source_path:
                patch_patterns.append(anomaly_source_path)
            else:
                patch_patterns.extend([
                    os.path.join(anomaly_source_path, '*.png'),
                    os.path.join(anomaly_source_path, '*', '*.png'),
                ])

        # 자동 탐색 후보 경로 (실행 디렉토리 기준 상대 경로).
        patch_dirs = [
            '../defect_patches/crack',
            '../defect_patches/bow',
            './defect_patches/crack',
            './defect_patches/bow',
        ]

        for patch_pattern in patch_patterns:
            self.defect_patches.extend(glob.glob(patch_pattern))

        # 명시 경로로 못 찾았으면 표준 위치에서 자동 수집.
        if not self.defect_patches:
            for patch_dir in patch_dirs:
                if os.path.exists(patch_dir):
                    patches = glob.glob(os.path.join(patch_dir, '*.png'))
                    self.defect_patches.extend(patches)
        
        print(f"[CustomDRAEMTrainDataset] Loaded {len(self.defect_patches)} real defect patches")
        
        self.augmenters = [
            iaa.Invert(),
            iaa.Affine(rotate=(-45, 45))
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def getRandomZoomPositionAugmenter(self):
        """줌아웃 + 랜덤 이동 + 약한 elastic 뒤틀림 조합 augmenter 생성.

        - scale 0.7~0.9 (70~90% 축소)로 축소 후 남는 여백 내에서 랜덤 이동.
          이렇게 하면 이미지가 경계를 넘어 잘리지 않는다.
        - 회전과 elastic 은 각각 30% 확률로 독립 적용.
        현재는 transform_image 에서 호출하지 않고 실험용으로 보존.
        """
        scale = np.random.uniform(0.7, 0.9)

        # scale=0.8 이면 margin=0.2 (양쪽 10%씩 여백).
        margin = 1 - scale

        # margin 절반 범위 안에서만 이동해 경계 초과 방지.
        tx = np.random.uniform(-margin/2, margin/2)
        ty = np.random.uniform(-margin/2, margin/2)

        rotation = np.random.uniform(-90, 90)

        return iaa.Sequential([
            # 30% 확률로 줌/회전/이동 통합 Affine.
            iaa.Sometimes(0.3, iaa.Affine(
                scale=scale,
                rotate=rotation,
                translate_percent={"x": tx, "y": ty},
                mode='constant',
                cval=0  # 빈 영역은 0 (검은색)으로 채움.
            )),

            # 30% 확률로 약한 elastic 뒤틀림.
            iaa.Sometimes(0.3, iaa.ElasticTransformation(
                alpha=(3, 8),     # 변형 강도 (아주 약하게)
                sigma=(4, 6),     # 부드러움
                mode='constant',
                cval=0
            ))
        ])

    def randAugmenter(self):
        """self.augmenters 중 최대 3개를 무작위로 뽑아 Sequential 로 묶음."""
        n_aug = min(3, len(self.augmenters))
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), n_aug, replace=False)
        aug_list = [self.augmenters[aug_ind[i]] for i in range(n_aug)]
        aug = iaa.Sequential(aug_list)
        return aug

    def load_random_patch(self):
        """defect_patches 중 하나를 랜덤 선택해 RGBA 로 로드.

        - 한글 경로 대응을 위해 np.fromfile + cv2.imdecode 사용.
        - 4채널(RGBA) 이 아니면 mask 정보가 없으므로 None 반환.
        - RGB 평균을 grayscale patch 로, alpha 를 mask 로 분리한다.
        """
        if not self.defect_patches:
            return None, None

        patch_path = np.random.choice(self.defect_patches)
        # 한글/유니코드 경로 안전하게 로드.
        patch = cv2.imdecode(np.fromfile(patch_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if patch is None or patch.shape[2] < 4:
            return None, None

        # RGB → grayscale, alpha → mask 로 분리.
        patch_img = patch[:, :, :3].mean(axis=2)  # Grayscale
        patch_mask = patch[:, :, 3]  # Alpha 채널 (결함 영역)

        return patch_img, patch_mask

    def augment_patch(self, patch_img, patch_mask, target_h, target_w):
        """patch 에 다양한 기하/광학 변환을 적용해 다양성을 확보.

        적용 순서: scale → rotation → flip → elastic(30%) → perspective(20%)
                 → brightness → contrast
        이렇게 하면 동일 patch 여도 매번 다른 위치·형태·밝기로 등장한다.
        """
        if patch_img is None:
            return None, None

        h, w = patch_img.shape[:2]

        # 스케일 랜덤 (0.5x - 2.5x): 작은 결함 / 큰 결함 모두 학습.
        scale = np.random.uniform(0.5, 2.5)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 너무 작아지면 resize 가 깨지므로 최소 5px 보장.
        if new_w < 5 or new_h < 5:
            new_w, new_h = max(5, new_w), max(5, new_h)

        patch_img = cv2.resize(patch_img.astype(np.float32), (new_w, new_h))
        patch_mask = cv2.resize(patch_mask.astype(np.float32), (new_w, new_h))

        # 0~360도 랜덤 회전.
        angle = np.random.uniform(0, 360)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 회전 후 전체가 담기도록 새 bounding box 계산 후 이동 보정.
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2

        patch_img = cv2.warpAffine(patch_img, M, (rot_w, rot_h))
        patch_mask = cv2.warpAffine(patch_mask, M, (rot_w, rot_h))

        # 좌우/상하 50% 확률 flip.
        if np.random.rand() > 0.5:
            patch_img = np.fliplr(patch_img)
            patch_mask = np.fliplr(patch_mask)
        if np.random.rand() > 0.5:
            patch_img = np.flipud(patch_img)
            patch_mask = np.flipud(patch_mask)

        # === Elastic Transform (30% 확률): 결함 윤곽 자연스럽게 뒤틀기 ===
        if np.random.rand() < 0.3:
            h_e, w_e = patch_img.shape[:2]
            if h_e > 10 and w_e > 10:
                alpha = np.random.uniform(3, 8)  # 변위 강도 (약하게)
                sigma = np.random.uniform(3, 5)  # Gaussian smoothing 정도

                # -1~1 균등노이즈 → Gaussian blur → alpha 배율 = 부드러운 변위 필드.
                dx = cv2.GaussianBlur((np.random.rand(h_e, w_e) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
                dy = cv2.GaussianBlur((np.random.rand(h_e, w_e) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha

                x, y = np.meshgrid(np.arange(w_e), np.arange(h_e))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)

                patch_img = cv2.remap(patch_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                patch_mask = cv2.remap(patch_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # === Perspective Warp (20% 확률, 약하게): 촬영 각도 변화 시뮬레이션 ===
        if np.random.rand() < 0.2:
            h_p, w_p = patch_img.shape[:2]
            if h_p > 20 and w_p > 20:
                # 각 모서리를 최대 10% 범위 안에서 랜덤 이동.
                margin = 0.1
                pts1 = np.float32([[0, 0], [w_p, 0], [0, h_p], [w_p, h_p]])
                pts2 = np.float32([
                    [np.random.uniform(0, w_p * margin), np.random.uniform(0, h_p * margin)],
                    [w_p - np.random.uniform(0, w_p * margin), np.random.uniform(0, h_p * margin)],
                    [np.random.uniform(0, w_p * margin), h_p - np.random.uniform(0, h_p * margin)],
                    [w_p - np.random.uniform(0, w_p * margin), h_p - np.random.uniform(0, h_p * margin)]
                ])
                M_persp = cv2.getPerspectiveTransform(pts1, pts2)
                patch_img = cv2.warpPerspective(patch_img, M_persp, (w_p, h_p), borderMode=cv2.BORDER_REFLECT)
                patch_mask = cv2.warpPerspective(patch_mask, M_persp, (w_p, h_p), borderMode=cv2.BORDER_REFLECT)

        # 밝기 랜덤 조정 (±30%).
        brightness = np.random.uniform(0.7, 1.3)
        patch_img = np.clip(patch_img * brightness, 0, 255)

        # 대비 랜덤 조정 (평균 기준으로 분산 확대/축소).
        contrast = np.random.uniform(0.8, 1.2)
        mean_val = np.mean(patch_img)
        patch_img = np.clip((patch_img - mean_val) * contrast + mean_val, 0, 255)

        return patch_img, patch_mask

    def place_patch_on_image(self, image, patch_img, patch_mask):
        """patch 를 원본 이미지의 중앙 원형 영역(반지름 90%) 내부에 랜덤 배치.

        - X-ray 등 원형 촬영이라 경계부는 유효하지 않으므로 중앙 90% 원 안에서만 배치.
        - 최대 20회 시도해 화면 내부에 완전히 들어가는 위치 탐색, 실패 시 중앙 fallback.
        - 블렌딩은 50% soft-edge (Gaussian blur mask), 50% 원본 DRAEM 수식 을 섞음.
        """
        if patch_img is None or patch_mask is None:
            return image, np.zeros_like(image)

        img_h, img_w = image.shape[:2]
        patch_h, patch_w = patch_img.shape[:2]

        # 이미지 중앙 기준 90% 반지름 원 (유효 촬영 영역).
        center_x, center_y = img_w // 2, img_h // 2
        max_radius = 0.9 * min(center_x, center_y)

        # 원 내부에서 랜덤 위치 추첨 (각도, 거리).
        for _ in range(20):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, max_radius - max(patch_w, patch_h) // 2)

            cx = int(center_x + dist * np.cos(angle))
            cy = int(center_y + dist * np.sin(angle))

            # patch top-left 좌표.
            x1 = cx - patch_w // 2
            y1 = cy - patch_h // 2
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            # 이미지 경계 안에 완전히 들어오면 확정.
            if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h:
                break
        else:
            # 20회 모두 실패 → 중앙에 두고 넘치는 부분은 crop.
            x1 = max(0, center_x - patch_w // 2)
            y1 = max(0, center_y - patch_h // 2)
            x2 = min(img_w, x1 + patch_w)
            y2 = min(img_h, y1 + patch_h)
            patch_img = patch_img[:y2-y1, :x2-x1]
            patch_mask = patch_mask[:y2-y1, :x2-x1]

        # 결과용 mask / image 준비.
        full_mask = np.zeros((img_h, img_w), dtype=np.float32)
        result = image.copy()

        # mask 를 0~1 로 정규화.
        norm_mask = patch_mask.astype(np.float32) / 255.0

        # blending 영역 추출.
        roi = result[y1:y2, x1:x2]
        patch_region = patch_img[:y2-y1, :x2-x1] / 255.0
        mask_region = norm_mask[:y2-y1, :x2-x1]

        # 50% Soft Edge (Gaussian blur), 50% 원본 DRAEM 방식.
        use_soft_edge = np.random.rand() < 0.5

        if use_soft_edge:
            # patch 최소변의 10% 정도로 blur kernel 크기 결정 (홀수로 보정).
            blur_size = max(5, int(min(patch_img.shape[:2]) * 0.1))
            if blur_size % 2 == 0:
                blur_size += 1

            # mask 가장자리를 흐리게 → 경계선이 자연스러워짐.
            mask_blurred = cv2.GaussianBlur(mask_region, (blur_size, blur_size), 0)

            # 어두워지는 효과 + soft edge 로 블렌드.
            beta = np.random.uniform(0.6, 1.0)
            blended = roi * (1 - mask_blurred * beta) + patch_region * mask_blurred * beta
            result[y1:y2, x1:x2] = np.clip(blended, 0, 1)
        else:
            # 원본 DRAEM 합성: mask 영역 = (1-beta)*patch + beta*original.
            beta = np.random.uniform(0.0, 0.8)
            blended = roi * (1 - mask_region) + ((1 - beta) * patch_region + beta * roi) * mask_region
            result[y1:y2, x1:x2] = np.clip(blended, 0, 1)

        full_mask[y1:y2, x1:x2] = mask_region

        return result, full_mask

    def augment_image(self, image):
        """결함 합성: 50% 실제 patch, 50% Perlin noise.

        반환: (augmented_image, anomaly_mask, has_anomaly)
        추가로 최종 50% 확률로 '결함 없음' 샘플을 만들어 학습 분포 균형을 맞춘다.
        """
        perlin_scale = 6
        min_perlin_scale = 0

        # 사용 가능한 실제 patch 가 있고 동전던지기 통과 시 real patch 모드.
        use_real_patch = len(self.defect_patches) > 0 and np.random.rand() < 0.5

        if use_real_patch:
            # --- Real Patch 모드: 실제 결함 patch 1~3개를 중첩 배치 ---
            num_patches = np.random.randint(1, 4)

            result_image = image.copy()
            combined_mask = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)

            for _ in range(num_patches):
                patch_img, patch_mask = self.load_random_patch()
                if patch_img is not None:
                    # patch 별로 증강 → 랜덤 배치 → mask 를 OR 로 합침.
                    patch_img, patch_mask = self.augment_patch(
                        patch_img, patch_mask,
                        self.resize_shape[0], self.resize_shape[1]
                    )
                    result_image, single_mask = self.place_patch_on_image(
                        result_image[:, :, 0] if len(result_image.shape) == 3 else result_image,
                        patch_img, patch_mask
                    )
                    if len(result_image.shape) == 2:
                        result_image = np.expand_dims(result_image, axis=2)
                    combined_mask = np.maximum(combined_mask, single_mask)

            perlin_thr = np.expand_dims(combined_mask, axis=2)
            augmented_image = result_image

        else:
            # --- Perlin Noise 모드: 원본 DRAEM 스타일 3종 (crack / circle / blob) ---
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)

            rand_val = torch.rand(1).numpy()[0]

            if rand_val < 0.5:
                # Crack: noise 절댓값이 작은 영역 = 능선/골 → 가는 선 모양 결함.
                thickness = np.random.uniform(0.02, 0.08)
                perlin_thr = np.where(np.abs(perlin_noise) < thickness, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            elif rand_val < 0.75:
                # 기하학적 원 결함 (반점 시뮬레이션).
                perlin_thr = np.zeros_like(perlin_noise)
                rows, cols = self.resize_shape[0], self.resize_shape[1]
                center_x, center_y = cols // 2, rows // 2
                max_radius = 0.9 * min(center_x, center_y)

                # 최대 10회 시도해 유효 영역 안에 원 그리기.
                for _ in range(10):
                    cx = np.random.randint(0, cols)
                    cy = np.random.randint(0, rows)
                    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    if dist < max_radius:
                        circle_radius = np.random.randint(5, 50)
                        cv2.circle(perlin_thr, (cx, cy), circle_radius, 1.0, -1)
                        break
            else:
                # Blob: noise 임계값 초과 영역 = 불규칙 덩어리.
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

            # 원형 유효 영역 밖은 0 으로 마스킹 (X-ray 경계부 제외).
            rows, cols = self.resize_shape[0], self.resize_shape[1]
            center_x, center_y = cols // 2, rows // 2
            radius = 0.9 * min(center_x, center_y)

            Y, X = np.ogrid[:rows, :cols]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            circular_mask = dist_from_center <= radius

            perlin_thr = perlin_thr * circular_mask
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            # Perlin 모드에서는 결함 영역을 검게 칠함 (img_thr=0).
            img_thr = np.zeros_like(perlin_thr, dtype=np.float32)
            beta = torch.rand(1).numpy()[0] * 0.8

            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        # 50% 확률로 '결함 없음' 샘플 생성 → 모델이 정상/결함 균형 학습.
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr.astype(np.float32)
            # mask 영역만 augmented_image 로 교체, 나머지는 원본 유지.
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            # mask 가 전부 0 이면 결함 없음으로 재분류.
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path):
        """이미지 로드 → resize → augment_image → CHW 텐서 포맷 변환."""
        # 한글/유니코드 경로 대응.
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # 로드 실패 시 빈 이미지로 대체 (학습 중단 방지).
            image = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.expand_dims(image, axis=2)

        # 회전 증강은 전처리(a4/a5) 단계에서 이미 적용되어 여기선 비활성화.
        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        # Zoom/Elastic 증강도 비활성화 (품질 저하 이슈로 일시 off).
        # zoom_aug = self.getRandomZoomPositionAugmenter()
        # image = zoom_aug(image=image)

        # 0~1 float 정규화 및 단일 채널로 reshape.
        image = np.array(image).reshape((image.shape[0], image.shape[1], 1)).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image)
        # HWC → CHW (PyTorch 형식).
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class CustomDRAEMValidDataset(Dataset):
    """Validation Dataset.

    증강 없이 이미지 로드 → resize → 정규화만 수행.
    학습 중 매 epoch 의 validation SSIM loss 계산 및 ReduceLROnPlateau 스케줄러
    업데이트에 사용된다.
    """

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        # root_dir 에 와일드카드(*) 포함 시 그대로 glob, 아니면 하위 png/jpg 수집.
        if '*' in root_dir:
            self.image_paths = sorted(glob.glob(root_dir))
        else:
            self.image_paths = sorted(glob.glob(root_dir+"/*.png") + glob.glob(root_dir+"/*.jpg"))

        print(f"[CustomDRAEMValidDataset] Loaded {len(self.image_paths)} validation images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 한글/유니코드 경로 대응.
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8)

        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = image.astype(np.float32) / 255.0

        # (H, W) → (1, H, W): 채널 축 추가 후 CHW 형식 반환.
        image = np.expand_dims(image, axis=0)

        return {'image': image, 'idx': idx}
