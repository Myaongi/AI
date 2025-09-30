# preprocessor.py
# 480 리사이즈 + Normalize(학습 때 mean, std 변경시 함께 변경해주기)
from __future__ import annotations
from typing import Tuple, Optional
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# 학습 때 쓴 값 그대로 (ImageNet)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ToTensorNoNumpy(torch.nn.Module):
    """NumPy 없이 PIL→Tensor [C,H,W], float32, [0,1]."""
    def __call__(self, img: Image.Image) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            raise TypeError("ToTensorNoNumpy expects a PIL.Image")
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        c = 3
        buf = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        t = buf.view(h, w, c).permute(2, 0, 1).contiguous().float().div(255.0)
        return t


_to_tensor_safe = ToTensorNoNumpy()


def _resize_square(img: Image.Image, size: int = 480,
                   interpolation: InterpolationMode = InterpolationMode.BICUBIC) -> Image.Image:
    """정사각 입력을 지정 크기로 리사이즈. (detector가 이미 정사각 pad=0.16 적용함)"""
    if not isinstance(img, Image.Image):
        raise TypeError("_resize_square expects a PIL.Image")
    # torchvision 버전별 antialias 인자 호환
    try:
        return TF.resize(img, [size, size], interpolation=interpolation, antialias=True)
    except TypeError:
        return TF.resize(img, [size, size], interpolation=interpolation)


def preprocess(
    crop_img: Image.Image,
    size: int = 480,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float]  = IMAGENET_STD,
    device: Optional[str] = None,           # "cuda:0" | "cpu" | None(이동 안 함)
    channels_last: bool = False,            # True면 NHWC 메모리 포맷로 이동
    add_batch: bool = True,                 # 배치 차원 추가(N, C, H, W)
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    detector.detect_and_crop()가 반환한 정사각 PIL 이미지를
    1) 480 리사이즈 2) ToTensor 3) Normalize까지 수행해 분류기 입력 텐서로 변환.

    반환: Tensor [N,C,H,W] 또는 [C,H,W] (add_batch=False일 때)
    """
    # 1) 리사이즈 (정사각 보장 가정)
    img_resized = _resize_square(crop_img, size=size, interpolation=InterpolationMode.BICUBIC)

    # 2) ToTensor (NumPy 없이)
    x = _to_tensor_safe(img_resized)  # [C,H,W], float32, [0,1]

    # 3) Normalize
    x = TF.normalize(x, mean=mean, std=std)

    # 4) 배치/디바이스/메모리 포맷
    if add_batch:
        x = x.unsqueeze(0)  # [1,C,H,W]
    if device is not None:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
    if dtype is not torch.float32:
        x = x.to(dtype)

    return x


__all__ = ["preprocess", "IMAGENET_MEAN", "IMAGENET_STD"]
