import numpy as np
import torch
from kornia.filters import canny
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


def apply_canny(image):
    _, transformed = canny(image)
    output = torch.cat([transformed] * 3, dim=1)

    return output


def apply_preprocessing(image, preprocessor):
    fn = preprocessors.get(preprocessor)
    if not fn:
        return image

    output = fn(image)

    return output


preprocessors = {"canny": apply_canny}
