import numpy as np
import torch
from kornia.filters import canny
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


def apply_canny(image):
    _, transformed = canny(image)
    transformed /= 255.0

    return transformed


def apply_inpainting_mask(image):
    image /= 255.0
    output = torch.where(image < 0.0, -1.0, image)

    return output


def apply_preprocessing(image, preprocessor):
    fn = preprocessors.get(preprocessor)
    if not fn:
        return image

    output = fn(image)

    return output


preprocessors = {"canny": apply_canny, "inpainting": apply_inpainting_mask}
