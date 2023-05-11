import torch
from kornia.filters import canny
from torchvision.transforms import ToPILImage, ToTensor
from transformers import pipeline

depth_estimator = pipeline("depth-estimation")


def apply_canny(image_tensor):
    _, transformed = canny(image_tensor)
    output = torch.cat([transformed] * 3, dim=1)

    return output


def apply_depth_estimation(image):
    image = ToPILImage()(image)
    depth_map = depth_estimator(image)["depth"]
    output = torch.cat([depth_map] * 3, dim=1)

    return output


def apply_preprocessing(image, preprocessor):
    fn = preprocessors.get(preprocessor)
    if not fn:
        return image

    output = fn(image)

    return output


preprocessors = {"canny": apply_canny, "depth": apply_depth_estimation}
