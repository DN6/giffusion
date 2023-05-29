import os

import torch
from kornia.filters import canny
from torchvision.transforms import ToPILImage, ToTensor
from transformers import pipeline

MODEL_PATH = os.getenv("MODEL_PATH", "models")
os.makedirs(MODEL_PATH, exist_ok=True)

depth_estimator = pipeline("depth-estimation", cache_dir=MODEL_PATH)


def apply_canny(image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.float().to(device)

    _, transformed = canny(image_tensor)
    transformed = transformed.to("cpu")
    torch.cuda.empty_cache()

    output = torch.cat([transformed] * 3, dim=1)

    return output


def apply_depth_estimation(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = ToPILImage()(image[0])

    depth_estimator.device = device
    depth_estimator.model.to(device)

    depth_map = depth_estimator(image)["depth"]
    depth_map = ToTensor()(depth_map).unsqueeze(0)

    depth_estimator.device = "cpu"
    depth_estimator.model.to("cpu")
    torch.cuda.empty_cache()

    output = torch.cat([depth_map] * 3, dim=1)
    output = output

    return output


def apply_preprocessing(image, preprocessor):
    fn = preprocessors.get(preprocessor)
    if not fn:
        return image

    output = fn(image)

    return output


preprocessors = {"canny": apply_canny, "depth": apply_depth_estimation}
