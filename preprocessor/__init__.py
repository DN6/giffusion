import io
import logging
import os
from typing import Any, Dict, Optional, Union

import torch
from controlnet_aux.processor import MODELS
from kornia.filters import canny
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from transformers import pipeline

MODEL_PATH = os.getenv("MODEL_PATH", "models")
LOGGER = logging.getLogger(__name__)
os.makedirs(MODEL_PATH, exist_ok=True)

depth_estimator = pipeline("depth-estimation", cache_dir=MODEL_PATH)


def apply_canny(image_tensor):
    _, transformed = canny(image_tensor)
    transformed = transformed.to("cpu")
    torch.cuda.empty_cache()

    output = torch.cat([transformed] * 3, dim=1)

    return output


def apply_depth_estimation(image):
    image = ToPILImage()(image[0])

    depth_map = depth_estimator(image)["depth"]
    depth_map = ToTensor()(depth_map).unsqueeze(0)

    output = torch.cat([depth_map] * 3, dim=1)
    output = output

    return output


def apply_preprocessing(image, preprocessor):
    fn = preprocessors.get(preprocessor)
    if not fn:
        return image

    output = fn(image)

    return output


class Processor:
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors
        Copied from:
        https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/processor.py

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODELS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

        self.resize = self.params.pop("resize", False)
        if self.resize:
            LOGGER.warning(
                f"Warning: {self.processor_id} will resize image to {self.resize}x{self.resize}"
            )

    def load_processor(self, processor_id: str) -> "Processor":
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet aux processor
        """
        processor = MODELS[processor_id]["class"]

        # check if the proecssor is a checkpoint model
        if MODELS[processor_id]["checkpoint"]:
            processor = processor.from_pretrained(
                "lllyasviel/Annotators", cache_dir=MODEL_PATH
            )
        else:
            processor = processor()
        return processor

    def __call__(
        self, image: Union[Image.Image, bytes], to_pil: bool = True
    ) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_pil (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        image_height, image_width = image.size
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        if self.resize:
            image = image.resize((self.resize, self.resize))

        processed_image = self.processor(image, **self.params)

        if to_pil:
            processed_image = processed_image.resize(
                (image_height, image_width), Image.LANCZOS
            )
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format="JPEG")
            return output_bytes.getvalue()


class Preprocessor:
    def __init__(self, processor_id) -> None:
        self.processor_id = processor_id
        if not MODELS.get(processor_id):
            self.processor = Processor(processor_id)
        else:
            self.processor = None

    def __call__(self, image) -> Any:
        if not self.processor:
            return image

        return self.processor(image)
