import io
import logging
import os
from typing import Any, Dict, Optional, Union

from controlnet_aux.processor import MODEL_PARAMS, MODELS
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "models")
LOGGER = logging.getLogger(__name__)
os.makedirs(MODEL_PATH, exist_ok=True)


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
        self.params = MODEL_PARAMS[self.processor_id]
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
            pretrained_model_path = "lllyasviel/Annotators"
            processor = processor.from_pretrained(
                pretrained_model_path, cache_dir=MODEL_PATH
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
    def __init__(self, processor_ids) -> None:
        self.processor_ids = processor_ids
        self.processors = [
            self.load_processor(processor_id) for processor_id in self.processor_ids
        ]

    def load_processor(self, processor_id):
        if not MODELS.get(processor_id):
            return None
        else:
            return Processor(processor_id)

    def __call__(self, images) -> Any:
        outputs = []

        if not isinstance(images, list):
            images = [images]

        for processor in self.processors:
            for image in images:
                if processor is None:
                    outputs.append(image)
                else:
                    outputs.append(processor(image))

        if not outputs:
            return images

        return outputs
