import torch
from PIL import Image


class BaseFlow:
    def __init__(self, pipe, device, batch_size=1):
        self.pipe = pipe
        self.device = device
        self.batch_size = batch_size

    def postprocess(self, image_tensors):
        image_tensors = (image_tensors / 2 + 0.5).clamp(0, 1)
        image_tensors = (image_tensors * 255).to(torch.uint8)
        image_tensors = image_tensors.permute(0, 2, 3, 1)

        image_arrays = image_tensors.cpu().numpy()

        return image_arrays

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @torch.no_grad()
    def denoise(self):
        raise NotImplementedError

    @torch.no_grad()
    def diffuse(self):
        raise NotImplementedError

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        sample = self.pipe.vae.decode(latents).sample

        return sample

    @torch.no_grad()
    def encode_latents(self, x):
        init_latent_dist = self.pipe.vae.encode(x.to(self.device)).latent_dist
        latent = 0.18215 * init_latent_dist.sample()
        return latent

    @torch.no_grad()
    def prompt_to_embedding(self, prompt):
        if "|" in prompt:
            prompt = [x.strip() for x in prompt.split("|")]

        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids.to(self.pipe.text_encoder.device)
        text_embeddings = self.pipe.text_encoder(text_inputs)[0]

        return text_embeddings

    def pad_embedding(self, start, end):
        if start.shape == start.shape:
            return start, end

        smaller = min(
            [start, end],
            key=lambda key: key.shape[0],
        )
        larger = max(
            [start, end],
            key=lambda key: key.shape[0],
        )
        diff = larger.shape[0] - smaller.shape[0]

        padding = torch.cat([self.prompt_to_embedding(self.pipe, "")] * diff)
        if start.shape[0] < end.shape[0]:
            start = torch.cat([start, padding])
        else:
            end = torch.cat([end, padding])

        return start, end

    def safety_check(self, images_array):
        images = self.numpy_to_pil(images_array)

        safety_checker_input = self.pipe.feature_extractor(
            images, return_tensors="pt"
        ).to(self.device)

        output_images, has_nsfw_concept = self.pipe.safety_checker(
            images=images_array, clip_input=safety_checker_input.pixel_values
        )

        return output_images, has_nsfw_concept
