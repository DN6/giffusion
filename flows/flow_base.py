import inspect

import torch
import torchvision.transforms as T
from PIL import Image

to_pil = T.ToPILImage("RGB")
to_tensor = T.ToTensor()


class BaseFlow:
    def __init__(self, pipe, device, batch_size=1):
        self.pipe = pipe
        self.device = device
        self.batch_size = batch_size

    def preprocess(self, image, image_size=(512, 512)):
        image = to_pil(image)
        image = image.resize(image_size, resample=Image.LANCZOS)
        image = to_tensor(image)
        return 2.0 * image - 1.0

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
    def encode_latents(self, x, generator=None):
        init_latent_dist = self.pipe.vae.encode(x.to(self.device)).latent_dist
        latent = 0.18215 * init_latent_dist.sample(generator=generator)
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

    @torch.no_grad()
    def denoise(self, latents, text_embeddings, i, t, guidance_scale):
        accepts_eta = "eta" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

        latent_model_input = torch.cat(
            list(
                map(
                    lambda latent, text_embedding: torch.cat(
                        [latent] * text_embedding.shape[0]
                    ),
                    latents.chunk(self.batch_size),
                    text_embeddings.chunk(self.batch_size),
                )
            )
        )

        max_length = text_embeddings.shape[1]
        uncond_input = self.pipe.tokenizer(
            [""] * latent_model_input.shape[0],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        latent_model_input = torch.cat([latents] * 2)

        noise_pred = self.pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        )["sample"]

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_cond = torch.cat(
            list(
                map(
                    lambda x: x.mean(dim=0, keepdim=True),
                    noise_pred_cond.chunk(self.batch_size),
                )
            )
        )
        noise_pred_uncond = torch.cat(
            list(
                map(
                    lambda x: x.mean(dim=0, keepdim=True),
                    noise_pred_uncond.chunk(self.batch_size),
                )
            )
        )

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
            "prev_sample"
        ]

        return latents

    @torch.no_grad()
    def diffuse(
        self,
        cond_embeddings,
        cond_latents,
        num_inference_steps=50,
        guidance_scale=7.5,
        offset=1,
        eta=0.0,
    ):

        self.pipe.scheduler.set_timesteps(num_inference_steps)
        self.pipe.scheduler.config.steps_offset = 1

        cond_latents = cond_latents * self.pipe.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = cond_latents
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            latents = self.pipe.scheduler.scale_model_input(latents, t)
            latents = self.denoise(latents, cond_embeddings, i, t, guidance_scale)

        return latents

    def pad_embedding(self, start, end):
        if start.shape == end.shape:
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
