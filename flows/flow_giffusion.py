import inspect

import numpy as np
import torch
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from utils import parse_key_frames, slerp

from .flow_base import BaseFlow


class GiffusionFlow(BaseFlow):
    def __init__(
        self,
        pipe,
        text_prompts,
        guidance_scale,
        num_inference_steps,
        width,
        height,
        use_fixed_latent,
        device,
        seed=42,
        batch_size=1,
        fps=10,
        generator=None,
    ):
        super().__init__(pipe, device, batch_size)

        self.text_prompts = text_prompts
        self.width, self.height = width, height
        self.use_fixed_latent = use_fixed_latent
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        self.seed = seed

        self.key_frames = parse_key_frames(text_prompts)
        (
            self.init_latents,
            self.text_embeddings,
        ) = self.get_init_latents_and_text_embeddings(
            self.key_frames,
            self.height,
            self.width,
            self.generator,
            self.use_fixed_latent,
        )

    @torch.no_grad()
    def get_init_latents_and_text_embeddings(
        self, key_frames, height, width, generator, use_fixed_latent=False
    ):
        text_output = {}
        latent_output = {}

        start_key_frame, *key_frames = key_frames
        start_frame_idx, start_prompt = start_key_frame

        start_latent = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            device=self.pipe.device,
            generator=generator,
        )
        start_text_embeddings = self.prompt_to_embedding(start_prompt)

        for key_frame in key_frames:
            current_frame_idx, current_prompt = key_frame

            current_latent = (
                start_latent
                if use_fixed_latent
                else torch.randn(
                    (1, self.pipe.unet.in_channels, height // 8, width // 8),
                    device=self.pipe.device,
                    generator=generator,
                )
            )
            current_text_embeddings = self.prompt_to_embedding(current_prompt)

            num_steps = current_frame_idx - start_frame_idx
            for i, t in enumerate(np.linspace(0, 1, num_steps + 1)):
                latents = slerp(float(t), start_latent, current_latent)

                start_text_embeddings, current_text_embeddings = self.pad_embedding(
                    start_text_embeddings, current_text_embeddings
                )

                embeddings = slerp(
                    float(t), start_text_embeddings, current_text_embeddings
                )

                latent_output[i + start_frame_idx] = latents
                text_output[i + start_frame_idx] = embeddings

            start_latent = current_latent
            start_text_embeddings = current_text_embeddings

            start_frame_idx = current_frame_idx

        return latent_output, text_output

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

        batch_size = self.batch_size

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.pipe.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = offset
        self.pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        cond_latents = cond_latents * self.pipe.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        max_length = cond_embeddings.shape[1]
        uncond_input = self.pipe.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        latents = cond_latents
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            latents = self.pipe.scheduler.scale_model_input(latents, t)
            latents = self.denoise(latents, text_embeddings, i, t, guidance_scale)

        return latents

    @torch.no_grad()
    def denoise(self, latents, text_embeddings, i, t, guidance_scale):
        accepts_eta = "eta" in set(
            inspect.signature(self.pipe.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

        latent_model_input = torch.cat([latents] * text_embeddings.shape[0])

        noise_pred = self.pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        )["sample"]

        pred_decomp = noise_pred.chunk(text_embeddings.shape[0])
        noise_pred_uncond, noise_pred_cond = pred_decomp[0], torch.cat(
            pred_decomp[1:], dim=0
        ).mean(dim=0, keepdim=True)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
            "prev_sample"
        ]

        return latents

    def create(self, frame_idx):
        init_latents, text_embeddings = (
            self.init_latents[frame_idx],
            self.text_embeddings[frame_idx],
        )
        latents = self.diffuse(
            text_embeddings, init_latents, self.num_inference_steps, self.guidance_scale
        )
        image_tensors = self.decode_latents(latents)

        image_array = self.postprocess(image_tensors)
        image_array, has_nsfw_content = self.safety_check(image_array)

        images = self.numpy_to_pil(image_array)

        return images
