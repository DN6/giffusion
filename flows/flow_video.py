import inspect

import numpy as np
import torch
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from utils import load_video_frames, parse_key_frames, slerp

from .flow_base import BaseFlow


class VideoInitFlow(BaseFlow):
    def __init__(
        self,
        pipe,
        text_prompts,
        video_input,
        guidance_scale,
        strength,
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
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.generator = generator
        self.seed = seed

        self.frames, self.audio, metadata = load_video_frames(video_input)
        self.key_frames = parse_key_frames(text_prompts)

        self.max_frames = len(self.frames)
        self.fps = metadata["video_fps"]
        (
            self.init_latents,
            self.text_embeddings,
        ) = self.get_init_latents_and_text_embeddings(
            self.key_frames,
            self.frames,
            self.height,
            self.width,
            self.generator,
            self.use_fixed_latent,
        )

    @torch.no_grad()
    def get_init_latents_and_text_embeddings(
        self, key_frames, frames, height, width, generator, use_fixed_latent=False
    ):
        text_output = {}
        latent_output = {}

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, start_prompt = start_key_frame
            end_frame, end_prompt = end_key_frame

            start_image = self.preprocess(frames[start_frame], (height, width))
            start_latent = self.encode_latents(start_image.unsqueeze(0))

            end_image = self.preprocess(frames[end_frame], (height, width))
            end_latent = self.encode_latents(end_image.unsqueeze(0))

            start_text_embeddings = self.prompt_to_embedding(start_prompt)
            end_text_embeddings = self.prompt_to_embedding(end_prompt)

            num_frames = end_frame - start_frame
            interp_schedule = np.linspace(0, 1, num_frames + 1)
            for i, t in enumerate(interp_schedule):
                latents = slerp(float(t), start_latent, end_latent)

                start_text_embeddings, end_text_embeddings = self.pad_embedding(
                    start_text_embeddings, end_text_embeddings
                )
                embeddings = slerp(float(t), start_text_embeddings, end_text_embeddings)

                latent_output[i + start_frame] = latents
                text_output[i + start_frame] = embeddings

        return latent_output, text_output

    @torch.no_grad()
    def diffuse(
        self,
        cond_embeddings,
        cond_latents,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=1.0,
        offset=1,
        eta=0.0,
        generator=None,
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

        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(
            cond_latents.shape,
            generator=generator,
            device=self.device,
            dtype=cond_embeddings.dtype,
        )
        cond_latents = self.scheduler.add_noise(cond_latents, noise, timesteps)

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

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        latents = cond_latents

        diffuse_timesteps = self.pipe.scheduler.timesteps[t_start:].to(self.device)
        for i, t in enumerate(diffuse_timesteps):
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
        latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
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
        images = self.numpy_to_pil(image_array)

        return images
