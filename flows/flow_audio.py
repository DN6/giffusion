import inspect

import librosa
import numpy as np
import torch
from utils import parse_key_frames, slerp

from .flow_base import BaseFlow


class AudioReactiveFlow(BaseFlow):
    def __init__(
        self,
        pipe,
        text_prompts,
        audio_input,
        audio_component,
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
        self.max_frames, _ = max(self.key_frames, key=lambda x: x[0])
        self.fps = fps
        (
            self.init_latents,
            self.text_embeddings,
        ) = self.get_init_latents_and_text_embeddings(
            self.key_frames,
            audio_input,
            audio_component,
            self.height,
            self.width,
            self.generator,
            self.use_fixed_latent,
        )

    def get_interpolation_schedule(self, audio_array, sr, num_frames):
        # from https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
        onset_env = librosa.onset.onset_strength(audio_array, sr=sr)
        onset_env = librosa.util.normalize(onset_env)

        schedule_x = np.linspace(0, len(onset_env), len(onset_env))
        schedule_y = np.cumsum(onset_env)
        schedule_y /= schedule_y[-1]

        resized_schedule = np.linspace(0, len(schedule_y), num_frames)
        interp_schedule = np.interp(resized_schedule, schedule_x, schedule_y)

        return interp_schedule

    @torch.no_grad()
    def get_init_latents_and_text_embeddings(
        self,
        key_frames,
        audio_input,
        audio_component,
        height,
        width,
        generator,
        use_fixed_latent=False,
    ):
        text_output = {}
        latent_output = {}

        audio_array, sr = librosa.load(audio_input)
        harmonic, percussive = librosa.effects.hpss(audio_array, margin=1.0)

        if audio_component == "percussive":
            audio_array = percussive

        if audio_component == "harmonic":
            audio_array = harmonic

        start_latent = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            device=self.pipe.device,
            generator=generator,
        )

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):

            start_frame, start_prompt = start_key_frame
            end_frame, end_prompt = end_key_frame
            num_frames = (end_frame - start_frame) + 1

            end_latent = (
                start_latent
                if use_fixed_latent
                else torch.randn(
                    (1, self.pipe.unet.in_channels, height // 8, width // 8),
                    device=self.pipe.device,
                    generator=generator,
                )
            )

            start_text_embeddings = self.prompt_to_embedding(start_prompt)
            end_text_embeddings = self.prompt_to_embedding(end_prompt)

            start_sample = int((start_frame / self.fps) * sr)
            end_sample = int((end_frame / self.fps) * sr)

            audio_slice = audio_array[start_sample:end_sample]
            interp_schedule = self.get_interpolation_schedule(
                audio_slice, sr, num_frames
            )

            for i, t in enumerate(interp_schedule):
                latents = slerp(float(t), start_latent, end_latent)
                start_text_embeddings, end_text_embeddings = self.pad_embedding(
                    start_text_embeddings, end_text_embeddings
                )
                embeddings = torch.lerp(start_text_embeddings, end_text_embeddings, t)

                latent_output[i + start_frame] = latents
                text_output[i + start_frame] = embeddings

            start_latent = end_latent

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

        self.pipe.scheduler.set_timesteps(num_inference_steps)
        self.pipe.scheduler.config.steps_offset = 1
        timesteps_tensor = self.pipe.scheduler.timesteps.to(self.device)

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
        for i, t in enumerate(timesteps_tensor):
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
