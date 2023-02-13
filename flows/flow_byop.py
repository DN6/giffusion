import inspect
import random

import librosa
import numpy as np
import torch
from utils import (load_video_frames, parse_key_frames, slerp,
                   sync_prompts_to_video)

from .flow_base import BaseFlow, to_tensor


class BYOPFlow(BaseFlow):
    def __init__(
        self,
        pipe,
        text_prompts,
        device,
        guidance_scale=7.5,
        num_inference_steps=50,
        strength=0.5,
        height=512,
        width=512,
        use_fixed_latent=False,
        num_latent_channels=4,
        image_input=None,
        audio_input=None,
        audio_component="both",
        video_input=None,
        seed=42,
        batch_size=1,
        fps=10,
        negative_prompts=[""],
    ):
        super().__init__(pipe, device, batch_size)

        self.pipe_signature = set(inspect.signature(self.pipe).parameters.keys())

        self.text_prompts = text_prompts
        self.negative_prompts = negative_prompts

        self.use_fixed_latent = use_fixed_latent
        self.num_latent_channels = num_latent_channels
        self.vae_scale_factor = self.pipe.vae_scale_factor

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.seed = seed

        self.device = device
        self.generator = torch.Generator(self.device)

        self.fps = fps

        self.check_inputs(image_input, video_input)
        self.image_input = image_input
        self.video_input = video_input

        if self.video_input is not None:
            self.frames, _, _ = load_video_frames(self.video_input)
            _, self.height, self.width = self.frames[0].size()
            self.key_frames = sync_prompts_to_video(text_prompts, self.frames)

        else:
            self.frames, self.frames, _, _ = (None, None, None, None)
            self.key_frames = parse_key_frames(text_prompts)
            self.height, self.width = height, width

        if audio_input is not None:
            self.audio_array, self.sr = librosa.load(audio_input)
            harmonic, percussive = librosa.effects.hpss(self.audio_array, margin=1.0)

            if audio_component == "percussive":
                self.audio_array = percussive

            if audio_component == "harmonic":
                self.audio_array = harmonic
        else:
            self.audio_array, self.sr = (None, None)

        last_frame, _ = max(self.key_frames, key=lambda x: x[0])
        self.max_frames = last_frame + 1

        random.seed(self.seed)
        self.seed_schedule = [
            random.randint(0, 18446744073709551615) for i in range(self.max_frames)
        ]

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

    def check_inputs(self, image_input, video_input):
        if image_input is not None and video_input is not None:
            raise ValueError(
                f"Cannot forward both `image_input` and `video_input`. Please make sure to"
                " only forward one of the two."
            )

    def get_interpolation_schedule(
        self,
        start_frame,
        end_frame,
        fps,
        audio_array=None,
        sr=None,
    ):
        if audio_array is not None:
            return self.get_interpolation_schedule_from_audio(
                start_frame, end_frame, fps, audio_array, sr
            )

        num_frames = (end_frame - start_frame) + 1
        return np.linspace(0, 1, num_frames)

    def get_interpolation_schedule_from_audio(
        self, start_frame, end_frame, fps, audio_array, sr
    ):
        num_frames = (end_frame - start_frame) + 1

        start_sample = int((start_frame / fps) * sr)
        end_sample = int((end_frame / fps) * sr)
        audio_slice = audio_array[start_sample:end_sample]

        # from https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
        onset_env = librosa.onset.onset_strength(audio_slice, sr=sr)
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
        height,
        width,
        generator,
        use_fixed_latent=False,
    ):
        text_output = {}
        latent_output = {}

        start_latent = torch.randn(
            (
                1,
                self.num_latent_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            device=self.pipe.device,
            generator=generator.manual_seed(self.seed),
        )

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, start_prompt = start_key_frame
            end_frame, end_prompt = end_key_frame

            end_latent = (
                start_latent
                if use_fixed_latent
                else torch.randn(
                    (
                        1,
                        self.num_latent_channels,
                        height // self.vae_scale_factor,
                        width // self.vae_scale_factor,
                    ),
                    device=self.pipe.device,
                    generator=generator.manual_seed(self.seed_schedule[end_frame]),
                )
            )

            start_text_embeddings = self.prompt_to_embedding(start_prompt)
            end_text_embeddings = self.prompt_to_embedding(end_prompt)

            interp_schedule = self.get_interpolation_schedule(
                start_frame,
                end_frame,
                self.fps,
                self.audio_array,
                self.sr,
            )

            for i, t in enumerate(interp_schedule):
                latents = slerp(float(t), start_latent, end_latent)
                start_text_embeddings, end_text_embeddings = self.pad_embedding(
                    start_text_embeddings, end_text_embeddings
                )
                embeddings = slerp(float(t), start_text_embeddings, end_text_embeddings)

                latent_output[i + start_frame] = latents
                text_output[i + start_frame] = embeddings

            start_latent = end_latent

        return latent_output, text_output

    def batch_generator(self, frames, batch_size):
        text_batch = []
        latent_batch = []
        image_batch = []
        generator_batch = []

        for frame_idx in frames:
            text_batch.append(self.text_embeddings[frame_idx])
            latent_batch.append(self.init_latents[frame_idx])

            if self.frames is not None:
                image_batch.append(self.frames[frame_idx].unsqueeze(0))

            if len(text_batch) % batch_size == 0:
                text_batch = torch.cat(text_batch, dim=0)
                latent_batch = torch.cat(latent_batch, dim=0)

                generator_batch.append(
                    self.generator.manual_seed(self.seed_schedule[frame_idx])
                )

                if self.frames is not None:
                    image_batch = torch.cat(image_batch, dim=0)

                yield {
                    "text_embeddings": text_batch,
                    "init_latents": latent_batch,
                    "images": image_batch,
                    "generators": generator_batch,
                }

                text_batch = []
                latent_batch = []
                image_batch = []
                generator_batch = []

    def prepare_inputs(self, batch):
        prompt_embeds = batch["text_embeddings"]
        latents = batch["init_latents"]
        images = batch["images"]
        generators = batch["generators"]

        pipe_kwargs = dict(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )

        if "height" in self.pipe_signature:
            pipe_kwargs.update({"height": self.height})

        if "width" in self.pipe_signature:
            pipe_kwargs.update({"width": self.width})

        if "strength" in self.pipe_signature:
            pipe_kwargs.update({"strength": self.strength})

        if "latents" in self.pipe_signature:
            pipe_kwargs.update({"latents": latents})

        if "prompt_embeds" in self.pipe_signature:
            pipe_kwargs.update({"prompt_embeds": prompt_embeds})

        if "negative_prompts" in self.pipe_signature:
            pipe_kwargs.update({"negative_prompts": self.negative_prompts})

        if "image" in self.pipe_signature:
            if (self.video_input is not None) and (len(images) != 0):
                pipe_kwargs.update({"image": images})

            elif self.image_input is not None:
                pipe_kwargs.update({"image": self.image_input})

        if "generator" in self.pipe_signature:
            pipe_kwargs.update({"generator": generators})

        return pipe_kwargs

    def create(self, frames=None):
        for batch in self.batch_generator(
            frames if frames else [i for i in range(self.max_frames)], self.batch_size
        ):
            pipe_kwargs = self.prepare_inputs(batch)

            with torch.autocast("cuda"):
                images = self.pipe(**pipe_kwargs)

            yield images
