import inspect
import json
import random

import librosa
import numpy as np
import pandas as pd
import torch
from utils import (
    get_mel_reduce_func,
    load_video_frames,
    parse_key_frames,
    slerp,
    sync_prompts_to_video,
)

from .flow_base import BaseFlow


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
        use_prompt_embeds=True,
        num_latent_channels=4,
        image_input=None,
        audio_input=None,
        audio_component="both",
        audio_mel_spectogram_reduce="max",
        video_input=None,
        seed=42,
        batch_size=1,
        fps=10,
        negative_prompts="",
        additional_pipeline_arguments="{}",
        interpolation_type="linear",
        frequencies="",
    ):
        super().__init__(pipe, device, batch_size)

        self.pipe_signature = set(inspect.signature(self.pipe).parameters.keys())

        frequencies = frequencies.split(",")
        interpolation_config = {
            "interpolation_type": interpolation_type,
            "frequencies": frequencies,
        }

        self.text_prompts = text_prompts
        self.negative_prompts = negative_prompts

        self.use_fixed_latent = use_fixed_latent
        self.use_prompt_embeds = use_prompt_embeds
        self.num_latent_channels = num_latent_channels
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.additional_pipeline_argumenets = json.loads(additional_pipeline_arguments)

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
            key_frames = sync_prompts_to_video(text_prompts, self.frames)

        else:
            self.frames, self.frames, _, _ = (None, None, None, None)
            key_frames = parse_key_frames(text_prompts)
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

        self.audio_mel_reduce_func = get_mel_reduce_func(audio_mel_spectogram_reduce)

        last_frame, _ = max(key_frames, key=lambda x: x[0])
        self.max_frames = last_frame + 1

        random.seed(self.seed)
        self.seed_schedule = [
            random.randint(0, 18446744073709551615) for i in range(self.max_frames)
        ]

        self.init_latents = self.get_init_latents(key_frames, interpolation_config)
        if self.use_prompt_embeds:
            self.prompts = self.get_prompt_embeddings(key_frames, interpolation_config)
        else:
            self.prompts = self.get_prompts(key_frames)

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
        interpolation_config,
        audio_array=None,
        sr=None,
    ):
        if audio_array is not None:
            return self.get_interpolation_schedule_from_audio(
                start_frame, end_frame, fps, audio_array, sr
            )

        if interpolation_config["interpolation_type"] == "sine":
            frequencies = interpolation_config["frequencies"]
            return self.get_sine_interpolation_schedule(
                start_frame, end_frame, frequencies
            )

        num_frames = (end_frame - start_frame) + 1

        return np.linspace(0, 1, num_frames)

    def get_sine_interpolation_schedule(self, start_frame, end_frame, frequencies):
        output = []
        num_frames = (end_frame - start_frame) + 1
        frames = np.arange(num_frames) / num_frames

        if len(frequencies) == 0:
            frequencies = [1.0]
        else:
            frequencies = list(map(lambda x: float(x), frequencies))

        for frequency in frequencies:
            curve = np.sin(np.pi * frames * frequency) ** 2
            output.append(curve)

        schedule = sum(output)
        schedule = (schedule - np.min(schedule)) / np.ptp(schedule)

        return schedule

    def get_interpolation_schedule_from_audio(
        self, start_frame, end_frame, fps, audio_array, sr
    ):
        num_frames = (end_frame - start_frame) + 1
        frame_duration = sr // fps

        start_sample = int((start_frame / fps) * sr)
        end_sample = int((end_frame / fps) * sr)
        audio_slice = audio_array[start_sample:end_sample]

        # from https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
        spec = librosa.feature.melspectrogram(
            audio_slice, sr=sr, hop_length=frame_duration
        )
        spec = self.audio_mel_reduce_func(spec, axis=0)
        spec_norm = librosa.util.normalize(spec)

        schedule_x = np.linspace(0, len(spec_norm), len(spec_norm))
        schedule_y = spec_norm
        schedule_y = np.cumsum(spec_norm)
        schedule_y /= schedule_y[-1]

        resized_schedule = np.linspace(0, len(schedule_y), num_frames)
        interp_schedule = np.interp(resized_schedule, schedule_x, schedule_y)

        return interp_schedule

    @torch.no_grad()
    def get_prompt_embeddings(self, key_frames, interpolation_config):
        output = {}

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, start_prompt = start_key_frame
            end_frame, end_prompt = end_key_frame

            start_prompt_embed = self.prompt_to_embedding(start_prompt)
            end_prompt_embed = self.prompt_to_embedding(end_prompt)

            interp_schedule = self.get_interpolation_schedule(
                start_frame,
                end_frame,
                self.fps,
                interpolation_config,
                self.audio_array,
                self.sr,
            )

            for i, t in enumerate(interp_schedule):
                prompt_embed = slerp(float(t), start_prompt_embed, end_prompt_embed)
                output[i + start_frame] = prompt_embed

        return output

    def get_prompts(self, key_frames, integer=True, method="linear"):
        output = {}
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        for frame_idx, prompt in key_frames:
            key_frame_series[frame_idx] = prompt

        key_frame_series = key_frame_series.ffill()
        for frame_idx, prompt in enumerate(key_frame_series):
            output[frame_idx] = prompt

        return output

    @torch.no_grad()
    def get_init_latents(self, key_frames, interpolation_config):
        output = {}
        start_latent = torch.randn(
            (
                1,
                self.num_latent_channels,
                self.height // self.vae_scale_factor,
                self.width // self.vae_scale_factor,
            ),
            device=self.pipe.device,
            generator=self.generator.manual_seed(self.seed),
        )

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, _ = start_key_frame
            end_frame, _ = end_key_frame

            end_latent = (
                start_latent
                if self.use_fixed_latent
                else torch.randn(
                    (
                        1,
                        self.num_latent_channels,
                        self.height // self.vae_scale_factor,
                        self.width // self.vae_scale_factor,
                    ),
                    device=self.pipe.device,
                    generator=self.generator.manual_seed(self.seed_schedule[end_frame]),
                )
            )

            interp_schedule = self.get_interpolation_schedule(
                start_frame,
                end_frame,
                self.fps,
                interpolation_config,
                self.audio_array,
                self.sr,
            )

            for i, t in enumerate(interp_schedule):
                latents = slerp(float(t), start_latent, end_latent)
                output[i + start_frame] = latents

            start_latent = end_latent

        return output

    def batch_generator(self, frames, batch_size):
        prompt_batch = []
        latent_batch = []
        image_batch = []

        for frame_idx in frames:
            prompt_batch.append(self.prompts[frame_idx])
            latent_batch.append(self.init_latents[frame_idx])

            if self.frames is not None:
                image_batch.append(self.frames[frame_idx].unsqueeze(0))

            if len(prompt_batch) % batch_size == 0:
                if self.use_prompt_embeds:
                    prompt_batch = torch.cat(prompt_batch, dim=0)

                latent_batch = torch.cat(latent_batch, dim=0)

                if self.frames is not None:
                    image_batch = torch.cat(image_batch, dim=0)

                yield {
                    "prompts": prompt_batch,
                    "init_latents": latent_batch,
                    "images": image_batch,
                }

                prompt_batch = []
                latent_batch = []
                image_batch = []

    def prepare_inputs(self, batch):
        prompts = batch["prompts"]
        latents = batch["init_latents"]
        images = batch["images"]

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

        if "prompt_embeds" in self.pipe_signature and self.use_prompt_embeds:
            pipe_kwargs.update({"prompt_embeds": prompts})
        elif "prompt" in self.pipe_signature and not self.use_prompt_embeds:
            pipe_kwargs.update({"prompt": prompts})

        if "negative_prompts" in self.pipe_signature:
            pipe_kwargs.update(
                {"negative_prompts": [self.negative_prompts] * len(prompts)}
            )

        if "image" in self.pipe_signature:
            if (self.video_input is not None) and (len(images) != 0):
                pipe_kwargs.update({"image": images})

            elif self.image_input is not None:
                pipe_kwargs.update({"image": [self.image_input] * len(prompts)})

        if "generator" in self.pipe_signature:
            pipe_kwargs.update({"generator": self.generator.manual_seed(self.seed)})

        pipe_kwargs.update(self.additional_pipeline_argumenets)

        return pipe_kwargs

    def create(self, frames=None):
        for batch in self.batch_generator(
            frames if frames else [i for i in range(self.max_frames)], self.batch_size
        ):
            pipe_kwargs = self.prepare_inputs(batch)

            with torch.autocast("cuda"):
                images = self.pipe(**pipe_kwargs)

            yield images
