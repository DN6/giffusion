import inspect
import random
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from diffusers import ImagePipelineOutput
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from preprocessor import Preprocessor
from utils import (
    apply_lab_color_matching,
    apply_transformation2D,
    curve_from_cn_string,
    get_mel_reduce_func,
    load_video_frames,
    parse_key_frames,
    slerp,
)

from .flow_base import BaseFlow


class MotionCallback:
    def __init__(self, animation_args, padding_mode="border"):
        self.zoom = animation_args.get("zoom", curve_from_cn_string("0:(1.0)"))
        self.translate_x = animation_args.get(
            "translate_x", curve_from_cn_string("0:(0.0)")
        )
        self.translate_y = animation_args.get(
            "translate_y", curve_from_cn_string("0:(0.0)")
        )
        self.angle = animation_args.get("angle", curve_from_cn_string("0:(0.0)"))
        self.padding_mode = padding_mode

    def __call__(self, image, batch):
        frame_idx = batch["frame_ids"][0]
        image_tensor = ToTensor()(image).unsqueeze(0)

        animations = {
            "zoom": self.zoom[frame_idx],
            "translate_x": self.translate_x[frame_idx],
            "translate_y": self.translate_y[frame_idx],
            "angle": self.angle[frame_idx],
        }

        transformed = apply_transformation2D(
            image_tensor,
            animations,
            padding_mode=self.padding_mode,
        )
        transformed = ToPILImage()(transformed[0])

        return [transformed]


class CoherenceCallback:
    def __init__(
        self, init_latent=None, coherence_scale=300, coherence_alpha=0.0, steps=5
    ):
        self.init_latent = init_latent
        self.coherence_scale = coherence_scale
        self.coherence_alpha = coherence_alpha
        self.steps = steps

    def apply_coherence_guidance(self, latents):
        for step in range(self.steps):
            loss = (latents - self.init_latent).pow(2).mean()
            cond_grad = torch.autograd.grad(loss, latents)[0]

            latents = latents - (self.coherence_scale * cond_grad)

        return latents.detach()

    def __call__(self, latents):
        latents.requires_grad = True
        with torch.enable_grad():
            latents = self.apply_coherence_guidance(latents)

        self.init_latent = (self.coherence_alpha * latents) + (
            1.0 - self.coherence_alpha
        ) * self.init_latent

        return latents


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
        video_use_pil_format=False,
        seed=42,
        batch_size=1,
        fps=10,
        negative_prompts="",
        additional_pipeline_arguments={},
        interpolation_type="linear",
        interpolation_args="",
        motion_args=None,
        padding_mode="border",
        coherence_scale=350,
        coherence_alpha=1.0,
        coherence_steps=1,
        noise_schedule="0:(0)",
        use_color_matching=False,
        preprocess=[],
    ):
        super().__init__(pipe, device, batch_size)
        self.pipe_signature = set(inspect.signature(self.pipe).parameters.keys())

        self.text_prompts = text_prompts
        self.negative_prompts = negative_prompts

        self.use_fixed_latent = use_fixed_latent
        self.use_prompt_embeds = use_prompt_embeds
        self.num_latent_channels = num_latent_channels
        self.vae_scale_factor = self.pipe.vae_scale_factor
        self.additional_pipeline_arguments = additional_pipeline_arguments

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.seed = seed

        self.device = device
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        self.fps = fps

        self.preprocess = preprocess
        if len(self.preprocess) > 1 and self.batch_size > 1:
            raise ValueError(
                f"In order to use MultiControlnet",
                f"batch size must be set to 1 but found batch size {self.batch_size}",
            )
        self.preprocessor = Preprocessor(self.preprocess)

        self.check_inputs(image_input, video_input)

        if image_input is not None:
            image_input = self.resize_image_input(image_input)
            self.height, self.width = image_input.size

            self.reference_image = image_input.convert("RGB")
            self.image_input = self.preprocessor(image_input)

        else:
            self.reference_image = self.image_input = None
            self.height, self.width = height, width

        self.video_input = video_input
        self.video_use_pil_format = video_use_pil_format

        self.video_frames = None
        if self.video_input is not None:
            self.video_frames, _, _ = load_video_frames(self.video_input)
            _, self.height, self.width = self.video_frames[0].size()

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

        key_frames = parse_key_frames(text_prompts)
        last_frame, _ = max(key_frames, key=lambda x: x[0])
        self.max_frames = last_frame + 1

        random.seed(self.seed)
        self.seed_schedule = [
            random.randint(0, 18446744073709551615) for i in range(self.max_frames)
        ]

        interpolation_config = {
            "interpolation_type": interpolation_type,
            "interpolation_args": interpolation_args,
        }
        self.init_latents = self.get_init_latents(key_frames, interpolation_config)
        if self.use_prompt_embeds:
            self.prompts = self.get_prompt_embeddings(key_frames, interpolation_config)
        else:
            self.prompts = self.get_prompts(key_frames)

        motion_args = self.prep_animation_args(motion_args)
        if motion_args:
            if self.batch_size != 1:
                raise ValueError(
                    f"In order to use Animation Arguments",
                    f"batch size must be set to 1 but found batch size {self.batch_size}",
                )

            self.motion_callback = MotionCallback(
                motion_args, padding_mode=padding_mode
            )
            self.use_motion = True
        else:
            self.use_motion = False

        self.use_coherence = coherence_scale > 0.0 and self.batch_size == 1
        if self.use_coherence:
            self.coherence_callback = CoherenceCallback(
                coherence_scale=coherence_scale,
                coherence_alpha=coherence_alpha,
                steps=coherence_steps,
            )
        else:
            self.coherence_callback = None

        self.noise_schedule = curve_from_cn_string(noise_schedule)
        self.use_color_matching = use_color_matching
        self.use_color_matching_only = self.use_color_matching and not self.use_motion

    def check_inputs(self, image_input, video_input):
        if image_input is not None and video_input is not None:
            raise ValueError(
                f"Cannot forward both `image_input` and `video_input`. Please make sure to"
                " only forward one of the two."
            )

    def resize_image_input(self, image_input):
        # Resize so image size is divisible by 8
        height, width = image_input.size
        resized_height = height - (height % 8)
        resized_width = width - (width % 8)

        image_input = image_input.resize((resized_height, resized_width), Image.LANCZOS)

        return image_input

    def prep_animation_args(self, animation_args):
        output = {}
        for k, v in animation_args.items():
            if len(v) == 0:
                continue
            output[k] = curve_from_cn_string(v)

        return output

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
            interpolation_args = interpolation_config["interpolation_args"]
            return self.get_sine_interpolation_schedule(
                start_frame, end_frame, interpolation_args
            )

        if interpolation_config["interpolation_type"] == "curve":
            interpolation_args = interpolation_config["interpolation_args"]
            return self.get_curve_interpolation_schedule(
                start_frame, end_frame, interpolation_args
            )

        num_frames = (end_frame - start_frame) + 1

        return np.linspace(0, 1, num_frames)

    def get_sine_interpolation_schedule(
        self, start_frame, end_frame, interpolation_args
    ):
        output = []
        num_frames = (end_frame - start_frame) + 1
        frames = np.arange(num_frames) / num_frames

        interpolation_args = interpolation_args.split(",")
        if len(interpolation_args) == 0:
            interpolation_args = [1.0]
        else:
            interpolation_args = list(map(lambda x: float(x), interpolation_args))

        for frequency in interpolation_args:
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
            y=audio_slice, sr=sr, hop_length=frame_duration
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

    def get_curve_interpolation_schedule(
        self, start_frame, end_frame, interpolation_args
    ):
        curve = curve_from_cn_string(interpolation_args)
        curve_params = []
        for frame in range(start_frame, end_frame + 1):
            curve_params.append(curve[frame])

        return np.array(curve_params)

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
            generator=self.generator,
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
        for frame_idx in range(0, len(frames), batch_size):
            start = frame_idx
            end = frame_idx + batch_size

            frame_batch = frames[start:end]
            prompts = list(map(lambda x: self.prompts[x], frame_batch))
            if self.use_prompt_embeds:
                prompts = torch.cat(prompts, dim=0)

            latents = list(map(lambda x: self.init_latents[x], frame_batch))
            latents = torch.cat(latents, dim=0)

            if self.video_frames is not None:
                images = list(
                    map(lambda x: self.video_frames[x].unsqueeze(0), frame_batch)
                )
                if self.video_use_pil_format:
                    images = list(map(lambda x: ToPILImage()(x[0]), images))
                else:
                    images = torch.cat(images, dim=0)

            else:
                images = []

            yield {
                "prompts": prompts,
                "init_latents": latents,
                "images": images,
                "frame_ids": frame_batch,
            }

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
                # preprocess the current batch of images
                image_input = self.preprocessor(images)
                pipe_kwargs.update({"image": image_input})

            elif self.image_input is not None:
                pipe_kwargs.update({"image": self.image_input})

        if "generator" in self.pipe_signature:
            pipe_kwargs.update({"generator": self.generator})

        pipe_kwargs.update(self.additional_pipeline_arguments)

        return pipe_kwargs

    def get_reference_image(self, image_input):
        if self.reference_image is None:
            self.reference_image = image_input

        return self.reference_image

    def apply_color_matching(self, image_input):
        # Color match the transformed image to the reference
        reference_image = self.get_reference_image(image_input[0])
        image_input = [
            apply_lab_color_matching(image, reference_image) for image in image_input
        ]

        return image_input

    @torch.no_grad()
    def apply_motion(self, image, idx):
        image = image[0]
        image = image.convert("RGB")
        image_input = self.motion_callback(image, idx)

        return image_input

    def apply_coherence(self, latents, batch):
        frame_id = batch["frame_ids"][0]

        if self.coherence_callback.init_latent is None:
            self.coherence_callback.init_latent = latents

        latents = self.coherence_callback(latents)
        noise = self.noise_schedule[frame_id] * torch.randn(latents.shape).to(
            latents.device
        )
        latents = latents + noise
        return latents

    def run_inference(self, batch):
        pipe_kwargs = self.prepare_inputs(batch)
        with torch.autocast("cuda"):
            if self.use_coherence:
                output = self.pipe(**pipe_kwargs, output_type="latent")
                if isinstance(output, np.ndarray):
                    latents = ToTensor()(output)
                else:
                    latents = output.images

                latents = self.apply_coherence(latents, batch)
                with torch.no_grad():
                    image = self.pipe.decode_latents(latents)
                    image = self.postprocess(image, output_type="pil")

                return ImagePipelineOutput(images=image)

            else:
                output = self.pipe(**pipe_kwargs)
                return output

    def create(self, frames=None):
        batchgen = self.batch_generator(
            frames if frames else [i for i in range(self.max_frames)], self.batch_size
        )

        for batch_idx, batch in enumerate(batchgen):
            output = self.run_inference(batch)
            images = output.images

            if self.use_color_matching_only:
                images = self.apply_color_matching(images)
                output.images = images

            yield output, batch["frame_ids"]

            if self.use_motion:
                images = self.apply_motion(images, batch)
                if self.use_color_matching:
                    images = self.apply_color_matching(images)

                self.image_input = self.preprocessor(images)
