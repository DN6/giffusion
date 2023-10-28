import json
import logging
import os
from datetime import datetime

import typer
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils.logging import disable_progress_bar
from tqdm import tqdm

from comet import start_experiment
from flows import BYOPFlow
from flows.flow_byop import BYOPFlow
from utils import save_gif, save_parameters, save_video

logger = logging.getLogger(__name__)

# Disable denoising progress bar
disable_progress_bar()


def load_scheduler(scheduler, **kwargs):
    scheduler_map = dict(
        pndms=PNDMScheduler(**kwargs),
        ddim=DDIMScheduler(**kwargs),
        ddpm=DDPMScheduler(**kwargs),
        klms=LMSDiscreteScheduler(**kwargs),
        dpm=DPMSolverSinglestepScheduler(**kwargs),
        dpm_ads=KDPM2AncestralDiscreteScheduler(**kwargs),
        deis=DEISMultistepScheduler(**kwargs),
        euler=EulerDiscreteScheduler(**kwargs),
        euler_ads=EulerAncestralDiscreteScheduler(**kwargs),
        repaint=RePaintScheduler(**kwargs),
        unipc=UniPCMultistepScheduler(**kwargs),
    )
    return scheduler_map.get(scheduler)


def run(
    run_path,
    pipe,
    text_prompt_inputs,
    negative_prompt_inputs,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.5,
    batch_size=1,
    seed=42,
    fps=24,
    use_default_scheduler=False,
    scheduler="pndms",
    scheduler_kwargs="{}",
    use_fixed_latent=False,
    use_prompt_embeds=True,
    num_latent_channels=4,
    audio_input=None,
    audio_component="both",
    mel_spectogram_reduce="max",
    image_input=None,
    video_input=None,
    video_use_pil_format=False,
    output_format="mp4",
    model_name="runwayml/stable-diffusion-v1-5",
    controlnet_name=None,
    adapter_name=None,
    lora_name=None,
    additional_pipeline_arguments="{}",
    interpolation_type="linear",
    interpolation_args="",
    zoom="",
    translate_x="",
    translate_y="",
    angle="",
    padding_mode="border",
    coherence_scale=300,
    coherence_alpha=1.0,
    coherence_steps=3,
    noise_schedule=None,
    use_color_matching=False,
    preprocess=None,
):
    if pipe is None:
        raise ValueError(
            "Pipline object has not been created. Please load a Pipline before submitting a run"
        )

    timestamp = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    device = pipe.device

    if not use_default_scheduler:
        scheduler_kwargs = json.loads(scheduler_kwargs)
        if not scheduler_kwargs:
            scheduler_kwargs = {
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
            }

        pipe.scheduler = load_scheduler(scheduler, **scheduler_kwargs)

    motion_args = {
        "zoom": zoom,
        "translate_x": translate_x,
        "translate_y": translate_y,
        "angle": angle,
    }

    additional_pipeline_arguments = json.loads(additional_pipeline_arguments)

    flow = BYOPFlow(
        pipe=pipe,
        text_prompts=text_prompt_inputs,
        negative_prompts=negative_prompt_inputs,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        use_fixed_latent=use_fixed_latent,
        use_prompt_embeds=use_prompt_embeds,
        num_latent_channels=num_latent_channels,
        device=device,
        image_input=image_input,
        audio_input=audio_input,
        audio_component=audio_component,
        audio_mel_spectogram_reduce=mel_spectogram_reduce,
        video_input=video_input,
        video_use_pil_format=video_use_pil_format,
        seed=seed,
        batch_size=batch_size,
        fps=fps,
        additional_pipeline_arguments=additional_pipeline_arguments,
        interpolation_type=interpolation_type,
        interpolation_args=interpolation_args,
        motion_args=motion_args,
        padding_mode=padding_mode,
        coherence_scale=coherence_scale,
        coherence_alpha=coherence_alpha,
        coherence_steps=coherence_steps,
        noise_schedule=noise_schedule,
        use_color_matching=use_color_matching,
        preprocess=preprocess,
    )

    max_frames = flow.max_frames
    parameters = {
        "prompts": {
            "text_prompt_inputs": text_prompt_inputs,
            "negative_prompt_inputs": negative_prompt_inputs,
        },
        "diffusion_settings": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "batch_size": batch_size,
            "seed": seed,
            "use_fixed_latent": use_fixed_latent,
            "use_prompt_embeds": use_prompt_embeds,
            "strength": strength,
            "scheduler": scheduler,
            "use_default_scheduler": use_default_scheduler,
            "scheduler_kwargs": scheduler_kwargs,
            "image_height": height,
            "image_width": width,
            "additional_pipeline_arguments": additional_pipeline_arguments,
        },
        "preprocessing_settings": {
            "preprocess": preprocess,
        },
        "pipeline_settings": {
            "pipeline_name": pipe.__class__.__name__,
            "model_name": model_name,
            "controlnet_name": controlnet_name,
            "adapter_name": adapter_name,
            "lora_name": lora_name,
        },
        "animation_settings": {
            "interpolation_type": interpolation_type,
            "interpolation_args": interpolation_args,
            "zoom": zoom,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "angle": angle,
            "padding_mode": padding_mode,
            "coherence_scale": coherence_scale,
            "coherence_alpha": coherence_alpha,
            "coherence_steps": coherence_steps,
            "noise_schedule": noise_schedule,
            "use_color_matching": use_color_matching,
        },
        "media": {
            "audio_settings": {
                "audio_component": audio_component,
                "mel_spectogram_reduce": mel_spectogram_reduce,
            },
            "video_settings": {
                "video_use_pil_format": video_use_pil_format,
            },
        },
        "output_settings": {
            "output_format": output_format,
            "fps": fps,
        },
        "frame_information": {"last_frame_id": max_frames},
        "timestamp": timestamp,
    }
    if (video_input is not None) or (image_input is not None):
        parameters.update({"strength": strength})

    save_parameters(run_path, parameters)

    run_image_save_path = f"{run_path}/imgs"
    os.makedirs(run_image_save_path, exist_ok=True)

    image_generator = flow.create()
    for output, frame_ids in tqdm(image_generator, total=max_frames // flow.batch_size):
        images = output.images
        for image, frame_idx in zip(images, frame_ids):
            img_save_path = f"{run_image_save_path}/{frame_idx:04d}.png"
            image.save(img_save_path)

            yield image, img_save_path


if __name__ == "__main__":
    typer.run(run)
