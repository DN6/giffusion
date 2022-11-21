import logging
import os
from datetime import datetime

import torch
import typer
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from torch import autocast
from torchvision import transforms as T
from tqdm import tqdm

from comet import start_experiment
from flows import AudioReactiveFlow, GiffusionFlow, VideoInitFlow
from utils import save_gif, save_video

logger = logging.getLogger(__name__)


PRETRAINED_MODEL_NAME = os.getenv(
    "PRETRAINED_MODEL_NAME", "CompVis/stable-diffusion-v1-4"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(
    PRETRAINED_MODEL_NAME, use_auth_token=True
)
pipe.enable_attention_slicing()
pipe.to(device)

OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH", "../generated")

SCHEDULERS = dict(
    pndms=PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps=True,
    ),
    ddim=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
    klms=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    ),
)


def run(
    text_prompt_inputs,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=1.0,
    seed=42,
    fps=24,
    scheduler="pndms",
    use_fixed_latent=False,
    audio_input=None,
    video_input=None,
    output_format="gif",
):

    experiment = start_experiment()
    run_path = os.path.join(
        OUTPUT_BASE_PATH, datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    )
    os.makedirs(run_path, exist_ok=True)

    if experiment:
        experiment.log_parameters(
            {
                "text_prompt_inputs": text_prompt_inputs,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "fps": fps,
                "use_fixed_latent": use_fixed_latent,
            }
        )
    pipe.scheduler = SCHEDULERS.get(scheduler)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    if audio_input:
        if experiment:
            experiment.log_asset(audio_input)

        flow = AudioReactiveFlow(
            pipe=pipe,
            text_prompts=text_prompt_inputs,
            audio_input=audio_input,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            device=device,
            fps=fps,
            use_fixed_latent=use_fixed_latent,
            generator=generator,
        )
    if video_input:
        if experiment:
            experiment.log_asset(video_input)

        flow = VideoInitFlow(
            pipe=pipe,
            text_prompts=text_prompt_inputs,
            video_input=video_input,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            device=device,
            fps=fps,
            use_fixed_latent=use_fixed_latent,
            generator=generator,
        )

    else:
        flow = GiffusionFlow(
            pipe=pipe,
            text_prompts=text_prompt_inputs,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            device=device,
            use_fixed_latent=use_fixed_latent,
            generator=generator,
        )
    max_frames = flow.max_frames

    output_frames = []
    for frame_idx in tqdm(range(max_frames + 1), total=max_frames + 1):
        with autocast("cuda"):
            images = flow.create(frame_idx)

        img_save_path = f"{run_path}/{frame_idx:04d}.png"
        images[0].save(img_save_path)
        output_frames.append(img_save_path)

        if experiment:
            experiment.log_image(img_save_path, image_name="frame", step=frame_idx)

    if output_format == "gif":
        output_filename = f"{run_path}/output.gif"
        save_gif(frames=output_frames, filename=output_filename, fps=fps)

        preview_filename = f"{run_path}/output-preview.gif"
        save_gif(frames=output_frames, filename=preview_filename, fps=fps, quality=35)

    if output_format == "mp4":
        output_filename = f"{run_path}/output.mp4"
        save_video(frames=output_frames, filename=output_filename, fps=fps)

        preview_filename = f"{run_path}/output-preview.mp4"
        save_video(frames=output_frames, filename=preview_filename, fps=fps, quality=35)

    if experiment:
        experiment.log_asset(output_filename)
        experiment.log_asset(preview_filename)

    return preview_filename


if __name__ == "__main__":
    typer.run(run)
