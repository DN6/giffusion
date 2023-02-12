import logging
import os
from datetime import datetime

import typer
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils.logging import disable_progress_bar
from tqdm import tqdm

from comet import start_experiment
from flows import BYOPFlow
from flows.flow_byop import BYOPFlow
from utils import save_gif, save_video

logger = logging.getLogger(__name__)

# Disable denoising progress bar
disable_progress_bar()

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
    pipe,
    text_prompt_inputs,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=1.0,
    batch_size=1,
    seed=42,
    fps=24,
    scheduler="pndms",
    use_fixed_latent=False,
    num_latent_channels=4,
    audio_input=None,
    audio_component="both",
    image_input=None,
    video_input=None,
    output_format="mp4",
    model_name="runwayml/stable-diffusion-v1-5",
):
    if pipe is None:
        raise ValueError(
            "Pipline object has not been created. Please load a Pipline before submitting a run"
        )

    experiment = start_experiment()

    run_name = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    run_path = os.path.join(OUTPUT_BASE_PATH, run_name)
    os.makedirs(run_path, exist_ok=True)

    device = pipe.device

    if experiment:
        parameters = {
            "text_prompt_inputs": text_prompt_inputs,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "scheduler": scheduler,
            "seed": seed,
            "fps": fps,
            "use_fixed_latent": use_fixed_latent,
            "audio_component": audio_component,
            "output_format": output_format,
            "pipeline_name": pipe.config["_class_name"],
            "model_name": model_name,
        }
        if (video_input is not None) or (image_input is not None):
            parameters.update({"strength": strength})

        experiment.log_parameters(parameters)

    pipe.scheduler = SCHEDULERS.get(scheduler)
    flow = BYOPFlow(
        pipe=pipe,
        text_prompts=text_prompt_inputs,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        use_fixed_latent=use_fixed_latent,
        device=device,
        image_input=image_input,
        audio_input=audio_input,
        audio_component=audio_component,
        video_input=video_input,
        seed=seed,
        batch_size=batch_size,
        fps=fps,
    )

    max_frames = flow.max_frames
    output_frames = []

    image_generator = flow.create()
    frame_idx = 0

    for output in tqdm(image_generator, total=max_frames // flow.batch_size):
        images = output.images
        for image in images:
            img_save_path = f"{run_path}/{frame_idx:04d}.png"
            image.save(img_save_path)
            output_frames.append(img_save_path)

            if experiment:
                experiment.log_image(img_save_path, image_name="frame", step=frame_idx)
            frame_idx += 1

    if output_format == "gif":
        output_filename = f"{run_path}/output.gif"
        save_gif(frames=output_frames, filename=output_filename, fps=fps)

        preview_filename = f"{run_path}/output-preview.gif"
        save_gif(frames=output_frames, filename=preview_filename, fps=fps, quality=35)

    if output_format == "mp4":
        output_filename = f"{run_path}/output.mp4"

        save_video(
            frames=output_frames,
            filename=output_filename,
            fps=fps,
            audio_input=audio_input,
        )

        preview_filename = f"{run_path}/output-preview.mp4"
        save_video(
            frames=output_frames,
            filename=preview_filename,
            fps=fps,
            quality=35,
            audio_input=audio_input,
        )

    if experiment:
        experiment.log_asset(output_filename)
        experiment.log_asset(preview_filename)

    return preview_filename


if __name__ == "__main__":
    typer.run(run)
