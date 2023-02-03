import logging
import os
from datetime import datetime

import typer
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from tqdm import tqdm

from comet import start_experiment
from flows import AudioReactiveFlow, GiffusionFlow, VideoInitFlow
from utils import save_gif, save_video

logger = logging.getLogger(__name__)

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
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=1.0,
    batch_size=1,
    seed=42,
    fps=24,
    scheduler="pndms",
    use_fixed_latent=False,
    audio_input=None,
    audio_component="both",
    image_input=None,
    video_input=None,
    output_format="gif",
):
    if pipe is None:
        raise ValueError(
            "Pipline object has not been created. Please load a pipline before submitting"
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
        }
        if video_input is not None:
            parameters.update({"strength": strength})
        experiment.log_parameters(parameters)

    pipe.scheduler = SCHEDULERS.get(scheduler)

    if audio_input is not None:
        if experiment:
            experiment.log_asset(audio_input)

        flow = AudioReactiveFlow(
            pipe=pipe,
            text_prompts=text_prompt_inputs,
            audio_input=audio_input,
            audio_component=audio_component,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            device=device,
            fps=fps,
            use_fixed_latent=use_fixed_latent,
            seed=seed,
            batch_size=batch_size,
            init_image=image_input,
        )
    elif video_input is not None:
        if experiment:
            experiment.log_asset(video_input)

        flow = VideoInitFlow(
            pipe=pipe,
            text_prompts=text_prompt_inputs,
            video_input=video_input,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            device=device,
            fps=fps,
            use_fixed_latent=use_fixed_latent,
            batch_size=batch_size,
            seed=seed,
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
            batch_size=batch_size,
            seed=seed,
            init_image=image_input,
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
