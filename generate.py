import glob
import inspect
import os
import uuid

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import typer
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from PIL import Image
from torch import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from comet import start_experiment
from datamodule import LatentsDataset

PRETRAINED_MODEL_NAME = os.getenv(
    "PRETRAINED_MODEL_NAME", "CompVis/stable-diffusion-v1-4"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(
    PRETRAINED_MODEL_NAME, use_auth_token=True
)
pipe.to(device)

SCHEDULERS = dict(
    pndms=PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    ),
    ddim_scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
    klms_scheduler=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    ),
)


experiment = start_experiment()
if experiment:
    run_path = f"./{experiment.name}"
else:
    run_path = f"./{uuid.uuid4().hex}"

os.makedirs(run_path, exist_ok=True)


def save_gif(frames, filename="./output.gif"):
    imgs = (Image.open(f) for f in sorted(glob.glob(frames)))
    img = next(imgs)
    img.save(
        fp=filename,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=200,
        loop=1,
    )


def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images * 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1)
    images = images.cpu()

    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


@torch.no_grad()
def denoise(latents, pipe, text_embeddings, t, guidance_scale):
    latent_model_input = torch.cat([latents] * 2)

    noise_pred = pipe.unet(
        latent_model_input, t, encoder_hidden_states=text_embeddings
    )["sample"]

    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )

    latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    return latents


@torch.no_grad()
def diffuse(
    pipe,
    cond_embeddings,
    cond_latents,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
):

    batch_size, n, h, w = cond_latents.shape
    # set timesteps
    accepts_offset = "offset" in set(
        inspect.signature(pipe.scheduler.set_timesteps).parameters.keys()
    )
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = 0.0

    # Classifier Free Guidance
    max_length = cond_embeddings.shape[1]
    uncond_input = pipe.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    noise = torch.randn(cond_latents.shape, device=device, generator=generator)
    latents = noise
    for i, t in enumerate(pipe.scheduler.timesteps):
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            sigma = pipe.scheduler.sigmas[i]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latents = latents / ((sigma**2 + 1) ** 0.5)

        latents = denoise(latents, pipe, text_embeddings, t, guidance_scale)

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    images = pipe.vae.decode(latents).sample
    images = postprocess(images)

    return images


def run(
    text_prompt_inputs,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    scheduler="pndms",
    fps=24,
):
    if experiment:
        experiment.log_parameters(
            {
                "text_prompt_inputs": text_prompt_inputs,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "scheduler": scheduler,
                "seed": seed,
                "fps": fps,
            }
        )

    pipe.scheduler = SCHEDULERS.get(scheduler)

    generator = torch.Generator(device=device).manual_seed(seed)

    dataset = LatentsDataset(pipe, text_prompt_inputs, generator)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_frames = []
    for frame_idx, (cond_embeddings, text_embeddings) in tqdm(
        enumerate(dataloader), total=len(dataset)
    ):
        text_embeddings = text_embeddings.to(device)
        cond_embeddings = cond_embeddings.to(device)

        with autocast("cuda"):
            image_tensors = diffuse(
                pipe,
                text_embeddings,
                cond_embeddings,
                seed,
                num_inference_steps,
                guidance_scale,
                generator,
            )

        img_save_path = f"{run_path}/{frame_idx:04d}.png"
        images = numpy_to_pil(image_tensors.numpy())
        images[0].save(img_save_path)
        output_frames.append(img_save_path)

        if experiment:
            experiment.log_image(img_save_path, image_name=f"{frame_idx:04d}")

    output_filename = f"{run_path}/output.gif"
    save_gif(
        frames=output_frames,
        filename=output_filename,
    )

    if experiment:
        experiment.log_asset(output_filename, ftype="image")

    return output_filename


if __name__ == "__main__":
    typer.run(run)
