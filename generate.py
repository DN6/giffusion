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
from torchvision import transforms as T
from tqdm import tqdm

from comet import start_experiment
from utils import parse_key_frames, slerp

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


experiment = start_experiment()
if experiment:
    run_path = f"./{experiment.name}"
else:
    run_path = f"./{uuid.uuid4().hex}"

os.makedirs(run_path, exist_ok=True)


def save_gif(frames, filename="./output.gif", fps=24):
    imgs = (Image.open(f) for f in sorted(glob.glob(frames)))
    duration = len(imgs) // fps

    img = next(imgs)
    img.save(
        fp=filename,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=duration,
        loop=1,
    )


def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images * 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1)
    images = images.cpu().numpy()
    images = numpy_to_pil(images)

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
def denoise(latents, pipe, text_embeddings, i, t, guidance_scale):
    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = 0.0

    latent_model_input = torch.cat([latents] * 2)

    noise_pred = pipe.unet(
        latent_model_input, t, encoder_hidden_states=text_embeddings
    )["sample"]

    noise_pred_uncond, noise_pred_cond = noise_pred[0].unsqueeze(0), noise_pred[
        1:
    ].mean(dim=0, keepdim=True)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )
    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        latents = pipe.scheduler.step(noise_pred, i, latents, **extra_step_kwargs)[
            "prev_sample"
        ]
    else:
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
            "prev_sample"
        ]

    return latents


@torch.no_grad()
def diffuse_latents(
    pipe,
    cond_embeddings,
    cond_latents,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=None,
    offset=1,
    eta=0.0,
):

    batch_size, n, h, w = cond_latents.shape
    # set timesteps
    accepts_offset = "offset" in set(
        inspect.signature(pipe.scheduler.set_timesteps).parameters.keys()
    )
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = offset
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

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

        latents = denoise(latents, pipe, text_embeddings, i, t, guidance_scale)

    return latents


@torch.no_grad()
def latents_to_image(pipe, latents):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    images = pipe.vae.decode(latents).sample
    images = postprocess(images)

    return images


@torch.no_grad()
def get_text_embeddings(key_frames, pipe):
    frame_start = key_frames[0][0]

    if frame_start != 0:
        key_frames.append([0, ""])

    output = {}
    for i, prompt in key_frames:
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids.to(pipe.text_encoder.device)
        text_embeddings = pipe.text_encoder(text_inputs)[0]
        output[i] = text_embeddings.cpu()

    for start, end in zip(key_frames, key_frames[1:]):
        start_frame_idx = start[0]
        end_frame_idx = end[0]
        weights = torch.linspace(0, 1.0, steps=(end_frame_idx - start_frame_idx))

        start_embedding = output[start_frame_idx]
        end_embedding = output[end_frame_idx]

        for i in range(start_frame_idx + 1, end_frame_idx):
            weight = weights[i - start_frame_idx]
            embedding = slerp(weight.item(), start_embedding, end_embedding)
            output[i] = embedding

    return output


@torch.no_grad()
def get_init_latents(key_frames, pipe, height, width, generator):
    frame_start = key_frames[0][0]

    if frame_start != 0:
        key_frames.append([0, ""])

    output = {}
    for i, prompt in key_frames:
        output[i] = torch.randn(
            (pipe.unet.in_channels, height // 8, width // 8),
            device=pipe.device,
            generator=generator,
        ).cpu()

    for start, end in zip(key_frames, key_frames[1:]):
        start_frame_idx = start[0]
        end_frame_idx = end[0]
        weights = torch.linspace(0, 1.0, steps=(end_frame_idx - start_frame_idx))

        start_embedding = output[start_frame_idx]
        end_embedding = output[end_frame_idx]

        for i in range(start_frame_idx + 1, end_frame_idx):
            weight = weights[i - start_frame_idx]
            embedding = slerp(weight.item(), start_embedding, end_embedding)
            output[i] = embedding

    return output


def run(
    text_prompt_inputs,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    fps=24,
    scheduler="pndms",
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

    generator = torch.Generator(device=device).manual_seed(int(seed))

    key_frames = parse_key_frames(text_prompt_inputs)
    max_frames = max(key_frames, key=lambda x: x[0])[0]

    init_latents = get_init_latents(key_frames, pipe, 512, 512, generator)
    text_embeddings = get_text_embeddings(key_frames, pipe)

    output_frames = []
    for frame_idx in tqdm(range(max_frames + 1), total=max_frames + 1):
        init_latent = init_latents[frame_idx]
        text_embedding = text_embeddings[frame_idx]

        init_latent = init_latent.to(device)
        text_embedding = text_embedding.to(device)

        with autocast("cuda"):
            latents = diffuse_latents(
                pipe,
                text_embedding,
                init_latent,
                num_inference_steps,
                guidance_scale,
                generator,
            )
            image_tensors = latents_to_image(pipe, latents)

        img_save_path = f"{run_path}/{frame_idx:04d}.png"
        images = numpy_to_pil(image_tensors.numpy())
        images[0].save(img_save_path)

        output_frames.append(img_save_path)

        if experiment:
            experiment.log_image(img_save_path, image_name=f"{frame_idx:04d}")

    output_filename = f"{run_path}/output.gif"
    save_gif(frames=output_frames, filename=output_filename, fps=fps)

    if experiment:
        experiment.log_asset(output_filename, ftype="image")

    return output_filename


if __name__ == "__main__":
    typer.run(run)
