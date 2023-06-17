import importlib
import os
import pathlib

import gradio as gr
import torch
from controlnet_aux.processor import MODELS as CONTROLNET_PROCESSORS
from PIL import Image

from generate import run
from utils import (
    get_audio_key_frame_information,
    get_video_frame_information,
    load_video_frames,
    set_xformers,
    to_pil_image,
)

DEBUG = os.getenv("DEBUG_MODE", "false").lower() == "true"
OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH", "generated")
MODEL_PATH = os.getenv("MODEL_PATH", "models")

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

USE_XFORMERS = set_xformers()
CONTROLNET_PROCESSORS = ["None", "inpaint"] + list(CONTROLNET_PROCESSORS.keys())

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


def load_pipeline(model_name, pipeline_name, controlnet, pipe):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # clear existing model from memory
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()

        if controlnet:
            from diffusers import ControlNetModel

            controlnet_model = ControlNetModel.from_pretrained(
                controlnet, torch_dtype=torch.float16, cache_dir=MODEL_PATH
            )
            pipeline_name = "StableDiffusionControlNetPipeline"

            pipe_cls = getattr(importlib.import_module("diffusers"), pipeline_name)
            pipe = pipe_cls.from_pretrained(
                model_name,
                use_auth_token=True,
                torch_dtype=torch.float16,
                safety_checker=None,
                controlnet=controlnet_model,
                cache_dir=MODEL_PATH,
            )

        else:
            pipe_cls = getattr(importlib.import_module("diffusers"), pipeline_name)
            pipe = pipe_cls.from_pretrained(
                model_name,
                use_auth_token=True,
                torch_dtype=torch.float16,
                safety_checker=None,
                cache_dir=MODEL_PATH,
            )

        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if USE_XFORMERS:
            pipe.enable_xformers_memory_efficient_attention()

        return pipe, f"Successfully loaded Pipeline: {pipeline_name} with {model_name}"

    except Exception as e:
        print(e)
        return None, f"Failed to Load Pipeline: {pipeline_name} with {model_name}"


def generate_prompt(fps, topics=""):
    prompts = prompt_generator(topics)
    prompts = [
        f"{idx * fps}: {prompt}" for idx, prompt in enumerate(prompts.split("\n"))
    ]
    prompts = "\n".join(prompts)

    return prompts


def _get_audio_key_frame_information(audio_input, fps, audio_component):
    key_frames = get_audio_key_frame_information(audio_input, fps, audio_component)

    return "\n".join([f"{kf}: timestamp: {kf / fps:.2f}" for kf in key_frames])


def _get_video_frame_information(video_input):
    max_frames, fps = get_video_frame_information(video_input)

    return "\n".join(["0: ", f"{max_frames - 1}: "]), gr.update(value=int(fps))


def send_to_image_input(output, frame_id):
    extension = pathlib.Path(output).suffix
    if extension == "gif":
        image = Image.open(output)
        output_image = image.seek(frame_id)
    else:
        frames, _, _ = load_video_frames(output)
        output_image = to_pil_image(frames[int(frame_id)])

    return output_image


def send_to_video_input(video):
    return video


def predict(
    pipe,
    text_prompt_input,
    negative_prompt_input,
    image_width,
    image_height,
    num_iteration_steps,
    guidance_scale,
    strength,
    seed,
    batch_size,
    fps,
    use_default_scheduler,
    scheduler,
    scheduler_kwargs,
    use_fixed_latent,
    use_prompt_embeds,
    num_latent_channels,
    audio_input,
    audio_component,
    mel_spectogram_reduce,
    image_input,
    video_input,
    video_use_pil_format,
    output_format,
    model_name,
    controlnet_name,
    additional_pipeline_arguments,
    interpolation_type,
    interpolation_args,
    zoom,
    translate_x,
    translate_y,
    angle,
    padding_mode,
    coherence_scale,
    coherence_alpha,
    coherence_steps,
    apply_color_matching,
    preprocessing_type,
):
    output = run(
        pipe=pipe,
        text_prompt_inputs=text_prompt_input,
        negative_prompt_inputs=negative_prompt_input,
        num_inference_steps=int(num_iteration_steps),
        height=int(image_height),
        width=int(image_width),
        guidance_scale=guidance_scale,
        strength=strength,
        seed=int(seed),
        batch_size=int(batch_size),
        fps=int(fps),
        use_default_scheduler=use_default_scheduler,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        use_fixed_latent=use_fixed_latent,
        use_prompt_embeds=use_prompt_embeds,
        num_latent_channels=int(num_latent_channels),
        audio_input=audio_input,
        audio_component=audio_component,
        mel_spectogram_reduce=mel_spectogram_reduce,
        image_input=image_input,
        video_input=video_input,
        video_use_pil_format=video_use_pil_format,
        output_format=output_format,
        model_name=model_name,
        controlnet_name=controlnet_name,
        additional_pipeline_arguments=additional_pipeline_arguments,
        interpolation_type=interpolation_type,
        interpolation_args=interpolation_args,
        zoom=zoom,
        translate_x=translate_x,
        translate_y=translate_y,
        angle=angle,
        padding_mode=padding_mode,
        coherence_scale=coherence_scale,
        coherence_alpha=coherence_alpha,
        coherence_steps=int(coherence_steps),
        apply_color_matching=apply_color_matching,
        preprocess=preprocessing_type,
    )

    return output


demo = gr.Blocks()

with demo:
    gr.Markdown("# GIFfusion 💥")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Accordion("Pipeline Settings: Load Models and Pipelines"):
                    with gr.Column():
                        model_name = gr.Textbox(
                            label="Model Name", value="runwayml/stable-diffusion-v1-5"
                        )
                        pipeline_name = gr.Textbox(
                            label="Pipeline Name", value="DiffusionPipeline"
                        )
                        controlnet = gr.Textbox(label="ControlNet Checkpoint")
                    with gr.Column():
                        with gr.Row():
                            load_pipeline_btn = gr.Button(value="Load Pipeline")
                        with gr.Row():
                            load_message = gr.Markdown()

            with gr.Accordion(
                "Output Settings: Set output file format and FPS", open=False
            ):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            output_format = gr.Dropdown(
                                ["gif", "mp4"], value="mp4", label="Output Format"
                            )
                        with gr.Row():
                            fps = gr.Slider(
                                10, 60, step=1, value=10, label="Output Frame Rate"
                            )

            with gr.Accordion("Diffusion Settings", open=False):
                with gr.Tab("Diffusion"):
                    use_fixed_latent = gr.Checkbox(label="Use Fixed Init Latent")
                    use_prompt_embeds = gr.Checkbox(
                        label="Use Prompt Embeds", value=True, interactive=True
                    )
                    seed = gr.Number(value=42, label="Numerical Seed")
                    batch_size = gr.Slider(1, 64, step=1, value=1, label="Batch Size")
                    num_iteration_steps = gr.Slider(
                        10,
                        1000,
                        step=10,
                        value=20,
                        label="Number of Iteration Steps",
                    )
                    guidance_scale = gr.Slider(
                        0.5,
                        20,
                        step=0.5,
                        value=7.5,
                        label="Classifier Free Guidance Scale",
                    )
                    strength = gr.Slider(
                        0, 1.0, step=0.1, value=0.5, label="Image Strength"
                    )
                    num_latent_channels = gr.Number(
                        value=4, label="Number of Latent Channels"
                    )
                    image_height = gr.Number(value=512, label="Image Height")
                    image_width = gr.Number(value=512, label="Image Width")

                with gr.Tab("Scheduler"):
                    use_default_scheduler = gr.Checkbox(
                        label="Use Default Pipeline Scheduler"
                    )
                    scheduler = gr.Dropdown(
                        [
                            "klms",
                            "ddim",
                            "ddpm",
                            "pndms",
                            "dpm",
                            "dpm_ads",
                            "deis",
                            "euler",
                            "euler_ads",
                            "unipc",
                        ],
                        value="deis",
                        label="Scheduler",
                    )
                    scheduler_kwargs = gr.Textbox(
                        label="Scheduler Arguments",
                        value="{}",
                    )

                with gr.Tab("Pipeline"):
                    additional_pipeline_arguments = gr.Textbox(
                        label="Additional Pipeline Arguments",
                        value="{}",
                        interactive=True,
                        lines=4,
                        placeholder="A dictionary of key word arguments to pass to the pipeline",
                    )

            with gr.Accordion("Animation Settings", open=False):
                with gr.Tab("Interpolation"):
                    interpolation_type = gr.Dropdown(
                        ["linear", "sine", "curve"],
                        value="linear",
                        label="Interpolation Type",
                    )
                    interpolation_args = gr.Textbox(
                        "", label="Interpolation Parameters", visible=True
                    )
                with gr.Tab("Motion"):
                    zoom = gr.Textbox("", label="Zoom")
                    translate_x = gr.Textbox("", label="Translate_X")
                    translate_y = gr.Textbox("", label="Translate_Y")
                    angle = gr.Textbox("", label="Angle")
                    padding_mode = gr.Dropdown(
                        ["zero", "border", "reflection"],
                        label="Padding Mode",
                        value="border",
                    )

                with gr.Tab("Coherence"):
                    coherence_scale = gr.Slider(
                        0, 10000, step=50, value=0, label="Coherence Scale"
                    )
                    coherence_alpha = gr.Slider(
                        0, 1.0, step=0.1, value=0.1, label="Coherence Alpha"
                    )
                    coherence_steps = gr.Slider(
                        0, 100, step=1, value=1, label="Coherence Steps"
                    )
                    noise_schedule = gr.Textbox(
                        label="Noise Schedule", value="0:(0.01)", interactive=True
                    )
                with gr.Tab("Image Color"):
                    apply_color_matching = gr.Checkbox(
                        label="Use Color Matching", value=False, interactive=True
                    )

            with gr.Accordion("Inspiration Settings", open=False):
                with gr.Row():
                    topics = gr.Textbox(lines=1, value="", label="Inspiration Topics")

                with gr.Row():
                    generate_btn = gr.Button(
                        value="Give me some inspiration!",
                        variant="secondary",
                        elem_id="prompt-generator-btn",
                    )

        with gr.Column(elem_id="output", scale=2):
            output = gr.Video(label="Model Output", elem_id="output")
            with gr.Row():
                submit = gr.Button(
                    label="Submit",
                    value="Create",
                    variant="primary",
                    elem_id="submit-btn",
                )
                stop = gr.Button(
                    label="Submit",
                    value="Stop",
                    elem_id="stop-btn",
                )
            with gr.Row():
                text_prompt_input = gr.Textbox(
                    lines=10,
                    value="""0: A corgi in the clouds\n60: A corgi in the ocean""",
                    label="Text Prompts",
                    interactive=True,
                )
            with gr.Row():
                negative_prompt_input = gr.Textbox(
                    value="""low resolution, blurry, worst quality, jpeg artifacts""",
                    label="Negative Prompts",
                    interactive=True,
                )

        with gr.Column(scale=1):
            with gr.Accordion("Image Input", open=False):
                image_input = gr.Image(label="Initial Image", type="pil")

            with gr.Accordion("Audio Input", open=False):
                audio_input = gr.Audio(label="Audio Input", type="filepath")
                audio_component = gr.Dropdown(
                    ["percussive", "harmonic", "both"],
                    value="percussive",
                    label="Audio Component",
                )
                audio_info_btn = gr.Button(value="Get Key Frame Information")
                mel_spectogram_reduce = gr.Dropdown(
                    ["mean", "median", "max"],
                    label="Mel Spectrogram Reduction",
                    value="max",
                )

            with gr.Accordion("Video Input", open=False):
                video_input = gr.Video(label="Video Input")
                video_info_btn = gr.Button(value="Get Key Frame Infomation")
                video_use_pil_format = gr.Checkbox(label="Use PIL Format", value=True)

            with gr.Accordion("Resample Output", open=False):
                with gr.Accordion("Send to Image Input", open=False):
                    frame_id = gr.Number(value=0, label="Frame ID")
                    send_to_image_input_btn = gr.Button("Send to Image Input")
                with gr.Accordion("Send to Video Input", open=False):
                    send_to_video_input_btn = gr.Button("Send to Video Input")

            with gr.Accordion("Controlnet Preprocessing Settings", open=False):
                preprocessing_type = gr.Dropdown(
                    CONTROLNET_PROCESSORS,
                    value="None",
                    label="Preprocessing",
                )

    pipe = gr.State()

    load_pipeline_btn.click(
        load_pipeline,
        [model_name, pipeline_name, controlnet, pipe],
        [pipe, load_message],
    )

    generate_btn.click(
        generate_prompt,
        inputs=[fps, topics],
        outputs=text_prompt_input,
    )
    audio_info_btn.click(
        _get_audio_key_frame_information,
        inputs=[audio_input, fps, audio_component],
        outputs=[text_prompt_input],
    )
    video_info_btn.click(
        _get_video_frame_information,
        inputs=[video_input],
        outputs=[text_prompt_input, fps],
    )

    send_to_image_input_btn.click(send_to_image_input, [output, frame_id], image_input)
    send_to_video_input_btn.click(send_to_video_input, [output], [video_input])

    submit_event = submit.click(
        fn=predict,
        inputs=[
            pipe,
            text_prompt_input,
            negative_prompt_input,
            image_width,
            image_height,
            num_iteration_steps,
            guidance_scale,
            strength,
            seed,
            batch_size,
            fps,
            use_default_scheduler,
            scheduler,
            scheduler_kwargs,
            use_fixed_latent,
            use_prompt_embeds,
            num_latent_channels,
            audio_input,
            audio_component,
            mel_spectogram_reduce,
            image_input,
            video_input,
            video_use_pil_format,
            output_format,
            model_name,
            controlnet,
            additional_pipeline_arguments,
            interpolation_type,
            interpolation_args,
            zoom,
            translate_x,
            translate_y,
            angle,
            padding_mode,
            coherence_scale,
            coherence_alpha,
            coherence_steps,
            apply_color_matching,
            preprocessing_type,
        ],
        outputs=output,
    )
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])

if __name__ == "__main__":
    demo.queue(concurrency_count=2)
    demo.launch(share=True, debug=DEBUG)
