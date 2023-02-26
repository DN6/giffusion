import importlib
import pathlib

import gradio as gr
import torch
from PIL import Image

from generate import run
from utils import (
    get_audio_key_frame_information,
    get_video_frame_information,
    load_video_frames,
    to_pil_image,
)

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


def load_pipeline(model_name, pipeline_name, pipe):
    try:
        # clear existing model from memory
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipe_cls = getattr(importlib.import_module("diffusers"), pipeline_name)
        pipe = pipe_cls.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(device)

        return pipe, f"Successfully loaded Pipeline: {pipeline_name} with {model_name}"

    except Exception:
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


def display_sine_parameters(value):
    if value == "sine":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


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
    scheduler,
    use_fixed_latent,
    use_prompt_embeds,
    num_latent_channels,
    audio_input,
    audio_component,
    mel_spectogram_reduce,
    image_input,
    video_input,
    output_format,
    model_name,
    additional_pipeline_arguments,
    interpolation_type,
    frequencies,
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
        scheduler=scheduler,
        use_fixed_latent=use_fixed_latent,
        use_prompt_embeds=use_prompt_embeds,
        num_latent_channels=int(num_latent_channels),
        audio_input=audio_input,
        audio_component=audio_component,
        mel_spectogram_reduce=mel_spectogram_reduce,
        image_input=image_input,
        video_input=video_input,
        output_format=output_format,
        model_name=model_name,
        additional_pipeline_arguments=additional_pipeline_arguments,
        interpolation_type=interpolation_type,
        frequencies=frequencies,
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
                            label="Model Name", value="DiffusionPipeline"
                        )
                    with gr.Column():
                        with gr.Row():
                            load_pipeline_btn = gr.Button(value="Load Pipeline")
                        with gr.Row():
                            load_message = gr.Markdown()

            with gr.Accordion("Output Settings: Set output file format and FPS"):
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
                    value=30,
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
                        "repaint",
                    ],
                    value="deis",
                    label="Scheduler",
                )
                image_height = gr.Number(value=512, label="Image Height")
                image_width = gr.Number(value=512, label="Image Width")
                num_latent_channels = gr.Number(
                    value=4, label="Number of Latent Channels"
                )

                with gr.Accordion("Additional Pipeline Arguments", open=False):
                    additional_pipeline_arguments = gr.Textbox(
                        label="",
                        value="{}",
                        interactive=True,
                        lines=4,
                        placeholder="A dictionary of key word arguments to pass to the pipeline",
                    )

            with gr.Accordion("Animation Settings", open=False):
                interpolation_type = gr.Dropdown(["linear", "sine"], value="linear")
                frequencies = gr.Textbox("", label="Frequencies", visible=False)
                interpolation_type.change(
                    display_sine_parameters, [interpolation_type], [frequencies]
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
            submit = gr.Button(
                label="Submit",
                value="Create",
                variant="primary",
                elem_id="submit-btn",
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
                    value="""low resolution""",
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

            with gr.Accordion("Resample Output", open=False):
                with gr.Accordion("Send to Image Input", open=False):
                    frame_id = gr.Number(value=0, label="Frame ID")
                    send_to_image_input_btn = gr.Button("Send to Image Input")
                with gr.Accordion("Send to Video Input", open=False):
                    send_to_video_input_btn = gr.Button("Send to Video Input")

    pipe = gr.State()

    load_pipeline_btn.click(
        load_pipeline, [model_name, pipeline_name, pipe], [pipe, load_message]
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

    submit.click(
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
            scheduler,
            use_fixed_latent,
            use_prompt_embeds,
            num_latent_channels,
            audio_input,
            audio_component,
            mel_spectogram_reduce,
            image_input,
            video_input,
            output_format,
            model_name,
            additional_pipeline_arguments,
            interpolation_type,
            frequencies,
        ],
        outputs=output,
    )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True)
