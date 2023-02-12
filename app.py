import importlib

import gradio as gr
import torch

from generate import run
from utils import get_audio_key_frame_information, get_video_frame_information

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


def load_pipeline(model_name, pipeline_name):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipe_cls = getattr(importlib.import_module("diffusers"), pipeline_name)
        pipe = pipe_cls.from_pretrained(
            model_name, use_auth_token=True, torch_dtype=torch.float16
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

    return "\n".join(["0: ", f"{max_frames - 1}: "]), gr.update(value=fps)


def predict(
    pipe,
    text_prompt_input,
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
    audio_input,
    audio_component,
    image_input,
    video_input,
    output_format,
    model_name,
):
    output = run(
        pipe=pipe,
        text_prompt_inputs=text_prompt_input,
        num_inference_steps=num_iteration_steps,
        height=image_height,
        width=image_width,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=int(seed),
        batch_size=batch_size,
        fps=fps,
        scheduler=scheduler,
        use_fixed_latent=use_fixed_latent,
        audio_input=audio_input,
        audio_component=audio_component,
        image_input=image_input,
        video_input=video_input,
        output_format=output_format,
        model_name=model_name,
    )

    return output


demo = gr.Blocks()

with demo:
    gr.Markdown("# GIFfusion 💥")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    model_name = gr.Textbox(
                        label="Model Name", value="runwayml/stable-diffusion-v1-5"
                    )
                    pipeline_name = gr.Textbox(
                        label="Model Name", value="StableDiffusionPipeline"
                    )
                with gr.Column():
                    with gr.Row():
                        load_pipeline_btn = gr.Button(value="Load Pipeline")
                    with gr.Row():
                        load_message = gr.Markdown()

                pipe = gr.State()

            with gr.Accordion("Output Settings"):
                with gr.Row():
                    output_format = gr.Radio(
                        ["gif", "mp4"], value="mp4", label="Output Format"
                    )
                    fps = gr.Slider(10, 60, step=1, value=10, label="Output Frame Rate")

            with gr.Row():
                text_prompt_input = gr.Textbox(
                    lines=10,
                    value="""0: A corgi in the clouds\n60: A corgi in the ocean""",
                    label="Text Prompts",
                    interactive=True,
                )

            with gr.Accordion("Inspiration Settings"):
                with gr.Row():
                    topics = gr.Textbox(lines=1, value="", label="Inspiration Topics")

                with gr.Row():
                    generate = gr.Button(
                        value="Give me some inspiration!",
                        variant="secondary",
                        elem_id="prompt-generator-btn",
                    )

            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("Diffusion Settings"):
                        with gr.Row():
                            use_fixed_latent = gr.Checkbox(
                                label="Use Fixed Init Latent"
                            )
                            seed = gr.Number(value=42, label="Numerical Seed")
                            num_iteration_steps = gr.Slider(
                                10,
                                1000,
                                step=10,
                                value=50,
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
                                ["klms", "ddim", "pndms"],
                                value="pndms",
                                label="Scheduler",
                            )
                            batch_size = gr.Slider(
                                1, 64, step=1, value=1, label="Batch Size"
                            )
                        with gr.Row():
                            with gr.Column():
                                image_height = gr.Number(
                                    value=512, label="Image Height"
                                )
                            with gr.Column():
                                image_width = gr.Number(value=512, label="Image Width")

                    with gr.TabItem("Audio Input Settings"):
                        audio_input = gr.Audio(label="Audio Input", type="filepath")
                        audio_component = gr.Radio(
                            ["percussive", "harmonic", "both"],
                            value="percussive",
                            label="Audio Component",
                        )
                        audio_info_btn = gr.Button(value="Get Key Frame Information")

                    with gr.TabItem("Video Input Settings"):
                        with gr.Row():
                            video_input = gr.Video(label="Video Input")
                            video_info_btn = gr.Button(value="Get Key Frame Infomation")

                    with gr.TabItem("Image Input Settings"):
                        with gr.Row():
                            image_input = gr.Image(label="Initial Image", type="pil")

            with gr.Row():
                submit = gr.Button(
                    label="Submit",
                    value="Create",
                    variant="primary",
                    elem_id="submit-btn",
                )

        with gr.Column(elem_id="output"):
            output = gr.Video(label="Model Output", elem_id="output")

    load_pipeline_btn.click(
        load_pipeline, [model_name, pipeline_name], [pipe, load_message]
    )

    generate.click(
        generate_prompt,
        inputs=[text_prompt_input, fps, topics],
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

    submit.click(
        fn=predict,
        inputs=[
            pipe,
            text_prompt_input,
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
            audio_input,
            audio_component,
            image_input,
            video_input,
            output_format,
            model_name,
        ],
        outputs=output,
    )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
