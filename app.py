import gradio as gr

from generate import run
from utils import get_audio_key_frame_information, get_video_frame_information

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


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
    text_prompt_input,
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
    video_input,
    output_format,
):
    output = run(
        text_prompt_inputs=text_prompt_input,
        num_inference_steps=num_iteration_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
        batch_size=batch_size,
        fps=fps,
        scheduler=scheduler,
        use_fixed_latent=use_fixed_latent,
        audio_input=audio_input,
        audio_component=audio_component,
        video_input=video_input,
        output_format=output_format,
    )

    return output


demo = gr.Blocks(css="css/styles.css")

with demo:
    gr.Markdown("# GIFfusion 💥")
    with gr.Row():
        with gr.Column():
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
                        use_fixed_latent = gr.Checkbox(label="Use Fixed Init Latent")
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
                        scheduler = gr.Dropdown(
                            ["klms", "ddim", "pndms"],
                            value="pndms",
                            label="Scheduler",
                        )
                        batch_size = gr.Slider(
                            1, 16, step=1, value=1, label="Batch Size"
                        )

                    with gr.TabItem("Audio Input Settings"):
                        audio_input = gr.Audio(label="Audio Input", type="filepath")
                        audio_info_btn = gr.Button(value="Get Key Frame Information")
                        audio_component = gr.Radio(
                            ["percussive", "harmonic", "both"],
                            value="percussive",
                            label="Audio Component",
                        )

                    with gr.TabItem("Video Input Settings"):
                        video_input = gr.Video(label="Video Input")
                        strength = gr.Slider(
                            0, 1.0, step=0.1, value=0.5, label="Image Strength"
                        )
                        video_info_btn = gr.Button(value="Get Key Frame Infomation")

            with gr.Row():
                submit = gr.Button(
                    label="Submit",
                    value="Create",
                    variant="primary",
                    elem_id="submit-btn",
                )

        with gr.Column(elem_id="output"):
            output = gr.Video(label="Model Output", elem_id="output")

    generate.click(generate_prompt, inputs=[fps, topics], outputs=text_prompt_input)
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
            text_prompt_input,
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
            video_input,
            output_format,
        ],
        outputs=output,
    )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
