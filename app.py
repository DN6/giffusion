import gradio as gr

from generate import run
from utils import sync_prompts_to_audio, sync_prompts_to_video

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


def generate_prompt(fps):
    prompts = prompt_generator("")
    prompts = [
        f"{idx * fps}: {prompt}" for idx, prompt in enumerate(prompts.split("\n"))
    ]
    prompts = "\n".join(prompts)

    return prompts


def _sync_prompts_to_audio(text_inputs, audio_input, fps):
    key_frames = sync_prompts_to_audio(text_inputs, audio_input, fps)
    prompts = [f"{frame_idx}: {prompt}" for frame_idx, prompt in key_frames]
    prompts = "\n".join(prompts)

    return prompts


def _sync_prompts_to_video(text_inputs, video_input):
    key_frames = sync_prompts_to_video(text_inputs, video_input)
    prompts = [f"{frame_idx}: {prompt}" for frame_idx, prompt in key_frames]
    prompts = "\n".join(prompts)

    return prompts


def predict(
    text_prompt_input,
    num_iteration_steps,
    guidance_scale,
    strength,
    seed,
    fps,
    scheduler,
    use_fixed_latent,
    audio_input,
    video_input,
    output_format,
):
    output = run(
        text_prompt_inputs=text_prompt_input,
        num_inference_steps=num_iteration_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
        fps=fps,
        scheduler=scheduler,
        use_fixed_latent=use_fixed_latent,
        audio_input=audio_input,
        video_input=video_input,
        output_format=output_format,
    )

    return output


demo = gr.Blocks(css="css/styles.css")

with demo:
    with gr.Row():
        text_prompt_input = gr.Textbox(
            lines=20,
            value="""0: A corgi in the clouds\n60: A corgi in the ocean""",
            label="Text Prompts",
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
                strength = gr.Slider(
                    0, 1.0, step=0.1, value=0.5, label="Image Strength"
                )
                scheduler = gr.Dropdown(
                    ["klms", "ddim", "pndms"],
                    value="pndms",
                    label="Scheduler",
                )
            with gr.TabItem("Audio Input Settings"):
                audio_input = gr.Audio(label="Audio Input", type="filepath")
                sync_audio_btn = gr.Button(value="Sync Prompts to Audio")

            with gr.TabItem("Video Input Settings"):
                video_input = gr.Video(label="Video Input")
                sync_video_btn = gr.Button(value="Sync Prompts to Video")

    with gr.Row():
        output_format = gr.Radio(["gif", "mp4"], value="mp4", label="Output Format")
        fps = gr.Slider(10, 60, step=1, value=10, label="Output Frame Rate")

    with gr.Row():
        generate = gr.Button(
            value="Give me some inspiration!",
            variant="secondary",
            elem_id="prompt-generator-btn",
        )
        generate.click(generate_prompt, inputs=[fps], outputs=text_prompt_input)
        submit = gr.Button(
            label="Submit",
            value="Create",
            variant="primary",
            elem_id="submit-btn",
        )

    with gr.Row(elem_id="output-row"):
        output = gr.Video(label="Model Output", elem_id="output")

    sync_audio_btn.click(
        _sync_prompts_to_audio,
        inputs=[text_prompt_input, audio_input, fps],
        outputs=[text_prompt_input],
    )
    sync_video_btn.click(
        _sync_prompts_to_video,
        inputs=[text_prompt_input, video_input],
        outputs=[text_prompt_input],
    )

    submit.click(
        fn=predict,
        inputs=[
            text_prompt_input,
            num_iteration_steps,
            guidance_scale,
            strength,
            seed,
            fps,
            scheduler,
            use_fixed_latent,
            audio_input,
            video_input,
            output_format,
        ],
        outputs=output,
    )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
