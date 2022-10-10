import gradio as gr

from generate import run

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")


def generate_prompt(fps):
    prompts = prompt_generator("")
    prompts = [
        f"{idx * fps}: {prompt}" for idx, prompt in enumerate(prompts.split("\n"))
    ]
    prompts = "\n".join(prompts)

    return prompts


def predict(
    text_prompt_input,
    num_iteration_steps,
    guidance_scale,
    seed,
    fps,
    scheduler,
    use_fixed_latent,
    audio_input,
    output_format,
):
    output = run(
        text_prompt_input,
        num_iteration_steps,
        guidance_scale,
        seed,
        fps,
        scheduler,
        use_fixed_latent,
        audio_input,
        output_format,
    )

    return output


demo = gr.Blocks(css="css/styles.css")

with demo:
    with gr.Row():
        text_prompt_input = gr.Textbox(
            lines=10,
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
                scheduler = gr.Dropdown(
                    ["klms", "ddim", "pndms"],
                    value="pndms",
                    label="Scheduler",
                )
            with gr.TabItem("Audio Settings"):
                audio_input = gr.Audio(label="Audio Input")

            with gr.TabItem("Output Settings"):
                output_format = gr.Radio(["gif", "mp4"], value="gif")
                fps = gr.Slider(10, 60, step=1, value=10, label="Output GIF Frame Rate")

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

    submit.click(
        fn=predict,
        inputs=[
            text_prompt_input,
            num_iteration_steps,
            guidance_scale,
            seed,
            fps,
            scheduler,
            use_fixed_latent,
            audio_input,
            output_format,
        ],
        outputs=output,
    )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
