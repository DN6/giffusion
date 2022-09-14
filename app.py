import gradio as gr

from generate import run


def predict(
    text_prompt_input,
    num_iteration_steps,
    guidance_scale,
    seed,
    fps,
    scheduler,
):
    output = run(
        text_prompt_input,
        num_iteration_steps,
        guidance_scale,
        seed,
        fps,
        scheduler,
    )

    return output


def set_output(output_option):
    if output_option == "mp4":
        return gr.Video

    return gr.Image


demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("Diffusion"):
                with gr.Row():
                    text_prompt_input = gr.Textbox(
                        lines=10,
                        value="""0: A corgi in the clouds\n60: A corgi in the ocean""",
                        label="Text Prompts",
                    )
                with gr.Row():
                    seed = gr.Number(value=42, label="Numerical Seed")
                    num_iteration_steps = gr.Slider(
                        10, 1000, step=10, value=50, label="Number of Iteration Steps"
                    )
                    guidance_scale = gr.Slider(
                        0.5,
                        20,
                        step=0.5,
                        value=7.5,
                        label="Classifier Free Guidance Scale",
                    )
                    fps = gr.Slider(
                        10, 60, step=1, value=24, label="Output GIF Frames Rate"
                    )
                    scheduler = gr.Dropdown(
                        ["klms", "ddim", "pndms"], value="pndms", label="Scheduler"
                    )

    with gr.Row():
        output = gr.Image(label="Model Output", elem_id="output")

    with gr.Row():
        submit = gr.Button(label="Submit", variant="primary")
        submit.click(
            fn=predict,
            inputs=[
                text_prompt_input,
                num_iteration_steps,
                guidance_scale,
                seed,
                fps,
                scheduler,
            ],
            outputs=output,
        )

demo.queue(concurrency_count=2)

if __name__ == "__main__":
    demo.launch(share=True)
