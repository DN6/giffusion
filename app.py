import importlib
import os
import pathlib

import gradio as gr
import torch
from controlnet_aux.processor import MODELS as CONTROLNET_PROCESSORS
from PIL import Image
from wonderwords import RandomWord

from generate import run
from session import load_session, save_session
from utils import (
    ToPILImage,
    get_audio_key_frame_information,
    get_video_frame_information,
    load_video_frames,
    save_gif,
    save_video,
    set_xformers,
)

AUTOSAVE = os.getenv("GIFFUSION_AUTO_SAVE", True)
ORG_ID = os.getenv("ORG_ID", None)
REPO_ID = os.getenv("REPO_ID", "giffusion")
DEBUG = os.getenv("DEBUG_MODE", "false").lower() == "true"
OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH", "generated")
MODEL_PATH = os.getenv("MODEL_PATH", "models")

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

USE_XFORMERS = set_xformers()
CONTROLNET_PROCESSORS = ["no-processing"] + list(CONTROLNET_PROCESSORS.keys())

prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")
wordgen = RandomWord()


def load_pipeline(
    model_name, pipeline_name, controlnet, adapter, lora, custom_pipeline, pipe
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # clear existing model from memory
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()

        success_message = (
            f"Successfully loaded Pipeline: {pipeline_name} with {model_name}"
        )
        pipe_cls = getattr(importlib.import_module("diffusers"), pipeline_name)

        if controlnet and adapter:
            raise gr.Error("Cannot load both ControlNet and T2IAdapter")

        if controlnet:
            from diffusers import ControlNetModel

            controlnets = [controlnet.strip() for controlnet in controlnet.split(",")]

            # temporary solution to multicontrolnet issue in sdxl
            if len(controlnets) == 1:
                controlnet_models = ControlNetModel.from_pretrained(
                    controlnets[0], torch_dtype=torch.float16, cache_dir=MODEL_PATH
                )
            else:
                controlnet_models = [
                    ControlNetModel.from_pretrained(
                        controlnet, torch_dtype=torch.float16, cache_dir=MODEL_PATH
                    )
                    for controlnet in controlnets
                ]

            pipe = pipe_cls.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                controlnet=controlnet_models,
                cache_dir=MODEL_PATH,
                custom_pipeline=custom_pipeline if custom_pipeline else None,
            )
            success_message = f"Successfully loaded Pipeline: {pipeline_name} with {model_name} and {controlnets}"

        elif adapter:
            from diffusers import T2IAdapter

            adapters = [adapter.strip() for adapter in adapter.split(",")]

            if len(adapters) == 1:
                adapter_models = T2IAdapter.from_pretrained(
                    adapters[0], torch_dtype=torch.float16, cache_dir=MODEL_PATH
                )
            else:
                adapter_models = [
                    T2IAdapter.from_pretrained(
                        adapter, torch_dtype=torch.float16, cache_dir=MODEL_PATH
                    )
                    for adapter in adapters
                ]

            pipe = pipe_cls.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                adapter=adapter_models,
                cache_dir=MODEL_PATH,
                custom_pipeline=custom_pipeline if custom_pipeline else None,
            )
            success_message = f"Successfully loaded Pipeline: {pipeline_name} with {model_name} and {adapters}"

        else:
            pipe = pipe_cls.from_pretrained(
                model_name,
                use_auth_token=True,
                torch_dtype=torch.float16,
                safety_checker=None,
                cache_dir=MODEL_PATH,
                custom_pipeline=custom_pipeline if custom_pipeline else None,
            )

        if lora:
            pipe.load_lora_weights(lora)

        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if USE_XFORMERS:
            pipe.enable_xformers_memory_efficient_attention()

        return pipe, success_message

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
        output_image = ToPILImage()(frames[int(frame_id)])

    return output_image


def send_to_video_input(video):
    return video


def send_to_video_output(state):
    video_output = next(state)
    return video_output


def create_run_path():
    run_name = f"{wordgen.word(include_parts_of_speech=['adjectives'])}-{wordgen.word(include_parts_of_speech=['nouns'])}"
    run_path = os.path.join(OUTPUT_BASE_PATH, run_name)
    os.makedirs(run_path, exist_ok=True)

    return run_path


def _save_session(org_id, repo_id, run_path, session_name):
    save_session(org_id, repo_id, run_path, session_name)
    name = session_name if session_name is not None else run_path.split("/")[-1]

    return f"Successfully saved session to {org_id}/{repo_id}/{name}"


def predict(
    run_path,
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
    adapter_name,
    lora_name,
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
    noise_schedule,
    use_color_matching,
    preprocessing_type,
):
    try:
        image_generator = run(
            run_path=run_path,
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
            adapter_name=adapter_name,
            lora_name=lora_name,
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
            noise_schedule=noise_schedule,
            use_color_matching=use_color_matching,
            preprocess=preprocessing_type,
        )

        output_frames = []
        for image, image_save_path in image_generator:
            yield image
            output_frames.append(image_save_path)
    except Exception as e:
        raise gr.Error(e)


demo = gr.Blocks()

with demo:
    gr.Markdown("# GIFfusion 💥")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Session Settings", open=False):
                with gr.Tab("Save"):
                    save_org_id = gr.Textbox(label="Org ID", value=ORG_ID)
                    save_repo_id = gr.Textbox(label="Repo ID", value=REPO_ID)
                    save_session_name = gr.Textbox(label="Session Name")
                    save_session_btn = gr.Button(value="Save Session")
                    save_session_status = gr.Markdown()

                with gr.Tab("Load"):
                    load_org_id = gr.Textbox(label="Org ID", value=ORG_ID)
                    load_repo_id = gr.Textbox(label="Repo ID", value=REPO_ID)
                    load_session_name = gr.Textbox(label="Session Name")
                    load_session_settings_filter = gr.Dropdown(
                        [
                            "prompts",
                            "negative_prompts",
                            "diffusion_settings",
                            "preprocessing_settings",
                            "pipeline_settings",
                            "animation_settings",
                        ],
                        label="Filter Settings",
                        multiselect=True,
                    )
                    load_session_settings_btn = gr.Button(value="Load Session Settings")

            with gr.Accordion("Pipeline Settings: Load Models and Pipelines"):
                with gr.Column():
                    model_name = gr.Textbox(
                        label="Model Name", value="runwayml/stable-diffusion-v1-5"
                    )
                    pipeline_name = gr.Textbox(
                        label="Pipeline Name", value="DiffusionPipeline"
                    )
                    lora = gr.Textbox(label="LoRA Checkpoint")

                    with gr.Tab("ControlNet"):
                        controlnet = gr.Textbox(label="ControlNet Checkpoint")
                    with gr.Tab("T2I Adapters"):
                        adapter = gr.Textbox(label="T2I Adapter Checkpoint")

                    custom_pipeline = gr.Textbox(label="Custom Pipeline")

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
                        with gr.Row():
                            save_video_btn = gr.Button(value="Save Video")

            with gr.Accordion("Diffusion Settings", open=False):
                with gr.Tab("Diffusion"):
                    use_fixed_latent = gr.Checkbox(
                        label="Use Fixed Init Latent", elem_id="use_fixed_latent"
                    )
                    use_prompt_embeds = gr.Checkbox(
                        label="Use Prompt Embeds",
                        value=False,
                        interactive=True,
                        elem_id="use_prompt_embed",
                    )
                    seed = gr.Number(value=42, label="Numerical Seed", elem_id="seed")
                    batch_size = gr.Slider(
                        1, 64, step=1, value=1, label="Batch Size", elem_id="batch_size"
                    )
                    num_iteration_steps = gr.Slider(
                        10,
                        1000,
                        step=10,
                        value=20,
                        label="Number of Iteration Steps",
                        elem_id="num_iteration_steps",
                    )
                    guidance_scale = gr.Slider(
                        0.5,
                        20,
                        step=0.5,
                        value=7.5,
                        label="Classifier Free Guidance Scale",
                        elem_id="guidance_scale",
                    )
                    strength = gr.Textbox(
                        label="Image Strength Schedule",
                        value="0:(0.5)",
                        elem_id="strength",
                    )
                    num_latent_channels = gr.Number(
                        value=4,
                        label="Number of Latent Channels",
                        elem_id="num_latent_channels",
                    )
                    image_height = gr.Number(
                        value=512, label="Image Height", elem_id="image_height"
                    )
                    image_width = gr.Number(
                        value=512, label="Image Width", elem_id="image_width"
                    )

                with gr.Tab("Scheduler"):
                    use_default_scheduler = gr.Checkbox(
                        label="Use Default Pipeline Scheduler",
                        elem_id="use_default_scheduler",
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
                        elem_id="scheduler",
                    )
                    scheduler_kwargs = gr.Textbox(
                        label="Scheduler Arguments",
                        value="{}",
                        elem_id="scheduler_kwargs",
                    )

                with gr.Tab("Pipeline"):
                    additional_pipeline_arguments = gr.Textbox(
                        label="Additional Pipeline Arguments",
                        value="{}",
                        interactive=True,
                        lines=4,
                        placeholder="A dictionary of key word arguments to pass to the pipeline",
                        elem_id="additional_pipeline_arguments",
                    )

            with gr.Accordion("Animation Settings", open=False):
                with gr.Tab("Interpolation"):
                    interpolation_type = gr.Dropdown(
                        ["linear", "sine", "curve"],
                        value="linear",
                        label="Interpolation Type",
                        elem_id="interpolation_type",
                    )
                    interpolation_args = gr.Textbox(
                        "",
                        label="Interpolation Parameters",
                        visible=True,
                        elem_id="interpolation_args",
                    )
                with gr.Tab("Motion"):
                    zoom = gr.Textbox("", label="Zoom", elem_id="zoom")
                    translate_x = gr.Textbox(
                        "", label="Translate_X", elem_id="translate_x"
                    )
                    translate_y = gr.Textbox(
                        "", label="Translate_Y", elem_id="translate_y"
                    )
                    angle = gr.Textbox("", label="Angle", elem_id="angle")
                    padding_mode = gr.Dropdown(
                        ["zero", "border", "reflection"],
                        label="Padding Mode",
                        value="border",
                        elem_id="padding_mode",
                    )

                with gr.Tab("Coherence"):
                    coherence_scale = gr.Slider(
                        0,
                        100000,
                        step=50,
                        value=0,
                        label="Coherence Scale",
                        elem_id="coherence",
                    )
                    coherence_alpha = gr.Slider(
                        0,
                        1.0,
                        step=0.1,
                        value=1.0,
                        label="Coherence Alpha",
                        elem_id="coherence_alpha",
                    )
                    coherence_steps = gr.Slider(
                        0,
                        100,
                        step=1,
                        value=1,
                        label="Coherence Steps",
                        elem_id="coherence_steps",
                    )
                    noise_schedule = gr.Textbox(
                        label="Noise Schedule",
                        value="0:(0.01)",
                        interactive=True,
                        elem_id="noise_schedule",
                    )
                    apply_color_matching = gr.Checkbox(
                        label="Apply Color Matching",
                        value=False,
                        interactive=True,
                        elem_id="use_color_matching",
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
            with gr.Row():
                with gr.Tab("Preview"):
                    preview = gr.Image(label="Current Generation")
                with gr.Tab("Video Output"):
                    output = gr.Image(label="Model Output", elem_id="output")

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
                    elem_id="text_prompt_inputs",
                )
            with gr.Row():
                negative_prompt_input = gr.Textbox(
                    value="""low resolution, blurry, worst quality, jpeg artifacts""",
                    label="Negative Prompts",
                    interactive=True,
                    elem_id="negative_prompt_inputs",
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
                    elem_id="audio_component",
                )
                audio_info_btn = gr.Button(value="Get Key Frame Information")
                mel_spectogram_reduce = gr.Dropdown(
                    ["mean", "median", "max"],
                    label="Mel Spectrogram Reduction",
                    value="max",
                    elem_id="mel_spectrogram_reduce",
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
                    label="Preprocessing",
                    multiselect=True,
                    elem_id="preprocess",
                )

    pipe = gr.State()
    run_path = gr.State()

    load_pipeline_btn.click(
        load_pipeline,
        [model_name, pipeline_name, controlnet, adapter, lora, custom_pipeline, pipe],
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

    submit_event = submit.click(create_run_path, outputs=[run_path])
    gen_event = submit_event.success(
        fn=predict,
        inputs=[
            run_path,
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
            adapter,
            lora,
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
            noise_schedule,
            apply_color_matching,
            preprocessing_type,
        ],
        outputs=[preview],
    )

    stop.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])
    save_session_btn.click(
        _save_session,
        inputs=[save_org_id, save_repo_id, run_path, save_session_name],
        outputs=[save_session_status],
    )

    def _load_session(org_id, repo_id, session_name, settings_filter):
        config = load_session(org_id, repo_id, session_name)
        if settings_filter:
            output = {k: config[k] for k in settings_filter}
            config = output

        output = {}
        for module_key, module_value in config.items():
            if module_key == "pipeline_settings":
                output.update(
                    {
                        model_name: config["pipeline_settings"]["model_name"],
                        pipeline_name: config["pipeline_settings"]["pipeline_name"],
                        lora: config["pipeline_settings"]["lora_name"],
                        controlnet: config["pipeline_settings"]["controlnet_name"],
                        adapter: config["pipeline_settings"]["adapter_name"],
                    }
                )

            if module_key == "prompts":
                output.update(
                    {
                        text_prompt_input: config["prompts"]["text_prompt_inputs"],
                        negative_prompt_input: config["prompts"][
                            "negative_prompt_inputs"
                        ],
                    }
                )

            if module_key == "diffusion_settings":
                output.update(
                    {
                        image_height: config["diffusion_settings"]["image_height"],
                        image_width: config["diffusion_settings"]["image_width"],
                        num_iteration_steps: config["diffusion_settings"][
                            "num_inference_steps"
                        ],
                        guidance_scale: config["diffusion_settings"]["guidance_scale"],
                        strength: config["diffusion_settings"]["strength"],
                        seed: config["diffusion_settings"]["seed"],
                        batch_size: config["diffusion_settings"]["batch_size"],
                        scheduler: config["diffusion_settings"]["scheduler"],
                        use_default_scheduler: config["diffusion_settings"][
                            "use_default_scheduler"
                        ],
                        use_prompt_embeds: config["diffusion_settings"][
                            "use_prompt_embeds"
                        ],
                        use_fixed_latent: config["diffusion_settings"][
                            "use_fixed_latent"
                        ],
                        additional_pipeline_arguments: config["diffusion_settings"][
                            "additional_pipeline_arguments"
                        ],
                    }
                )
            if module_key == "animation_settings":
                output.update(
                    {
                        interpolation_type: config["animation_settings"][
                            "interpolation_type"
                        ],
                        interpolation_args: config["animation_settings"][
                            "interpolation_args"
                        ],
                        zoom: config["animation_settings"]["zoom"],
                        translate_x: config["animation_settings"]["translate_x"],
                        translate_y: config["animation_settings"]["translate_y"],
                        angle: config["animation_settings"]["angle"],
                        padding_mode: config["animation_settings"]["padding_mode"],
                        coherence_scale: config["animation_settings"][
                            "coherence_scale"
                        ],
                        coherence_alpha: config["animation_settings"][
                            "coherence_alpha"
                        ],
                        coherence_steps: config["animation_settings"][
                            "coherence_steps"
                        ],
                        noise_schedule: config["animation_settings"]["noise_schedule"],
                        apply_color_matching: config["animation_settings"][
                            "use_color_matching"
                        ],
                    }
                )
            if module_key == "preprocessing_settings":
                output.update(
                    {preprocessing_type: config["preprocessing_settings"]["preprocess"]}
                )

        return output

    load_session_settings_event = load_session_settings_btn.click(
        _load_session,
        [
            load_org_id,
            load_repo_id,
            load_session_name,
            load_session_settings_filter,
        ],
        outputs=[
            pipeline_name,
            model_name,
            lora,
            controlnet,
            adapter,
            custom_pipeline,
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
            noise_schedule,
            apply_color_matching,
            preprocessing_type,
        ],
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=2)
    demo.launch(share=True, debug=DEBUG)
