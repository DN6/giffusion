import re

import librosa
import numpy as np
import torch
from keyframed.dsl import curve_from_cn_string
from kornia.geometry.transform import Affine
from PIL import Image
from torchvision.io import read_video, write_video
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def apply_transformation2D(image, animations, padding_mode="border"):
    zoom = torch.tensor([animations["zoom"], animations["zoom"]]).unsqueeze(0)

    translate_x = animations["translate_x"]
    translate_y = animations["translate_y"]

    translate = torch.tensor((translate_x, translate_y)).unsqueeze(0)
    angle = torch.tensor([animations["angle"]])

    transformed_img = Affine(
        angle=angle, translation=translate, scale_factor=zoom, padding_mode=padding_mode
    )(image)

    return transformed_img


def parse_key_frames(prompts, prompt_parser=None):
    frames = []
    pattern = r"([0-9]+):[\s]*?(.*)[\S\s]"

    key_frame_prompts = re.findall(pattern, prompts)

    for kf_idx, kf_prompt in key_frame_prompts:
        frames.append([int(kf_idx), kf_prompt])

    return frames


def onset_detect(audio, fps, audio_component):
    x, sr = librosa.load(audio)
    harmonic, percussive = librosa.effects.hpss(x, margin=1.0)
    if audio_component == "percussive":
        x = percussive
    if audio_component == "harmonic":
        x = harmonic

    max_audio_frame = int((len(x) / sr) * fps)

    onset_frames = librosa.onset.onset_detect(
        x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
    )
    onset_times = librosa.frames_to_time(onset_frames)

    frames = [int(ot * fps) for ot in onset_times]
    frames = [0] + frames
    frames.append(max_audio_frame)

    return {"frames": frames}


def get_audio_key_frame_information(audio_input, fps, audio_component):
    onsets = onset_detect(audio_input, fps, audio_component)
    audio_key_frames = onsets["frames"]

    return audio_key_frames


def get_mel_reduce_func(reduce_name):
    return {"max": np.amax, "median": np.median, "mean": np.mean}.get(reduce_name)


def get_video_frame_information(video_input):
    video_frames, audio, metadata = load_video_frames(video_input)
    n_frames = len(video_frames)

    return n_frames, metadata["video_fps"]


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def save_gif(frames, filename="./output.gif", fps=24, quality=95, loop=1):
    imgs = [Image.open(f) for f in sorted(frames)]
    if quality < 95:
        imgs = list(map(lambda x: x.resize((128, 128), Image.LANCZOS), imgs))

    imgs += imgs[-1:1:-1]
    duration = len(imgs) // fps
    imgs[0].save(
        fp=filename,
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=duration,
        loop=loop,
        quality=quality,
    )


def load_video_frames(path):
    frames, audio, metadata = read_video(
        filename=path, pts_unit="sec", output_format="TCHW"
    )

    return frames, audio, metadata


def sync_prompts_to_video(text_prompt_inputs, video_frames):
    n_frames = len(video_frames)
    text_key_frames = parse_key_frames(text_prompt_inputs)

    output = {}
    for start, end in zip(text_key_frames, text_key_frames[1:]):
        start_key_frame, start_prompt = start
        end_key_frame, end_prompt = end

        for vf in range(n_frames):
            if output.get(vf) is not None:
                continue

            if vf < end_key_frame:
                output[vf] = start_prompt

    max_text_key_frame_idx, max_text_key_frame_prompt = max(
        text_key_frames, key=lambda x: x[0]
    )

    for vf in range(n_frames):
        if vf >= max_text_key_frame_idx:
            output[vf] = max_text_key_frame_prompt

    min_text_key_frame_idx, min_text_key_frame_prompt = min(
        text_key_frames, key=lambda x: x[0]
    )
    output[min_text_key_frame_idx] = min_text_key_frame_prompt

    output = [[k, v] for k, v in output.items()]
    output = sorted(output, key=lambda x: x[0])

    return output


def save_video(frames, filename="./output.mp4", fps=24, quality=95, audio_input=None):
    imgs = [Image.open(f) for f in sorted(frames)]
    if quality < 95:
        imgs = list(map(lambda x: x.resize((128, 128), Image.LANCZOS), imgs))

    img_tensors = [pil_to_tensor(img) for img in imgs]
    img_tensors = list(map(lambda x: x.unsqueeze(0), img_tensors))

    img_tensors = torch.cat(img_tensors)
    img_tensors = img_tensors.permute(0, 2, 3, 1)

    if audio_input is not None:
        audio_duration = len(img_tensors) / fps
        audio, sr = librosa.load(
            audio_input, sr=None, mono=True, duration=audio_duration
        )
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        write_video(
            filename,
            video_array=img_tensors,
            fps=fps,
            audio_array=audio_tensor,
            audio_fps=sr,
            audio_codec="aac",
        )
    else:
        write_video(
            filename,
            video_array=img_tensors,
            fps=fps,
        )
