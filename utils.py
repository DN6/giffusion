import re

import imageio
import librosa
import numpy as np
import torch
from PIL import Image


def parse_key_frames(prompts, prompt_parser=None):
    frames = []
    pattern = r"([0-9]+):[\s]*?(.*)[\S\s]"

    key_frame_prompts = re.findall(pattern, prompts)

    for kf_idx, kf_prompt in key_frame_prompts:
        frames.append([int(kf_idx), kf_prompt])

    return frames


def onset_detect(audio, fps):
    x, sr = librosa.load(audio)
    max_audio_frame = int((len(x) / sr) * fps)

    onset_frames = librosa.onset.onset_detect(
        x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
    )
    onset_times = librosa.frames_to_time(onset_frames)
    frames = [int(ot * fps) for ot in onset_times]
    frames.append(max_audio_frame)

    return frames


def sync_prompts_to_audio(text_prompt_inputs, audio_input, fps):
    audio_key_frames = onset_detect(audio_input, fps)
    text_key_frames = parse_key_frames(text_prompt_inputs)

    output = {}
    for start, end in zip(text_key_frames, text_key_frames[1:]):
        start_key_frame, start_prompt = start
        end_key_frame, end_prompt = end

        for akf in audio_key_frames:
            if output.get(akf) is not None:
                continue

            if akf < end_key_frame:
                output[akf] = start_prompt

    max_text_key_frame_idx, max_text_key_frame_prompt = max(
        text_key_frames, key=lambda x: x[0]
    )

    for akf in audio_key_frames:
        if akf >= max_text_key_frame_idx:
            output[akf] = max_text_key_frame_prompt

    min_text_key_frame_idx, min_text_key_frame_prompt = min(
        text_key_frames, key=lambda x: x[0]
    )
    output[min_text_key_frame_idx] = min_text_key_frame_prompt

    output = [[k, v] for k, v in output.items()]
    output = sorted(output, key=lambda x: x[0])

    return output


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


def save_gif(frames, filename="./output.gif", fps=24, quality=95):
    imgs = [Image.open(f) for f in sorted(frames)]
    if quality < 95:
        imgs = [img.resize((128, 128), Image.LANCZOS) for img in imgs]

    imgs += imgs[-1:1:-1]
    duration = len(imgs) // fps
    imgs[0].save(
        fp=filename,
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=duration,
        loop=1,
        quality=99,
    )


def save_video(frames, filename="./output.mp4", fps=24, quality=95):
    imgs = [Image.open(f) for f in sorted(frames)]
    if quality < 95:
        imgs = [img.resize((128, 128), Image.LANCZOS) for img in imgs]

    writer = imageio.get_writer(filename, fps=fps)
    for img in imgs:
        writer.append_data(np.array(img))
    writer.close()
