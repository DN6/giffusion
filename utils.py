import json
import re

import librosa
import numpy as np
import torch
from keyframed.dsl import curve_from_cn_string
from kornia.color import lab_to_rgb, rgb_to_lab
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.io import read_video, write_video
from torchvision.transforms import ToPILImage, ToTensor


def apply_transformation2D(
    image, animations, padding_mode="border", fill_value=torch.zeros(3)
):
    _, c, h, w = image.shape
    center = torch.tensor((h / 2, w / 2)).unsqueeze(0)

    zoom = torch.tensor([animations["zoom"], animations["zoom"]]).unsqueeze(0)

    translate_x = animations["translate_x"]
    translate_y = animations["translate_y"]

    translate = torch.tensor((translate_x, translate_y)).unsqueeze(0)
    angle = torch.tensor([animations["angle"]])

    M = get_affine_matrix2d(
        center=center, translations=translate, angle=angle, scale=zoom
    )
    transformed_img = warp_affine(
        image,
        M=M[:, :2],
        dsize=image.shape[2:],
        padding_mode=padding_mode,
        fill_value=fill_value,
    )

    return transformed_img


def apply_lab_color_matching(image, reference_image):
    to_tensor = ToTensor()
    to_pil_image = ToPILImage()

    image = to_tensor(image).unsqueeze(0)
    reference_image = to_tensor(reference_image).unsqueeze(0)

    image = rgb_to_lab(image)
    reference_image = rgb_to_lab(reference_image)

    output = match_histograms(
        np.array(image[0].permute(1, 2, 0)),
        np.array(reference_image[0].permute(1, 2, 0)),
        channel_axis=-1,
    )

    output = to_tensor(output).unsqueeze(0)
    output = lab_to_rgb(output)
    output = to_pil_image(output[0])

    return output


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
        y=x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
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


def save_video(frames, filename="./output.mp4", fps=24, quality=95, audio_input=None):
    imgs = [Image.open(f) for f in sorted(frames)]
    if quality < 95:
        imgs = list(map(lambda x: x.resize((128, 128), Image.LANCZOS), imgs))

    img_tensors = [ToTensor()(img) for img in imgs]
    img_tensors = list(map(lambda x: x.unsqueeze(0), img_tensors))

    img_tensors = torch.cat(img_tensors)
    img_tensors = img_tensors * 255.0
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    img_tensors = img_tensors.to(torch.uint8)

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
            video_codec="libx264",
        )
    else:
        write_video(
            filename,
            video_array=img_tensors,
            fps=fps,
            video_codec="libx264",
        )


def save_parameters(save_path, parameters):
    with open(f"{save_path}/parameters.json", "w") as f:
        json.dump(parameters, f)


def set_xformers():
    torch_is_version_2 = int(torch.__version__.split(".")[0]) == 2
    try:
        import xformers

        xformers_available = True
    except (ImportError, ModuleNotFoundError):
        xformers_available = False

    if (not torch_is_version_2) and xformers_available:
        return True

    return False
