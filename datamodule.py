import PIL
import torch
import torchvision
from torchvision import transforms as T

from utils import parse_key_frames, slerp


@torch.no_grad()
def get_text_embeddings(key_frames, pipe):
    frame_start = key_frames[0][0]

    if frame_start != 0:
        key_frames.append([0, ""])

    output = {}
    for i, prompt in key_frames:
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids.to(pipe.text_encoder.device)
        text_embeddings = pipe.text_encoder(text_inputs)[0]
        output[i] = text_embeddings.squeeze(0).cpu()

    for start, end in zip(key_frames, key_frames[1:]):
        start_frame_idx = start[0]
        end_frame_idx = end[0]
        weights = torch.linspace(0, 1.0, steps=(end_frame_idx - start_frame_idx))

        start_embedding = output[start_frame_idx]
        end_embedding = output[end_frame_idx]

        for i in range(start_frame_idx + 1, end_frame_idx):
            weight = weights[i - start_frame_idx]
            embedding = slerp(weight.item(), start_embedding, end_embedding)
            output[i] = embedding.squeeze(0)

    return output


@torch.no_grad()
def get_noise_embeddings(key_frames, pipe, height, width, generator):
    frame_start = key_frames[0][0]

    if frame_start != 0:
        key_frames.append([0, ""])

    output = {}
    for i, prompt in key_frames:
        output[i] = torch.randn(
            (pipe.unet.in_channels, height // 8, width // 8),
            device=pipe.device,
            generator=generator,
        ).cpu()

    for start, end in zip(key_frames, key_frames[1:]):
        start_frame_idx = start[0]
        end_frame_idx = end[0]
        weights = torch.linspace(0, 1.0, steps=(end_frame_idx - start_frame_idx))

        start_embedding = output[start_frame_idx]
        end_embedding = output[end_frame_idx]

        for i in range(start_frame_idx + 1, end_frame_idx):
            weight = weights[i - start_frame_idx]
            embedding = slerp(weight.item(), start_embedding, end_embedding)
            output[i] = embedding

    return output


class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, pipe, text_prompt_inputs, generator, image_size=(512, 512)):
        super(LatentsDataset).__init__()
        key_frames = parse_key_frames(text_prompt_inputs)

        self.max_frames = max(key_frames, lambda x: x[0])[0]
        self.text_embeddings = get_text_embeddings(key_frames, pipe)

        height, width = image_size
        self.noise_embeddings = get_noise_embeddings(
            key_frames, pipe, height, width, generator
        )
        self.image_size = image_size

    def __len__(self):
        return self.max_frames

    def __getitem__(self, frame_idx):
        noise_embedding = self.noise_embeddings[frame_idx]
        text_embedding = self.text_embeddings[frame_idx]

        return (
            noise_embedding,
            text_embedding,
        )
