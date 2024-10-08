import importlib
import os
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from core.utils import to_tensors
from PIL import Image
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from python_video import frames_to_video, video_frames, video_info


def get_ref_index(f, neighbor_ids, length):
    ref_length = 10
    num_ref = -1
    ref_index = []

    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))

        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break

                ref_index.append(i)

    return ref_index


def read_mask(path, size):
    masks = []
    mnames = os.listdir(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    mnames.sort()

    for mp in mnames:
        m = Image.open(os.path.join(path, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert("L"))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=4)

        masks.append(Image.fromarray(m * 255))

    return masks


root = Path.cwd().parent
dataset = conf.active.dataset
video_in_dir = root / conf[dataset].path
video_in_ext = conf[dataset].ext
video_reader = conf.active.video.reader
mask_dir = root / conf[dataset].annotation.mask.path
annotation_dir = root / conf[dataset].annotation.path
checkpoint = Path(conf.e2fgvi.checkpoint)
video_out_dir = root / conf.e2fgvi.output[dataset]
video_out_ext = conf.e2fgvi.output.ext
model_path = conf.e2fgvi.model
input_type = conf.e2fgvi.input[dataset].type

assert_that(video_in_dir).is_directory().is_readable()
assert_that(mask_dir).is_directory().is_readable()
assert_that(checkpoint).is_file().is_readable()

assert_that(conf.e2fgvi.input[dataset].video.max_len).is_positive()
assert_that(model_path).is_not_empty()

print("Dataset:", dataset)
print("Input:", video_in_dir)
print("Output:", video_out_dir)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = importlib.import_module(model_path)
model = net.InpaintGenerator().to(device)
data = torch.load(checkpoint, map_location=device)
neighbor_stride = 5
bar = tqdm(total=count_files(annotation_dir), dynamic_ncols=True)

model.load_state_dict(data)
model.eval()

for action in mask_dir.iterdir():
    for file in action.iterdir():
        action = file.parent.name
        video_out_path = video_out_dir / action / file.with_suffix(video_out_ext).name

        if video_out_path.exists():
            continue

        bar.set_description(file.name[:50].ljust(50))

        video_in_path = video_in_dir / action / file.with_suffix(video_in_ext).name
        info = video_info(video_in_path)
        w, h, fps = info["width"], info["height"], info["fps"]
        n_masks = count_files(file)

        if input_type == "frames":
            frames_dir = root / conf[dataset].frames
            frames_path = frames_dir / action / file.stem
            n_frames = count_files(frames_path)
            frames = []

            for i, frame_file in enumerate(frames_path.iterdir()):
                frame = cv2.imread(str(frame_file))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                frames.append(frame)
        else:
            n_frames = info["n_frames"]
            frames_gen = video_frames(video_in_path, reader=video_reader)
            frames = [
                Image.fromarray(f) for i, f in enumerate(frames_gen) if i < n_masks
            ]

        if n_frames > conf.e2fgvi.input[dataset].video.max_len:
            print(f"Skipping long video: {video_in_path.name} ({n_frames} frames)")
            count += 1
            continue

        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        # frames = [np.array(f).astype(np.uint8) for f in frames]
        masks = read_mask(file, frames[0].size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None] * n_frames

        for f in range(0, n_frames, neighbor_stride):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(n_frames, f + neighbor_stride + 1),
                )
            ]

            ref_ids = get_ref_index(f, neighbor_ids, n_frames)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]

            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108

                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : h + h_pad, :
                ]

                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : w + w_pad
                ]

                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks[
                        idx
                    ] + frames[idx] * (1 - binary_masks[idx])

                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )

        out_frames = []

        for f in range(n_frames):
            frame = comp_frames[f].astype(np.uint8)

            out_frames.append(frame)

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(
            out_frames,
            target=video_out_path,
            writer=conf.active.video.writer,
            fps=fps,
        )

        bar.update(1)

bar.close()
