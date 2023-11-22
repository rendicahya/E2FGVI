import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from assertpy.assertpy import assert_that
from core.utils import to_tensors
from PIL import Image
from python_config import Config
from python_file import count_files
from python_video import video_frames, video_info, video_writer_like
from tqdm import tqdm


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


def read_mask(path):
    masks = []
    mnames = os.listdir(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    mnames.sort()

    for mp in mnames:
        m = Image.open(os.path.join(path, mp))
        m = np.array(m.convert("L"))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=4)

        masks.append(Image.fromarray(m * 255))

    return masks


conf = Config("../intercutmix/config.json")
dataset_dir = Path(conf.ucf101.path)
mask_dir = Path(conf.e2fgvi.mask)
checkpoint = Path(conf.e2fgvi.checkpoint)
output_dir = Path(conf.e2fgvi.output)

assert_that(dataset_dir).is_directory().is_readable()
assert_that(mask_dir).is_directory().is_readable()
assert_that(checkpoint).is_file().is_readable()

assert_that(conf.e2fgvi.max_video_len).is_positive()
assert_that(conf.e2fgvi.model).is_not_empty()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = importlib.import_module(conf.e2fgvi.model)
model = net.InpaintGenerator().to(device)
data = torch.load(checkpoint, map_location=device)
neighbor_stride = 5
n_files = count_files(dataset_dir)
count = 1

model.load_state_dict(data)
model.eval()

for action in mask_dir.iterdir():
    for video in action.iterdir():
        output_path = output_dir / action.name / video.with_suffix(".mp4").name

        if output_path.exists():
            continue

        print(f"{count}/{n_files}: {video.name}")

        video_path = dataset_dir / action.name / video.with_suffix(conf.ucf101.ext).name
        frames_gen = video_frames(video_path, reader=conf.e2fgvi.video_reader)
        info = video_info(video_path)
        w, h = info["width"], info["height"]
        video_length = info["n_frames"]
        video_writer = video_writer_like(video_path, target=output_path)

        if video_length > conf.e2fgvi.max_video_len:
            print(f"Skipping long video: {video_path.name} ({video_length} frames)")
            continue

        frames = [Image.fromarray(f) for f in frames_gen]
        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in frames]

        masks = read_mask(video)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]

        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None] * video_length

        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(video_length, f + neighbor_stride + 1),
                )
            ]

            ref_ids = get_ref_index(f, neighbor_ids, video_length)
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

        for f in range(video_length):
            frame = comp_frames[f].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer.release()

        count += 1
