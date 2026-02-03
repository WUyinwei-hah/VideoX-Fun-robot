import csv
import gc
import io
import json
import math
import os
import random
import re
from contextlib import contextmanager
from random import shuffle
from threading import Thread
from typing import Dict, List, Optional

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, Camera, VideoReader_contextmanager,
                    custom_meshgrid, get_random_mask, get_relative_pose,
                    get_random_mask_for_edit,
                    get_random_mask_for_robot,
                    get_video_reader_batch, padding_image, process_pose_file,
                    process_pose_params, ray_condition, resize_frame,
                    resize_image_with_target_area)


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            dataset_list = getattr(self.dataset, 'dataset', None)
            if isinstance(dataset_list, list):
                content_type = dataset_list[idx].get('type', 'image')
            else:
                # Fallback for datasets that don't expose a meta list (e.g. CharacterVideoEditingDataset).
                # Such datasets are typically video-only.
                content_type = 'video'
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


class ImageVideoEditingDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        elif ann_path.endswith('.jsonl'):
            dataset = []
            with open(ann_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    # Ditto subset jsonl compatibility
                    # Each line contains: source_abs, edited_abs, refined_source_prompt, edited_video_prompt
                    if isinstance(item, dict) and ('source_abs' in item and 'edited_abs' in item):
                        sep = "ad23r2 the screen suddenly turns white, then the camera view suddenly changes."
                        src_text = str(item.get('refined_source_prompt', '')).strip()
                        edt_text = str(item.get('edited_video_prompt', '')).strip()
                        merged_text = (src_text + " " + sep + " " + edt_text).strip()
                        dataset.append({
                            'type': 'video',
                            'source_file_path': item.get('source_abs', ''),
                            'edited_file_path': item.get('edited_abs', ''),
                            'refined_source_prompt': item.get('refined_source_prompt', ''),
                            'edited_video_prompt': item.get('edited_video_prompt', ''),
                            # for compatibility with bucket samplers / legacy code paths
                            'file_path': item.get('edited_abs', ''),
                            'text': merged_text,
                        })
                    else:
                        dataset.append(item)

        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            sep = "ad23r2 the screen suddenly turns white, then the camera view suddenly changes."

            # Ditto-style pair support: {source_file_path, edited_file_path, refined_source_prompt, edited_video_prompt}
            if 'source_file_path' in data_info and 'edited_file_path' in data_info:
                src_path = data_info['source_file_path']
                edt_path = data_info['edited_file_path']
                src_text = data_info.get('refined_source_prompt', '')
                edt_text = data_info.get('edited_video_prompt', '')
                text = (str(src_text).strip() + " " + sep + " " + str(edt_text).strip()).strip()

                if self.data_root is not None:
                    if not os.path.isabs(src_path):
                        src_path = os.path.join(self.data_root, src_path)
                    if not os.path.isabs(edt_path):
                        edt_path = os.path.join(self.data_root, edt_path)

                if self.video_sample_n_frames < 3:
                    raise ValueError("video_sample_n_frames must be >= 3 for source + white + edited concatenation")

                remaining = int(self.video_sample_n_frames - 1)
                n_src = int(remaining // 2)
                n_edt = int(remaining - n_src)
                if n_src == 0 or n_edt == 0:
                    raise ValueError("video_sample_n_frames too small to split into two halves with a middle white frame")

                with VideoReader_contextmanager(src_path, num_threads=2) as src_reader, VideoReader_contextmanager(edt_path, num_threads=2) as edt_reader:
                    min_len = min(len(src_reader), len(edt_reader))

                    min_sample_n_frames = min(
                        min(n_src, n_edt),
                        int(min_len * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                    )
                    if min_sample_n_frames == 0:
                        raise ValueError(f"No Frames in video pair.")

                    video_length = int(self.video_length_drop_end * min_len)
                    clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                    start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                    batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                    try:
                        src_args = (src_reader, batch_index)
                        edt_args = (edt_reader, batch_index)
                        src_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=src_args)
                        edt_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=edt_args)

                        src_resized, edt_resized = [], []
                        for i in range(len(src_pixels)):
                            src_resized.append(resize_frame(src_pixels[i], self.larger_side_of_image_and_video))
                        for i in range(len(edt_pixels)):
                            edt_resized.append(resize_frame(edt_pixels[i], self.larger_side_of_image_and_video))

                        src_pixels = np.array(src_resized)
                        edt_pixels = np.array(edt_resized)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from video pair. Error is {e}.")

                    # Ensure each half occupies half frames; when sampling yields fewer frames, repeat last frame.
                    def _pad_to(arr, n):
                        if arr.shape[0] == n:
                            return arr
                        if arr.shape[0] > n:
                            return arr[:n]
                        pad = np.repeat(arr[-1:, ...], n - arr.shape[0], axis=0)
                        return np.concatenate([arr, pad], axis=0)

                    src_pixels = _pad_to(src_pixels, n_src)
                    edt_pixels = _pad_to(edt_pixels, n_edt)
                    white_frame = np.full((1, src_pixels.shape[1], src_pixels.shape[2], src_pixels.shape[3]), 255, dtype=src_pixels.dtype)
                    pixel_values = np.concatenate([src_pixels, white_frame, edt_pixels], axis=0)

                    if not self.enable_bucket:
                        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                        pixel_values = pixel_values / 255.
                    else:
                        pixel_values = pixel_values

                    if not self.enable_bucket:
                        pixel_values = self.video_transforms(pixel_values)

                    if random.random() < self.text_drop_ratio:
                        text = ''

                return pixel_values, text, 'video', edt_path

            # Legacy single-video entry: {file_path, text}
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class CharacterVideoEditingDataset(Dataset):
    def __init__(
        self,
        recaption_root,
        video_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        return_file_name=False,
        sep=None,
        recaption_key="Recaption Prompt",
        pairing_strategy="random",
        max_pair_candidates=64,
    ):
        self.recaption_root = recaption_root
        self.video_root = video_root if video_root is not None else recaption_root

        self.text_drop_ratio = text_drop_ratio
        self.enable_bucket = enable_bucket
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.larger_side_of_image_and_video = min(self.video_sample_size)

        self.sep = sep or "ad23r2 A full-frame white flash for an instant, then a hard cut to the next shot."
        self.recaption_key = recaption_key
        self.pairing_strategy = pairing_strategy
        self.max_pair_candidates = int(max_pair_candidates)

        self._video_len_cache = {}
        self._character_to_clips = {}
        self._characters = []

        self._build_index()

    def _build_index(self):
        root = os.path.abspath(self.recaption_root)
        video_root = os.path.abspath(self.video_root)

        def _scene_id_from_stem(stem: str) -> str:
            return re.sub(r"__\d+$", "", str(stem))

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith('.json'):
                    continue
                if fn == 'report.json':
                    continue

                json_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(json_path, root)
                rel_dir = os.path.dirname(rel)

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                characters = data.get('Characters', {})
                if isinstance(characters, dict):
                    character_names = [str(k).strip() for k in characters.keys() if str(k).strip()]
                else:
                    character_names = []

                prompt = data.get(self.recaption_key, '')
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                prompt = prompt.strip()
                if not prompt:
                    continue

                stem = os.path.splitext(fn)[0]
                cand_video_dir = os.path.join(video_root, rel_dir)
                video_path = None
                for ext in ('.avi', '.mp4', '.mkv', '.webm', '.mov'):
                    cand = os.path.join(cand_video_dir, stem + ext)
                    if os.path.exists(cand):
                        video_path = cand
                        break
                if video_path is None:
                    continue

                scene_id = _scene_id_from_stem(stem)

                self._character_to_clips.setdefault(rel_dir, []).append(
                    {
                        'video_path': video_path,
                        'prompt': prompt,
                        'character_names': character_names,
                        'scene_id': scene_id,
                    }
                )

        self._characters = []
        for k, v in self._character_to_clips.items():
            if not isinstance(v, list) or len(v) < 2:
                continue
            scene_ids = {c.get('scene_id') for c in v if isinstance(c, dict)}
            scene_ids.discard(None)
            if len(scene_ids) < 2:
                continue
            self._characters.append(k)
        self._characters.sort()
        if len(self._characters) == 0:
            raise ValueError(
                f"No characters with >=2 clips found. recaption_root={self.recaption_root}, video_root={self.video_root}"
            )

    def __len__(self):
        return len(self._characters)

    def _get_video_len(self, video_path: str) -> int:
        if video_path in self._video_len_cache:
            return self._video_len_cache[video_path]
        with VideoReader_contextmanager(video_path, num_threads=2) as vr:
            n = len(vr)
        self._video_len_cache[video_path] = n
        return n

    def _sanitize_character_component(self, name: str) -> str:
        n = str(name).strip()
        n = re.sub(r"\s+", "_", n)
        n = re.sub(r"[^A-Za-z0-9._-]", "_", n)
        n = re.sub(r"_+", "_", n)
        n = n.strip("._-")
        return n

    def _guess_primary_name(self, character_key: str, names: List[str]) -> Optional[str]:
        folder = os.path.basename(character_key)
        folder_norm = self._sanitize_character_component(folder)
        for nm in names:
            if self._sanitize_character_component(nm) == folder_norm:
                return nm
        return None

    def _build_extra_id_map(self, character_key: str, src_names: List[str], edt_names: List[str]) -> Dict[str, str]:
        all_names: List[str] = []
        seen = set()
        for nm in (src_names or []) + (edt_names or []):
            if not isinstance(nm, str):
                nm = str(nm)
            nm = nm.strip()
            if not nm or nm in seen:
                continue
            seen.add(nm)
            all_names.append(nm)

        primary = self._guess_primary_name(character_key, all_names)
        ordered: List[str] = []
        if primary is not None:
            ordered.append(primary)
        for nm in sorted(all_names):
            if nm == primary:
                continue
            ordered.append(nm)

        mapping: Dict[str, str] = {}
        for i, nm in enumerate(ordered):
            mapping[nm] = f"<extra_id_{i}>"
        return mapping

    def _anonymize_prompt(self, text: str, mapping: Dict[str, str]) -> str:
        if not text or not mapping:
            return text
        out = str(text)
        for nm in sorted(mapping.keys(), key=len, reverse=True):
            token = mapping[nm]
            out = re.sub(re.escape(nm), token, out)
        return out

    def _sample_pair_for_character(self, character_key: str):
        clips = self._character_to_clips.get(character_key, [])
        if len(clips) < 2:
            raise ValueError(f"Character has <2 clips: {character_key}")

        for _ in range(16):
            a_idx = random.randrange(len(clips))
            a = clips[a_idx]
            a_scene = a.get('scene_id', None) if isinstance(a, dict) else None

            others = [c for i, c in enumerate(clips) if i != a_idx]
            if a_scene is not None:
                others = [c for c in others if c.get('scene_id', None) != a_scene]
            if not others:
                continue

            if str(self.pairing_strategy).lower() == "closest_length":
                a_len = self._get_video_len(a['video_path'])
                candidates = others
                if self.max_pair_candidates > 0 and len(candidates) > self.max_pair_candidates:
                    candidates = random.sample(candidates, self.max_pair_candidates)

                best = None
                best_diff = None
                for c in candidates:
                    try:
                        c_len = self._get_video_len(c['video_path'])
                    except Exception:
                        continue
                    diff = abs(int(c_len) - int(a_len))
                    if best is None or diff < best_diff:
                        best = c
                        best_diff = diff
                if best is None:
                    best = random.choice(others)
                return a, best

            return a, random.choice(others)

        a_idx = random.randrange(len(clips))
        a = clips[a_idx]
        others = [c for i, c in enumerate(clips) if i != a_idx]
        if not others:
            raise ValueError(f"Character has no other clip: {character_key}")
        return a, random.choice(others)

    def get_batch(self, idx):
        character_key = self._characters[idx % len(self._characters)]
        src_item, edt_item = self._sample_pair_for_character(character_key)

        src_path = src_item['video_path']
        edt_path = edt_item['video_path']
        src_text = src_item.get('prompt', '')
        edt_text = edt_item.get('prompt', '')

        name_map = self._build_extra_id_map(
            character_key=character_key,
            src_names=src_item.get('character_names', []) or [],
            edt_names=edt_item.get('character_names', []) or [],
        )
        src_text = self._anonymize_prompt(src_text, name_map)
        edt_text = self._anonymize_prompt(edt_text, name_map)
        text = (str(src_text).strip() + " " + self.sep + " " + str(edt_text).strip()).strip()

        if self.video_sample_n_frames < 3:
            raise ValueError("video_sample_n_frames must be >= 3 for source + white + edited concatenation")

        remaining = int(self.video_sample_n_frames - 1)
        n_src = int(remaining // 2)
        n_edt = int(remaining - n_src)
        if n_src == 0 or n_edt == 0:
            raise ValueError("video_sample_n_frames too small to split into two halves with a middle white frame")

        with VideoReader_contextmanager(src_path, num_threads=2) as src_reader, VideoReader_contextmanager(edt_path, num_threads=2) as edt_reader:
            src_len_total = len(src_reader)
            edt_len_total = len(edt_reader)
            min_len_total = min(src_len_total, edt_len_total)

            # We cut two clips to (approximately) the same temporal length.
            # If videos are too short, we still ensure both appear by sampling >=1 frame and padding.
            available = int(min_len_total * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            if available <= 0:
                available = 1

            min_sample_n_frames = min(min(n_src, n_edt), available)
            if min_sample_n_frames <= 0:
                min_sample_n_frames = 1

            video_length = int(self.video_length_drop_end * min_len_total)
            if video_length <= 0:
                video_length = min_len_total if min_len_total > 0 else 1

            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
            if clip_length <= 0:
                clip_length = 1

            start_low = int(self.video_length_drop_start * video_length)
            start_high = max(start_low, video_length - clip_length)
            start_idx = random.randint(start_low, start_high) if start_high > start_low else start_low
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                src_args = (src_reader, batch_index)
                edt_args = (edt_reader, batch_index)
                src_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=src_args)
                edt_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=edt_args)

                src_resized, edt_resized = [], []
                for i in range(len(src_pixels)):
                    src_resized.append(resize_frame(src_pixels[i], self.larger_side_of_image_and_video))
                for i in range(len(edt_pixels)):
                    edt_resized.append(resize_frame(edt_pixels[i], self.larger_side_of_image_and_video))

                src_pixels = np.array(src_resized)
                edt_pixels = np.array(edt_resized)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video pair. Error is {e}.")

            def _pad_to(arr, n):
                if arr.shape[0] == n:
                    return arr
                if arr.shape[0] > n:
                    return arr[:n]
                pad = np.repeat(arr[-1:, ...], n - arr.shape[0], axis=0)
                return np.concatenate([arr, pad], axis=0)

            src_pixels = _pad_to(src_pixels, n_src)
            edt_pixels = _pad_to(edt_pixels, n_edt)
            white_frame = np.full((1, src_pixels.shape[1], src_pixels.shape[2], src_pixels.shape[3]), 255, dtype=src_pixels.dtype)
            pixel_values = np.concatenate([src_pixels, white_frame, edt_pixels], axis=0)

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
            else:
                pixel_values = pixel_values

            if not self.enable_bucket:
                pixel_values = self.video_transforms(pixel_values)

            if random.random() < self.text_drop_ratio:
                text = ''

        return pixel_values, text, 'video', edt_path

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self._characters) - 1)
        return sample


class MovieBenchFirstFramePairDatasetSeg(Dataset):
    def __init__(
        self,
        ann_path,
        video_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        return_file_name=False,
        sep=None,
        video_path_key="video_path",
        foreground_video_path_key="foreground_video_path",
        prompt_key="prompt",
        first_frame_key="first_frame",
        force_start_frame0=True,
        use_white_frame=True,
    ):
        self.ann_path = ann_path
        self.video_root = video_root

        self.text_drop_ratio = text_drop_ratio
        self.enable_bucket = enable_bucket
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.larger_side_of_image_and_video = min(self.video_sample_size)

        self.use_white_frame = bool(use_white_frame)
        if sep is not None:
            self.sep = sep
        elif self.use_white_frame:
            self.sep = "ad23r2 A full-frame white flash for an instant, then a hard cut to the next shot."
        else:
            self.sep = "ad23r2 A hard cut to the next shot, featuring the same character, maintaining identity continuity across shots."
        self.video_path_key = str(video_path_key)
        self.foreground_video_path_key = str(foreground_video_path_key)
        self.prompt_key = str(prompt_key)
        self.first_frame_key = str(first_frame_key)
        self.force_start_frame0 = bool(force_start_frame0)

        self._first_frame_to_items = {}
        self._first_frames = []
        self._index_keys = []

        self._build_index()

    def _build_index(self):
        ann_path = os.path.abspath(self.ann_path)
        if not os.path.exists(ann_path):
            raise ValueError(f"ann_path not found: {ann_path}")

        if not ann_path.endswith('.csv'):
            raise ValueError(f"MovieBenchFirstFramePairDataset expects a .csv ann_path, got: {ann_path}")

        with open(ann_path, 'r', encoding='utf-8') as csvfile:
            rows = list(csv.DictReader(csvfile))

        for r in rows:
            if not isinstance(r, dict):
                continue
            first_frame = r.get(self.first_frame_key, None)
            video_rel = r.get(self.video_path_key, None)
            foreground_video_rel = r.get(self.foreground_video_path_key, None)
            prompt = r.get(self.prompt_key, None)

            if first_frame is None or str(first_frame).strip() == "":
                continue
            if video_rel is None or str(video_rel).strip() == "":
                continue
            if prompt is None:
                prompt = ""
            prompt = str(prompt).strip()
            if prompt == "":
                continue

            item = {
                'first_frame': str(first_frame).strip(),
                'video_path': str(video_rel).strip(),
                'foreground_video_path': str(foreground_video_rel).strip() if foreground_video_rel is not None else "",
                'prompt': prompt,
            }
            self._first_frame_to_items.setdefault(item['first_frame'], []).append(item)

        self._first_frames = [k for k, v in self._first_frame_to_items.items() if isinstance(v, list) and len(v) >= 2]
        self._first_frames.sort()
        if len(self._first_frames) == 0:
            raise ValueError(f"No first_frame groups with >=2 items found in {ann_path}")

        self._index_keys = []
        self.dataset = []
        for k in self._first_frames:
            items = self._first_frame_to_items.get(k, [])
            if not items:
                continue
            for it in items:
                rep = it.get('video_path', None) if isinstance(it, dict) else None
                if rep is None:
                    continue
                self._index_keys.append(k)
                self.dataset.append({'type': 'video', 'file_path': rep, 'text': ''})

    def __len__(self):
        return len(self._index_keys)

    def _resolve_video_path(self, video_path: str) -> str:
        p = str(video_path)
        if self.video_root is None:
            return p
        return os.path.join(self.video_root, p)

    def _sample_pair_in_group(self, first_frame_key: str):
        items = self._first_frame_to_items.get(first_frame_key, [])
        if not isinstance(items, list) or len(items) < 2:
            raise ValueError(f"first_frame group has <2 items: {first_frame_key}")
        a, b = random.sample(items, 2)
        return a, b

    def get_batch(self, idx):
        first_frame_key = self._index_keys[idx % len(self._index_keys)]
        src_item, edt_item = self._sample_pair_in_group(first_frame_key)

        fg_src_video_rel = src_item.get('foreground_video_path', None)
        if fg_src_video_rel is not None and str(fg_src_video_rel).strip() != "":
            _fg_src_path = self._resolve_video_path(fg_src_video_rel)
            if os.path.exists(_fg_src_path):
                src_path = _fg_src_path
            else:
                src_path = self._resolve_video_path(src_item.get('video_path', None))
        else:
            src_path = self._resolve_video_path(src_item.get('video_path', None))
        edt_path = self._resolve_video_path(edt_item['video_path'])
        edt_text = edt_item.get('prompt', '')
        text = (self.sep + " " + str(edt_text).strip()).strip()
        # print("text")
        # print(text)

        if self.use_white_frame:
            if self.video_sample_n_frames < 3:
                raise ValueError("video_sample_n_frames must be >= 3 for source + white + edited concatenation")
            remaining = int(self.video_sample_n_frames - 1)
            n_src = int(remaining // 2)
            n_edt = int(remaining - n_src)
            if n_src == 0 or n_edt == 0:
                raise ValueError("video_sample_n_frames too small to split into two halves with a middle white frame")
        else:
            if self.video_sample_n_frames < 2:
                raise ValueError("video_sample_n_frames must be >= 2 for source + edited concatenation")
            n_src = int(self.video_sample_n_frames // 2) + 1
            n_edt = int(self.video_sample_n_frames - n_src)
            if n_src == 0 or n_edt == 0:
                raise ValueError("video_sample_n_frames too small to split into source and edited parts")

        with VideoReader_contextmanager(src_path, num_threads=2) as src_reader, VideoReader_contextmanager(edt_path, num_threads=2) as edt_reader:
            src_len_total = len(src_reader)
            edt_len_total = len(edt_reader)
            min_len_total = min(src_len_total, edt_len_total)
            if min_len_total <= 0:
                raise ValueError("No Frames in paired videos")

            available = int(min_len_total * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            if available <= 0:
                available = 1

            min_sample_n_frames = min(min(n_src, n_edt), available)
            if min_sample_n_frames <= 0:
                min_sample_n_frames = 1

            video_length = int(self.video_length_drop_end * min_len_total)
            if video_length <= 0:
                video_length = min_len_total

            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
            if clip_length <= 0:
                clip_length = 1

            if self.force_start_frame0:
                start_idx = 0
            else:
                start_low = int(self.video_length_drop_start * video_length)
                start_high = max(start_low, video_length - clip_length)
                start_idx = random.randint(start_low, start_high) if start_high > start_low else start_low
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                src_args = (src_reader, batch_index)
                edt_args = (edt_reader, batch_index)
                src_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=src_args)
                edt_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=edt_args)

                src_resized, edt_resized = [], []
                for i in range(len(src_pixels)):
                    src_resized.append(resize_frame(src_pixels[i], self.larger_side_of_image_and_video))
                for i in range(len(edt_pixels)):
                    edt_resized.append(resize_frame(edt_pixels[i], self.larger_side_of_image_and_video))

                src_pixels = np.array(src_resized)
                edt_pixels = np.array(edt_resized)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video pair. Error is {e}.")

            def _pad_to(arr, n):
                if arr.shape[0] == n:
                    return arr
                if arr.shape[0] > n:
                    return arr[:n]
                pad = np.repeat(arr[-1:, ...], n - arr.shape[0], axis=0)
                return np.concatenate([arr, pad], axis=0)

            src_pixels = _pad_to(src_pixels, n_src)
            edt_pixels = _pad_to(edt_pixels, n_edt)
            if self.use_white_frame:
                white_frame = np.full((1, src_pixels.shape[1], src_pixels.shape[2], src_pixels.shape[3]), 255, dtype=src_pixels.dtype)
                pixel_values = np.concatenate([src_pixels, white_frame, edt_pixels], axis=0)
            else:
                pixel_values = np.concatenate([src_pixels, edt_pixels], axis=0)

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
            else:
                pixel_values = pixel_values

            if not self.enable_bucket:
                pixel_values = self.video_transforms(pixel_values)

            if random.random() < self.text_drop_ratio:
                text = ''

        return pixel_values, text, 'video', edt_path

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self) - 1)
        return sample

class MovieBenchFirstFramePairDataset(Dataset):
    def __init__(
        self,
        ann_path,
        video_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        return_file_name=False,
        sep=None,
        video_path_key="video_path",
        prompt_key="prompt",
        first_frame_key="first_frame",
        force_start_frame0=True,
        use_white_frame=True,
    ):
        self.ann_path = ann_path
        self.video_root = video_root

        self.text_drop_ratio = text_drop_ratio
        self.enable_bucket = enable_bucket
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.larger_side_of_image_and_video = min(self.video_sample_size)

        self.use_white_frame = bool(use_white_frame)
        if sep is not None:
            self.sep = sep
        elif self.use_white_frame:
            self.sep = "ad23r2 A full-frame white flash for an instant, then a hard cut to the next shot."
        else:
            self.sep = "ad23r2 A hard cut to the next shot, featuring the same character, maintaining identity continuity across shots."
        self.video_path_key = str(video_path_key)
        self.prompt_key = str(prompt_key)
        self.first_frame_key = str(first_frame_key)
        self.force_start_frame0 = bool(force_start_frame0)

        self._first_frame_to_items = {}
        self._first_frames = []
        self._index_keys = []

        self._build_index()

    def _build_index(self):
        ann_path = os.path.abspath(self.ann_path)
        if not os.path.exists(ann_path):
            raise ValueError(f"ann_path not found: {ann_path}")

        if not ann_path.endswith('.csv'):
            raise ValueError(f"MovieBenchFirstFramePairDataset expects a .csv ann_path, got: {ann_path}")

        with open(ann_path, 'r', encoding='utf-8') as csvfile:
            rows = list(csv.DictReader(csvfile))

        for r in rows:
            if not isinstance(r, dict):
                continue
            first_frame = r.get(self.first_frame_key, None)
            video_rel = r.get(self.video_path_key, None)
            prompt = r.get(self.prompt_key, None)

            if first_frame is None or str(first_frame).strip() == "":
                continue
            if video_rel is None or str(video_rel).strip() == "":
                continue
            if prompt is None:
                prompt = ""
            prompt = str(prompt).strip()
            if prompt == "":
                continue

            item = {
                'first_frame': str(first_frame).strip(),
                'video_path': str(video_rel).strip(),
                'prompt': prompt,
            }
            self._first_frame_to_items.setdefault(item['first_frame'], []).append(item)

        self._first_frames = [k for k, v in self._first_frame_to_items.items() if isinstance(v, list) and len(v) >= 2]
        self._first_frames.sort()
        if len(self._first_frames) == 0:
            raise ValueError(f"No first_frame groups with >=2 items found in {ann_path}")

        self._index_keys = []
        self.dataset = []
        for k in self._first_frames:
            items = self._first_frame_to_items.get(k, [])
            if not items:
                continue
            for it in items:
                rep = it.get('video_path', None) if isinstance(it, dict) else None
                if rep is None:
                    continue
                self._index_keys.append(k)
                self.dataset.append({'type': 'video', 'file_path': rep, 'text': ''})

    def __len__(self):
        return len(self._index_keys)

    def _resolve_video_path(self, video_path: str) -> str:
        p = str(video_path)
        if self.video_root is None:
            return p
        return os.path.join(self.video_root, p)

    def _sample_pair_in_group(self, first_frame_key: str):
        items = self._first_frame_to_items.get(first_frame_key, [])
        if not isinstance(items, list) or len(items) < 2:
            raise ValueError(f"first_frame group has <2 items: {first_frame_key}")
        a, b = random.sample(items, 2)
        return a, b

    def get_batch(self, idx):
        first_frame_key = self._index_keys[idx % len(self._index_keys)]
        src_item, edt_item = self._sample_pair_in_group(first_frame_key)

        src_path = self._resolve_video_path(src_item['video_path'])
        edt_path = self._resolve_video_path(edt_item['video_path'])
        src_text = src_item.get('prompt', '')
        edt_text = edt_item.get('prompt', '')
        text = (str(src_text).strip() + " " + self.sep + " " + str(edt_text).strip()).strip()
        # print("text")
        # print(text)

        if self.use_white_frame:
            if self.video_sample_n_frames < 3:
                raise ValueError("video_sample_n_frames must be >= 3 for source + white + edited concatenation")
            remaining = int(self.video_sample_n_frames - 1)
            n_src = int(remaining // 2)
            n_edt = int(remaining - n_src)
            if n_src == 0 or n_edt == 0:
                raise ValueError("video_sample_n_frames too small to split into two halves with a middle white frame")
        else:
            if self.video_sample_n_frames < 2:
                raise ValueError("video_sample_n_frames must be >= 2 for source + edited concatenation")
            n_src = int(self.video_sample_n_frames // 2) + 1
            n_edt = int(self.video_sample_n_frames - n_src)
            if n_src == 0 or n_edt == 0:
                raise ValueError("video_sample_n_frames too small to split into source and edited parts")

        with VideoReader_contextmanager(src_path, num_threads=2) as src_reader, VideoReader_contextmanager(edt_path, num_threads=2) as edt_reader:
            src_len_total = len(src_reader)
            edt_len_total = len(edt_reader)
            min_len_total = min(src_len_total, edt_len_total)
            if min_len_total <= 0:
                raise ValueError("No Frames in paired videos")

            available = int(min_len_total * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            if available <= 0:
                available = 1

            min_sample_n_frames = min(min(n_src, n_edt), available)
            if min_sample_n_frames <= 0:
                min_sample_n_frames = 1

            video_length = int(self.video_length_drop_end * min_len_total)
            if video_length <= 0:
                video_length = min_len_total

            clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
            if clip_length <= 0:
                clip_length = 1

            if self.force_start_frame0:
                start_idx = 0
            else:
                start_low = int(self.video_length_drop_start * video_length)
                start_high = max(start_low, video_length - clip_length)
                start_idx = random.randint(start_low, start_high) if start_high > start_low else start_low
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                src_args = (src_reader, batch_index)
                edt_args = (edt_reader, batch_index)
                src_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=src_args)
                edt_pixels = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=edt_args)

                src_resized, edt_resized = [], []
                for i in range(len(src_pixels)):
                    src_resized.append(resize_frame(src_pixels[i], self.larger_side_of_image_and_video))
                for i in range(len(edt_pixels)):
                    edt_resized.append(resize_frame(edt_pixels[i], self.larger_side_of_image_and_video))

                src_pixels = np.array(src_resized)
                edt_pixels = np.array(edt_resized)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video pair. Error is {e}.")

            def _pad_to(arr, n):
                if arr.shape[0] == n:
                    return arr
                if arr.shape[0] > n:
                    return arr[:n]
                pad = np.repeat(arr[-1:, ...], n - arr.shape[0], axis=0)
                return np.concatenate([arr, pad], axis=0)

            src_pixels = _pad_to(src_pixels, n_src)
            edt_pixels = _pad_to(edt_pixels, n_edt)
            if self.use_white_frame:
                white_frame = np.full((1, src_pixels.shape[1], src_pixels.shape[2], src_pixels.shape[3]), 255, dtype=src_pixels.dtype)
                pixel_values = np.concatenate([src_pixels, white_frame, edt_pixels], axis=0)
            else:
                pixel_values = np.concatenate([src_pixels, edt_pixels], axis=0)

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
            else:
                pixel_values = pixel_values

            if not self.enable_bucket:
                pixel_values = self.video_transforms(pixel_values)

            if random.random() < self.text_drop_ratio:
                text = ''

        return pixel_values, text, 'video', edt_path

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self) - 1)
        return sample



class ImageVideoDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video', video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1, 
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        padding_subject_info=True,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.padding_subject_info = padding_subject_info

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        if data_info.get('type', 'image')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''

            control_video_id = data_info['control_file_path']
            
            if control_video_id is not None:
                if self.data_root is None:
                    control_video_id = control_video_id
                else:
                    control_video_id = os.path.join(self.data_root, control_video_id)
                
            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                if control_video_id is not None:
                    with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            resized_frames = []
                            for i in range(len(control_pixel_values)):
                                frame = control_pixel_values[i]
                                resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                resized_frames.append(resized_frame)
                            control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values = control_pixel_values / 255.
                            del control_video_reader
                        else:
                            control_pixel_values = control_pixel_values

                        if not self.enable_bucket:
                            control_pixel_values = self.video_transforms(control_pixel_values)
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                control_camera_values = None
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = pixel_values.shape[-2:]
                else:
                    visual_height, visual_width = pixel_values.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            return pixel_values, control_pixel_values, subject_image, control_camera_values, text, "video"
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i]).convert('RGB')
                    width, height = subject_image.size
                    total_pixels = width * height

                    if self.padding_subject_info:
                        img = padding_image(subject_image, visual_width, visual_height)
                    else:
                        img = resize_image_with_target_area(subject_image, 1024 * 1024)

                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(np.array(img))
                if self.padding_subject_info:
                    subject_image = np.array(subject_images)
                else:
                    subject_image = subject_images
            else:
                subject_image = None

            return image, control_image, subject_image, None, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values, subject_image, control_camera_values, name, data_type = self.get_batch(idx)

                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["subject_image"] = subject_image
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample


class ImageVideoSafetensorsDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root
        self.dataset = dataset
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_root is None:
            path = self.dataset[idx]["file_path"]
        else:
            path = os.path.join(self.data_root, self.dataset[idx]["file_path"])
        state_dict = load_file(path)
        return state_dict


class TextDataset(Dataset):
    def __init__(self, ann_path, text_drop_ratio=0.0):
        print(f"loading annotations from {ann_path} ...")
        with open(ann_path, 'r') as f:
            self.dataset = json.load(f)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        self.text_drop_ratio = text_drop_ratio

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                item = self.dataset[idx]
                text = item['text']

                # Randomly drop text (for classifier-free guidance)
                if random.random() < self.text_drop_ratio:
                    text = ''

                sample = {
                    "text": text,
                    "idx": idx
                }
                return sample

            except Exception as e:
                print(f"Error at index {idx}: {e}, retrying with random index...")
                idx = np.random.randint(0, self.length - 1)