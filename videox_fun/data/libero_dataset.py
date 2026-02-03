"""
Minimal LIBERO HDF5 dataset loader for VideoX-Fun.

Supports two LIBERO formats:
1) Demo format (success_only): HDF5 files with group `data/demo_x/...`.
2) Episode format (all_episodes): each HDF5 is a single episode with
   `primary_images_jpeg`, `wrist_images_jpeg`, `actions`, `proprio`.
"""

import argparse
import importlib
import io
import os
import random
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image

try:
    from .normalization_utils import (
        load_or_compute_statistics,
        normalize_data,
        unnormalize_data,
    )
except ImportError:
    from normalization_utils import (
        load_or_compute_statistics,
        normalize_data,
        unnormalize_data,
    )

torch = importlib.import_module("torch")
torch_utils_data = importlib.import_module("torch.utils.data")
transforms = importlib.import_module("torchvision.transforms")
Dataset = torch_utils_data.Dataset


def get_hdf5_files(data_dir: str) -> list:
    """Recursively collect .h5/.hdf5/.he5 files under data_dir."""
    hdf5_files = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    for root, _, files in os.walk(data_dir, followlinks=True):
        for file in files:
            if file.lower().endswith((".h5", ".hdf5", ".he5")):
                hdf5_files.append(os.path.join(root, file))
    return sorted(hdf5_files)


def decode_jpeg_frame(jpeg_bytes: np.ndarray) -> np.ndarray:
    """Decode JPEG bytes to HxWxC uint8 array."""
    img = Image.open(io.BytesIO(jpeg_bytes.tobytes()))
    return np.array(img).astype(np.uint8)


def parse_instruction_from_filename(file_path: str) -> str:
    """Parse instruction from LIBERO file name."""
    base = os.path.basename(file_path)
    stem = base.replace("_demo.hdf5", "").replace(".hdf5", "")
    words = stem.split("_")
    instruction_words = []
    skip = True
    for w in words:
        if "SCENE" in w:
            skip = False
            continue
        if skip:
            continue
        instruction_words.append(w)
    if not instruction_words:
        # Fallback: try to extract task description from filename tokens
        for token in words:
            if token.startswith("task="):
                return token.replace("task=", "").replace("-", " ")
    return " ".join(instruction_words).strip()


def extract_suite_name(file_path: str) -> str:
    """Extract suite name from path or filename pattern."""
    rel = Path(file_path)
    for part in rel.parts:
        if part.startswith("libero_"):
            if part.endswith("_regen"):
                return part[: -len("_regen")]
            return part
    base = os.path.basename(file_path)
    if "suite=" in base:
        try:
            return base.split("suite=")[1].split("--")[0]
        except Exception:
            return ""
    return ""


def clamp_indices(indices: list, num_steps: int) -> list:
    if num_steps <= 0:
        return []
    return [max(0, min(i, num_steps - 1)) for i in indices]


def sample_three_chunks(num_steps: int, frames_per_chunk: int) -> list:
    """Sample 3*frames_per_chunk indices, starting after 3*frames_per_chunk when possible."""
    total_frames = frames_per_chunk * 3
    if num_steps <= 0:
        return []
    max_start = max(0, num_steps - total_frames)
    min_start = min(frames_per_chunk * 3, max_start)
    start = random.randint(min_start, max_start) if max_start > 0 else 0
    indices = list(range(start, start + total_frames))
    return clamp_indices(indices, num_steps)


class LiberoDataset(Dataset):
    """Minimal LIBERO dataset loader for VideoX-Fun with action/proprio normalization."""

    def __init__(
        self,
        data_root: str,
        video_sample_size: int = 256,
        video_sample_n_frames: int = 48,
        use_wrist_images: bool = False,
        use_third_person_images: bool = True,
        enable_bucket: bool = False,
        return_file_name: bool = False,
        suites: list | None = None,
        first_frame_prob: float = 0.1,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        dataset_stats_path: str | None = None,
        force_recompute_stats: bool = False,
    ):
        """
        Initialize LiberoDataset with normalization support.

        Args:
            data_root: Root directory containing HDF5 files
            video_sample_size: Target size for video frames
            video_sample_n_frames: Number of frames to sample per chunk
            use_wrist_images: Whether to use wrist camera images
            use_third_person_images: Whether to use third-person camera images
            enable_bucket: Whether to enable bucket sampling
            return_file_name: Whether to return file name in sample
            suites: List of suite names to filter (e.g., ["libero_spatial", "libero_10"])
            first_frame_prob: Probability of using first frame special sampling
            normalize_actions: Whether to normalize actions to [-1, +1]
            normalize_proprio: Whether to normalize proprio to [-1, +1]
            dataset_stats_path: Path to dataset statistics JSON file. If None, will look in data_root.
            force_recompute_stats: Force recomputation of statistics even if file exists
        """
        self.data_root = Path(data_root)
        self.video_sample_size = (
            (video_sample_size, video_sample_size)
            if isinstance(video_sample_size, int)
            else tuple(video_sample_size)
        )
        self.video_sample_n_frames = int(video_sample_n_frames)
        if self.video_sample_n_frames <= 0:
            raise ValueError("video_sample_n_frames must be a positive integer")
        if self.video_sample_n_frames % 3 != 0:
            raise ValueError(
                "video_sample_n_frames must be divisible by 3 (total frames across 3 chunks). "
                f"Got video_sample_n_frames={self.video_sample_n_frames}"
            )
        self.frames_per_chunk = self.video_sample_n_frames // 3
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.spatial_token_multiplier = 2
        self.enable_bucket = enable_bucket
        self.return_file_name = return_file_name
        self.suites = set(suites) if suites else None
        self.first_frame_prob = first_frame_prob
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio

        if not (use_wrist_images and use_third_person_images):
            raise ValueError("Must enable both wrist and third-person images for side-by-side views.")

        target_h = min(self.video_sample_size)
        target_w = target_h * 2
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize((target_h, target_w)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Build index of all episodes
        self.data_list = self._build_index()
        self.length = len(self.data_list)
        self.dataset = self.data_list

        # Load or compute normalization statistics if needed
        self.dataset_stats = None
        if self.normalize_actions or self.normalize_proprio:
            # Determine stats path
            if dataset_stats_path is not None:
                stats_path = Path(dataset_stats_path)
            else:
                stats_path = self.data_root / "dataset_statistics.json"

            # Check if we can load existing statistics
            if stats_path.exists() and not force_recompute_stats:
                try:
                    from .normalization_utils import load_dataset_statistics
                except ImportError:
                    from normalization_utils import load_dataset_statistics

                self.dataset_stats = load_dataset_statistics(stats_path)
                print(f"Loaded normalization statistics from: {stats_path}")
            else:
                # Need to compute statistics by loading all episodes
                print("Computing normalization statistics from all episodes...")
                print("This may take a while on first run...")
                episodes_data = self._load_all_episodes_for_stats()
                self.dataset_stats = load_or_compute_statistics(
                    data_root=self.data_root,
                    episodes_data=episodes_data,
                    force_recompute=force_recompute_stats,
                )

        print(f"LiberoDataset: loaded {self.length} demos from {data_root}")
        if self.normalize_actions:
            print("  - Action normalization: ENABLED")
        if self.normalize_proprio:
            print("  - Proprio normalization: ENABLED")

    def _load_all_episodes_for_stats(self) -> list[dict]:
        """Load all episodes to compute normalization statistics."""
        episodes = []
        print(f"Loading {len(self.data_list)} episodes for statistics computation...")

        for data_info in self.data_list:
            file_path = data_info["file_path"]
            demo_key = data_info["demo_key"]

            with h5py.File(file_path, "r") as f:
                if data_info["format"] == "demo":
                    episode = self._load_demo_episode(f, demo_key)
                else:
                    episode = self._load_episode(f)

            episodes.append(
                {
                    "actions": episode["actions"],
                    "proprio": episode["proprio"],
                }
            )

        return episodes

    def _build_index(self) -> list:
        dataset = []
        hdf5_files = get_hdf5_files(str(self.data_root))
        for file_path in hdf5_files:
            suite_name = extract_suite_name(file_path)
            if self.suites and suite_name and suite_name not in self.suites:
                continue

            with h5py.File(file_path, "r") as f:
                f_any: Any = f
                if "data" in f_any:
                    # Demo format with multiple demos per file
                    data_group: Any = f_any["data"]
                    demo_keys = sorted(list(data_group.keys()), key=lambda x: int(x.split("_")[1]))
                    for demo_key in demo_keys:
                        dataset.append(
                            {
                                "file_path": file_path,
                                "demo_key": demo_key,
                                "format": "demo",
                                "suite": suite_name,
                            }
                        )
                else:
                    # Episode format (single episode per file)
                    dataset.append(
                        {
                            "file_path": file_path,
                            "demo_key": None,
                            "format": "episode",
                            "suite": suite_name,
                        }
                    )
        return dataset

    def _load_demo_episode(self, f: h5py.File, demo_key: str) -> dict:
        f_any: Any = f
        obs_group: Any = f_any[f"data/{demo_key}/obs"]

        if "agentview_rgb" in obs_group:
            third_person = obs_group["agentview_rgb"][:].astype(np.uint8)
        elif "agentview_rgb_jpeg" in obs_group:
            third_person = obs_group["agentview_rgb_jpeg"][:]
        else:
            third_person = None

        if "eye_in_hand_rgb" in obs_group:
            wrist = obs_group["eye_in_hand_rgb"][:].astype(np.uint8)
        elif "eye_in_hand_rgb_jpeg" in obs_group:
            wrist = obs_group["eye_in_hand_rgb_jpeg"][:]
        else:
            wrist = None

        actions = f_any[f"data/{demo_key}/actions"][:].astype(np.float32)
        proprio = f_any[f"data/{demo_key}/robot_states"][:].astype(np.float32)

        return {
            "third_person": third_person,
            "wrist": wrist,
            "actions": actions,
            "proprio": proprio,
        }

    def _load_episode(self, f: h5py.File) -> dict:
        f_any: Any = f
        if "primary_images" in f_any:
            third_person = f_any["primary_images"][:].astype(np.uint8)
        elif "primary_images_jpeg" in f_any:
            third_person = f_any["primary_images_jpeg"][:]
        else:
            third_person = None

        if "wrist_images" in f_any:
            wrist = f_any["wrist_images"][:].astype(np.uint8)
        elif "wrist_images_jpeg" in f_any:
            wrist = f_any["wrist_images_jpeg"][:]
        else:
            wrist = None

        actions = f_any["actions"][:].astype(np.float32)
        proprio = f_any["proprio"][:].astype(np.float32)
        instruction = f_any.attrs.get("task_description", "")

        return {
            "third_person": third_person,
            "wrist": wrist,
            "actions": actions,
            "proprio": proprio,
            "instruction": instruction,
        }

    def _decode_frames(self, frames_or_jpeg, indices: list) -> Optional[np.ndarray]:
        if frames_or_jpeg is None:
            return None
        if isinstance(frames_or_jpeg, np.ndarray) and frames_or_jpeg.dtype != object:
            return frames_or_jpeg[indices].astype(np.uint8)

        decoded = [decode_jpeg_frame(frames_or_jpeg[i]) for i in indices]
        return np.stack(decoded, axis=0).astype(np.uint8)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.data_list[idx % len(self.data_list)]
        file_path = data_info["file_path"]
        demo_key = data_info["demo_key"]

        with h5py.File(file_path, "r") as f:
            if data_info["format"] == "demo":
                episode = self._load_demo_episode(f, demo_key)
                instruction = parse_instruction_from_filename(file_path)
            else:
                episode = self._load_episode(f)
                instruction = episode.get("instruction", "")

        third_person = episode["third_person"]
        wrist = episode["wrist"]
        actions = episode["actions"]
        proprio = episode["proprio"]

        # Apply normalization if enabled
        if self.normalize_actions and self.dataset_stats is not None:
            actions = normalize_data(
                actions,
                self.dataset_stats["actions_min"],
                self.dataset_stats["actions_max"],
                non_negative_only=False,
                scale_multiplier=1.0,
            )

        if self.normalize_proprio and self.dataset_stats is not None:
            proprio = normalize_data(
                proprio,
                self.dataset_stats["proprio_min"],
                self.dataset_stats["proprio_max"],
                non_negative_only=False,
                scale_multiplier=1.0,
            )

        num_steps = actions.shape[0]
        frames_per_chunk = self.frames_per_chunk
        total_frames = self.video_sample_n_frames

        if random.random() < self.first_frame_prob:
            base_indices = [0] * frames_per_chunk
            tail_indices = list(range(1, 1 + 2 * frames_per_chunk))
            indices = clamp_indices(base_indices + tail_indices, num_steps)
            action_chunk = np.zeros((total_frames, actions.shape[1]), dtype=actions.dtype)
            proprio_chunk = np.repeat(proprio[0:1], total_frames, axis=0)
        else:
            indices = sample_three_chunks(num_steps, frames_per_chunk)
            action_chunk = actions[indices]
            proprio_chunk = proprio[indices]

        third_person_frames = self._decode_frames(third_person, indices)
        wrist_frames = self._decode_frames(wrist, indices)

        if third_person_frames is None or wrist_frames is None:
            raise ValueError("Both wrist and third-person views are required for concatenation.")

        pixel_values = np.concatenate([wrist_frames, third_person_frames], axis=2)

        if not self.enable_bucket:
            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.0
            pixel_values = self.video_transforms(pixel_values)

        if isinstance(pixel_values, np.ndarray):
            height, width = pixel_values.shape[1], pixel_values.shape[2]
        else:
            height, width = pixel_values.shape[2], pixel_values.shape[3]

        sample = {
            "pixel_values": pixel_values,
            "actions": action_chunk,
            "proprio": proprio_chunk,
            "text": instruction,
            "data_type": "video",
            "idx": idx,
            "file_path": file_path,
            "width": width,
            "height": height,
            "spatial_token_multiplier": self.spatial_token_multiplier,
        }

        if self.return_file_name:
            sample["file_name"] = os.path.basename(file_path)
        if demo_key is not None:
            sample["demo_key"] = demo_key
        if data_info.get("suite"):
            sample["suite"] = data_info.get("suite")

        return sample


def main():
    parser = argparse.ArgumentParser(description="Test LiberoDataset loader with normalization")
    parser.add_argument(
        "--data-root",
        default="/gemini/code/datasets/LIBERO-Cosmos-Policy",
        help="Root path to LIBERO-Cosmos-Policy dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["success", "all"],
        default="success",
        help="Choose to load success-only demos or all episodes",
    )
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--size", type=int, default=256, help="Target frame size")
    parser.add_argument(
        "--normalize-actions",
        action="store_true",
        default=True,
        help="Enable action normalization",
    )
    parser.add_argument(
        "--normalize-proprio",
        action="store_true",
        default=True,
        help="Enable proprio normalization",
    )
    parser.add_argument(
        "--force-recompute-stats",
        action="store_true",
        help="Force recomputation of normalization statistics",
    )
    args = parser.parse_args()

    if args.mode == "success":
        data_root = os.path.join(args.data_root, "success_only")
    else:
        data_root = os.path.join(args.data_root, "all_episodes")

    dataset = LiberoDataset(
        data_root=data_root,
        video_sample_size=args.size,
        video_sample_n_frames=args.frames,
        use_wrist_images=True,
        use_third_person_images=True,
        enable_bucket=False,
        normalize_actions=args.normalize_actions,
        normalize_proprio=args.normalize_proprio,
        force_recompute_stats=args.force_recompute_stats,
    )

    print(f"\nDataset length: {len(dataset)}")
    if len(dataset) == 0:
        return

    sample = dataset[0]
    print(f"\nSample keys: {sorted(sample.keys())}")
    pixel_values = sample["pixel_values"]
    if isinstance(pixel_values, np.ndarray):
        print(f"pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
    else:
        print(f"pixel_values shape: {tuple(pixel_values.shape)}, dtype: {pixel_values.dtype}")

    actions = sample["actions"]
    proprio = sample["proprio"]
    print(f"\nactions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"actions sample (first 3 timesteps):\n{actions[:3]}")

    print(f"\nproprio shape: {proprio.shape}, dtype: {proprio.dtype}")
    print(f"proprio range: [{proprio.min():.3f}, {proprio.max():.3f}]")
    print(f"proprio sample (first 3 timesteps):\n{proprio[:3]}")

    print(f"\ntext: {sample['text']}")
    print(f"file_path: {sample['file_path']}")
    if "demo_key" in sample:
        print(f"demo_key: {sample['demo_key']}")
    if "suite" in sample:
        print(f"suite: {sample['suite']}")

    if dataset.dataset_stats is not None:
        print("\n" + "=" * 60)
        print("Dataset Statistics:")
        print("=" * 60)
        print(f"Actions min: {dataset.dataset_stats['actions_min']}")
        print(f"Actions max: {dataset.dataset_stats['actions_max']}")
        print(f"Proprio min: {dataset.dataset_stats['proprio_min']}")
        print(f"Proprio max: {dataset.dataset_stats['proprio_max']}")


if __name__ == "__main__":
    main()
