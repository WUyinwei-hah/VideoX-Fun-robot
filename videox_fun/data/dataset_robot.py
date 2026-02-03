"""
Robot Video Dataset for video editing models.
Loads paired negative/original videos from LIBERO perturbed dataset.
"""

import os
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from decord import VideoReader
from contextlib import contextmanager


@contextmanager
def VideoReader_contextmanager(video_path, num_threads=1):
    """Context manager for VideoReader to ensure proper cleanup."""
    vr = VideoReader(video_path, num_threads=num_threads)
    try:
        yield vr
    finally:
        del vr


def resize_frame(frame, target_size):
    """Resize frame to target size while maintaining aspect ratio."""
    img = Image.fromarray(frame)
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)


class RobotVideoDataset(Dataset):
    """
    Dataset for robot video editing task.
    
    Loads paired negative (perturbed) and original videos, aligns them strictly,
    and returns concatenated frames with instruction prompt.
    
    Alignment strategy:
    - Sample from end of video backwards
    - Try stride=2 first, fall back to stride=1 if not enough frames
    - Ensures strict frame-by-frame alignment between negative and original
    """
    
    def __init__(
        self,
        data_root,  # Path to perturbed dataset root (e.g., /gemini/code/datasets/libero/perturbed)
        source_data_root=None,  # Path to original dataset root for instruction extraction
        video_sample_size=512,
        video_sample_n_frames=16,
        text_drop_ratio=0.1,
        enable_bucket=False,
        return_file_name=False,
        suites=None,  # List of suites to include, e.g., ['libero_10', 'libero_90']
        source_fps=20,  # FPS of saved videos
        target_fps=16,  # FPS expected by model
    ):
        self.data_root = Path(data_root)
        self.source_data_root = Path(source_data_root) if source_data_root else None
        self.video_sample_n_frames = video_sample_n_frames
        self.text_drop_ratio = text_drop_ratio
        self.enable_bucket = enable_bucket
        self.return_file_name = return_file_name
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.fps_ratio = source_fps / target_fps  # 20/16 = 1.25
        
        # Video transforms
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        # Collect all video pairs
        self.data_list = self._collect_video_pairs(suites)
        self.length = len(self.data_list)
        # Alias for compatibility with training scripts that access .dataset
        self.dataset = self.data_list
        print(f"RobotVideoDataset: loaded {self.length} video pairs from {data_root}")
    
    def _collect_video_pairs(self, suites=None):
        """Collect all negative/original video pairs from the dataset."""
        dataset = []
        
        # Iterate through suites
        for suite_dir in sorted(self.data_root.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite_name = suite_dir.name
            
            # Filter by suites if specified
            if suites is not None and suite_name not in suites:
                continue
            
            # Iterate through tasks
            for task_dir in sorted(suite_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name
                
                negative_video_dir = task_dir / "negative_videos"
                original_video_dir = task_dir / "original_videos"
                hdf5_path = task_dir / "paired_negative_dataset.hdf5"
                
                if not negative_video_dir.exists() or not original_video_dir.exists():
                    continue
                if not hdf5_path.exists():
                    continue
                
                # Get instruction from source dataset
                instruction = self._get_instruction(suite_name, task_name, hdf5_path)
                
                # Collect video pairs
                for neg_video in sorted(negative_video_dir.glob("*.mp4")):
                    demo_name = neg_video.stem  # e.g., "demo_0"
                    orig_video = original_video_dir / f"{demo_name}.mp4"
                    
                    if not orig_video.exists():
                        continue

                    width, height = None, None
                    if cv2 is not None:
                        try:
                            cap = cv2.VideoCapture(str(neg_video))
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap.release()
                        except Exception:
                            width, height = None, None
                    
                    dataset.append({
                        'negative_video': str(neg_video),
                        'original_video': str(orig_video),
                        'instruction': instruction,
                        'suite': suite_name,
                        'task': task_name,
                        'demo': demo_name,
                        # For bucket_sampler compatibility
                        'file_path': str(neg_video),
                        'text': instruction,
                        'width': width,
                        'height': height,
                    })
        
        return dataset
    
    def _get_instruction(self, suite_name, task_name, hdf5_path):
        """Extract instruction from source dataset or hdf5 metadata."""
        instruction = ""
        
        try:
            # Try to get from source dataset
            if self.source_data_root:
                source_hdf5 = self.source_data_root / suite_name / f"{task_name}.hdf5"
                if source_hdf5.exists():
                    with h5py.File(source_hdf5, 'r') as f:
                        if 'data' in f and 'problem_info' in f['data'].attrs:
                            problem_info = json.loads(f['data'].attrs['problem_info'])
                            instruction = problem_info.get('language_instruction', '')
            
            # Fallback: try to get from perturbed hdf5's source_dataset attr
            if not instruction and hdf5_path.exists():
                with h5py.File(hdf5_path, 'r') as f:
                    if 'source_dataset' in f.attrs:
                        source_path = f.attrs['source_dataset']
                        if os.path.exists(source_path):
                            with h5py.File(source_path, 'r') as src_f:
                                if 'data' in src_f and 'problem_info' in src_f['data'].attrs:
                                    problem_info = json.loads(src_f['data'].attrs['problem_info'])
                                    instruction = problem_info.get('language_instruction', '')
        except Exception as e:
            print(f"Warning: failed to get instruction for {suite_name}/{task_name}: {e}")
        
        # Fallback: extract from task name
        if not instruction:
            # Convert task name like "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo"
            # to "turn on the stove and put the moka pot on it"
            parts = task_name.replace('_demo', '').split('_')
            # Find where the instruction starts (after SCENE prefix)
            for i, part in enumerate(parts):
                if part.startswith('SCENE'):
                    instruction = ' '.join(parts[i+1:])
                    break
        
        return instruction
    
    def _sample_aligned_frames(self, neg_reader, orig_reader, n_frames):
        """
        Sample frames from both videos with strict alignment and FPS conversion.
        
        Strategy:
        1. Account for FPS difference (source 20fps -> target 16fps)
        2. Start from end of video, sample backwards
        3. Try stride=2 first (in target fps space)
        4. Fall back to stride=1 if not enough frames
        5. Ensures both videos have same frame indices
        
        Returns:
            neg_frames: numpy array of shape (n_frames, H, W, 3)
            orig_frames: numpy array of shape (n_frames, H, W, 3)
        """
        min_len = min(len(neg_reader), len(orig_reader))
        
        # Convert target frame count to source frame space
        # fps_ratio = 1.25 means we need 1.25 source frames per target frame
        base_stride_source = self.fps_ratio  # Base stride in source fps to match target fps
        
        # Calculate required source frames with stride=2 (in target fps space)
        # In source space: stride = 2 * fps_ratio = 2.5
        stride_2_source = 2 * base_stride_source
        required_with_stride2 = int((n_frames - 1) * stride_2_source) + 1
        
        if min_len >= required_with_stride2:
            # Use stride=2 (target space), sample from end
            end_idx = min_len - 1
            # Generate indices with proper FPS conversion
            indices = [int(end_idx - i * stride_2_source) for i in range(n_frames)]
            indices = indices[::-1]  # Reverse to go forward in time
        elif min_len >= int((n_frames - 1) * base_stride_source) + 1:
            # Fall back to stride=1 (target space) with FPS conversion
            end_idx = min_len - 1
            indices = [int(end_idx - i * base_stride_source) for i in range(n_frames)]
            indices = indices[::-1]
        else:
            # Not enough frames, sample uniformly
            if min_len >= n_frames:
                # Sample uniformly across available frames
                indices = np.linspace(0, min_len - 1, n_frames, dtype=int).tolist()
            else:
                indices = list(range(min_len))
        
        # Ensure indices are valid and unique
        indices = [max(0, min(idx, min_len - 1)) for idx in indices]
        
        # Get frames from both videos using same indices
        neg_frames = neg_reader.get_batch(indices).asnumpy()
        orig_frames = orig_reader.get_batch(indices).asnumpy()
        
        # Pad if necessary
        if len(indices) < n_frames:
            pad_count = n_frames - len(indices)
            # Repeat last frame
            neg_pad = np.repeat(neg_frames[-1:], pad_count, axis=0)
            orig_pad = np.repeat(orig_frames[-1:], pad_count, axis=0)
            neg_frames = np.concatenate([neg_frames, neg_pad], axis=0)
            orig_frames = np.concatenate([orig_frames, orig_pad], axis=0)
        
        return neg_frames, orig_frames
    
    def get_batch(self, idx):
        """Get a single sample."""
        data_info = self.data_list[idx % len(self.data_list)]
        
        neg_path = data_info['negative_video']
        orig_path = data_info['original_video']
        instruction = data_info['instruction']
        
        # Use instruction as prompt directly
        text = instruction
        
        # Calculate frame allocation: negative + white + original
        if self.video_sample_n_frames < 3:
            raise ValueError("video_sample_n_frames must be >= 3")
        
        remaining = self.video_sample_n_frames - 1  # -1 for white frame
        n_neg = remaining // 2
        n_orig = remaining - n_neg
        
        with VideoReader_contextmanager(neg_path, num_threads=2) as neg_reader, \
             VideoReader_contextmanager(orig_path, num_threads=2) as orig_reader:
            
            # Sample aligned frames from both videos
            n_sample = max(n_neg, n_orig)
            neg_frames, orig_frames = self._sample_aligned_frames(neg_reader, orig_reader, n_sample)
            
            # Resize frames
            target_size = min(self.video_sample_size)
            neg_resized = np.array([resize_frame(f, target_size) for f in neg_frames])
            orig_resized = np.array([resize_frame(f, target_size) for f in orig_frames])
            
            # Trim to required frame counts
            neg_resized = neg_resized[:n_neg]
            orig_resized = orig_resized[:n_orig]
            
            # Create white separator frame
            white_frame = np.full(
                (1, neg_resized.shape[1], neg_resized.shape[2], 3),
                255, dtype=np.uint8
            )
            
            # Concatenate: negative + white + original
            pixel_values = np.concatenate([neg_resized, white_frame, orig_resized], axis=0)
            
            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.0
                pixel_values = self.video_transforms(pixel_values)
            
            # Random text drop
            if random.random() < self.text_drop_ratio:
                text = ''
        
        return pixel_values, text, 'video', neg_path
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Get item with error handling."""
        data_info = self.data_list[idx % len(self.data_list)]
        
        while True:
            try:
                pixel_values, text, data_type, file_path = self.get_batch(idx)
                
                # Get dimensions for bucket_sampler
                if isinstance(pixel_values, np.ndarray):
                    # Shape: (frames, height, width, channels)
                    height, width = pixel_values.shape[1], pixel_values.shape[2]
                else:
                    # Tensor shape: (frames, channels, height, width)
                    height, width = pixel_values.shape[2], pixel_values.shape[3]
                
                sample = {
                    "pixel_values": pixel_values,
                    "text": text,
                    "data_type": data_type,
                    "idx": idx,
                    "file_path": file_path,  # Required by bucket_sampler
                    "width": width,
                    "height": height,
                }
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                return sample
                
            except Exception as e:
                print(f"Error loading idx {idx}: {e}, {data_info}")
                idx = random.randint(0, self.length - 1)
                data_info = self.data_list[idx % len(self.data_list)]


if __name__ == "__main__":
    # Test the dataset
    dataset = RobotVideoDataset(
        data_root="/gemini/code/datasets/libero/perturbed",
        source_data_root="/gemini/code/datasets/libero/datasets",
        video_sample_size=512,
        video_sample_n_frames=16,
        suites=["all"],
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Pixel values shape: {sample['pixel_values'].shape}")
        print(f"Text: {sample['text'][:100]}...")
