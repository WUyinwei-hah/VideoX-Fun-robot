#!/usr/bin/env python3
"""
Script to segment person foreground from videos in wan_i2v_staircase_test_prompts.jsonl using SAM3.
"""

import os
import sys
import json

# Set GPU to use (GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add sam3 to path
sys.path.insert(0, "/gemini/code/sam3")

# Model checkpoint path
SAM3_CHECKPOINT = "/gemini/code/models/sam3/sam3.pt"

from sam3.model_builder import build_sam3_video_predictor


def load_video_frames(video_path):
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def propagate_in_video(predictor, session_id):
    """Propagate masks through all frames in the video."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def get_combined_person_mask(outputs):
    """Combine all person masks into a single mask."""
    if "out_binary_masks" not in outputs or len(outputs["out_binary_masks"]) == 0:
        return None
    
    masks = outputs["out_binary_masks"]
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined_mask = combined_mask | mask
    
    return combined_mask.astype(np.uint8) * 255


def apply_mask_to_frame(frame, mask):
    """Apply mask to frame, keeping only foreground (person) with black background."""
    if mask is None:
        return np.zeros_like(frame)
    
    if len(mask.shape) == 2:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
    else:
        mask_3ch = mask
    
    result = np.where(mask_3ch > 0, frame, 0)
    return result.astype(np.uint8)


def save_video(frames, output_path, fps):
    """Save frames as a video file."""
    if len(frames) == 0:
        print(f"Warning: No frames to save for {output_path}")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def process_video(predictor, video_path, output_path):
    """Process a single video to extract person foreground."""
    print(f"Processing: {video_path}")
    
    frames, fps = load_video_frames(video_path)
    if len(frames) == 0:
        print(f"Error: Could not load frames from {video_path}")
        return False
    
    print(f"  Loaded {len(frames)} frames at {fps} fps")
    
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )
    
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text="a person",
        )
    )
    
    initial_output = response["outputs"]
    if "out_binary_masks" not in initial_output or len(initial_output["out_binary_masks"]) == 0:
        print(f"  Warning: No person detected in {video_path}")
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        return False
    
    print(f"  Detected {len(initial_output['out_obj_ids'])} person(s)")
    
    outputs_per_frame = propagate_in_video(predictor, session_id)
    
    foreground_frames = []
    for frame_idx in range(len(frames)):
        if frame_idx in outputs_per_frame:
            mask = get_combined_person_mask(outputs_per_frame[frame_idx])
        else:
            mask = None
        
        foreground_frame = apply_mask_to_frame(frames[frame_idx], mask)
        foreground_frames.append(foreground_frame)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(foreground_frames, output_path, fps)
    print(f"  Saved foreground video to: {output_path}")
    
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    
    return True


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    jsonl_path = "/gemini/code/VideoX-Fun/examples/wan2.2_fun/wan_i2v_staircase_test_prompts.jsonl"
    foreground_root = "/gemini/code/datasets/Ditto-1M/videos_extracted/foreground_segmented"
    
    items = read_jsonl(jsonl_path)
    
    # Get unique source videos
    unique_sources = set()
    for item in items:
        src = item.get("source_abs")
        if src:
            unique_sources.add(src)
    
    unique_sources = sorted(list(unique_sources))
    print(f"Found {len(unique_sources)} unique source videos to process")
    
    # Initialize SAM3 video predictor
    print("Initializing SAM3 video predictor...")
    print(f"Using checkpoint: {SAM3_CHECKPOINT}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    
    gpus_to_use = [0]
    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CHECKPOINT,
        gpus_to_use=gpus_to_use
    )
    print("SAM3 predictor initialized")
    
    success_count = 0
    fail_count = 0
    
    for video_path in tqdm(unique_sources, desc="Processing videos"):
        # Build output path: replace /source/ with /foreground_segmented/
        if "/source/" in video_path:
            output_path = video_path.replace("/source/", "/foreground_segmented/")
        else:
            # Fallback: put under foreground_root with basename
            output_path = os.path.join(foreground_root, os.path.basename(video_path))
        
        if os.path.exists(output_path):
            print(f"Skipping (already exists): {output_path}")
            success_count += 1
            continue
        
        try:
            if process_video(predictor, video_path, output_path):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            fail_count += 1
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Foreground videos saved under: {foreground_root}")


if __name__ == "__main__":
    main()
