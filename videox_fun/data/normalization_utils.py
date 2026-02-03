"""
Normalization utilities for action and proprio data.

This module provides functions to:
1. Compute dataset statistics (min, max, mean, std) for actions and proprio
2. Save/load statistics to/from JSON
3. Normalize/unnormalize data using these statistics
"""

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def compute_dataset_statistics(episodes_data: list[dict]) -> dict:
    """
    Compute statistics (min, max, mean, std, median) for actions and proprio across all episodes.

    Args:
        episodes_data: List of episode dictionaries, each containing:
            - "actions": np.ndarray of shape (T, action_dim)
            - "proprio": np.ndarray of shape (T, proprio_dim)

    Returns:
        Dictionary containing statistics for actions and proprio:
        {
            "actions_min": np.ndarray of shape (action_dim,),
            "actions_max": np.ndarray of shape (action_dim,),
            "actions_mean": np.ndarray of shape (action_dim,),
            "actions_std": np.ndarray of shape (action_dim,),
            "actions_median": np.ndarray of shape (action_dim,),
            "proprio_min": np.ndarray of shape (proprio_dim,),
            "proprio_max": np.ndarray of shape (proprio_dim,),
            "proprio_mean": np.ndarray of shape (proprio_dim,),
            "proprio_std": np.ndarray of shape (proprio_dim,),
            "proprio_median": np.ndarray of shape (proprio_dim,),
        }
    """
    print("Collecting all actions and proprio from episodes...")
    all_actions = []
    all_proprio = []

    for episode in tqdm(episodes_data, desc="Collecting data"):
        actions = episode["actions"]  # Shape: (T, action_dim)
        proprio = episode["proprio"]  # Shape: (T, proprio_dim)
        all_actions.append(actions)
        all_proprio.append(proprio)

    # Concatenate all timesteps: (total_timesteps, D)
    all_actions_array = np.concatenate(all_actions, axis=0)
    all_proprio_array = np.concatenate(all_proprio, axis=0)

    print(f"Total timesteps for actions: {all_actions_array.shape[0]}")
    print(f"Total timesteps for proprio: {all_proprio_array.shape[0]}")

    # Compute statistics along the timestep dimension (axis=0)
    # Each statistic will have shape (D,)
    print("Computing dataset statistics...")
    stats = {
        "actions_min": np.min(all_actions_array, axis=0),
        "actions_max": np.max(all_actions_array, axis=0),
        "actions_mean": np.mean(all_actions_array, axis=0),
        "actions_std": np.std(all_actions_array, axis=0),
        "actions_median": np.median(all_actions_array, axis=0),
        "proprio_min": np.min(all_proprio_array, axis=0),
        "proprio_max": np.max(all_proprio_array, axis=0),
        "proprio_mean": np.mean(all_proprio_array, axis=0),
        "proprio_std": np.std(all_proprio_array, axis=0),
        "proprio_median": np.median(all_proprio_array, axis=0),
    }

    return stats


def save_dataset_statistics(stats: dict, save_path: str | Path):
    """
    Save dataset statistics to a JSON file.

    Args:
        stats: Statistics dictionary from compute_dataset_statistics()
        save_path: Path to save the JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    stats_json = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            stats_json[key] = value.tolist()
        else:
            stats_json[key] = value

    with open(save_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print(f"Saved dataset statistics to: {save_path}")


def load_dataset_statistics(load_path: str | Path) -> dict:
    """
    Load dataset statistics from a JSON file.

    Args:
        load_path: Path to the JSON file

    Returns:
        Statistics dictionary with numpy arrays
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {load_path}")

    with open(load_path, "r") as f:
        stats_json = json.load(f)

    # Convert lists back to numpy arrays
    stats = {}
    for key, value in stats_json.items():
        if isinstance(value, list):
            stats[key] = np.array(value)
        else:
            stats[key] = value

    print(f"Loaded dataset statistics from: {load_path}")
    return stats


def normalize_data(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    non_negative_only: bool = False,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Normalize data to the range [-1, +1] or [0, +1].

    Formula for [-1, +1]:
        normalized = 2 * ((data - data_min) / (data_max - data_min)) - 1

    Formula for [0, +1]:
        normalized = (data - data_min) / (data_max - data_min)

    Then multiply by scale_multiplier to get final range.

    Args:
        data: Data to normalize, shape (T, D) or (D,)
        data_min: Minimum values per dimension, shape (D,)
        data_max: Maximum values per dimension, shape (D,)
        non_negative_only: If True, scale to [0, +1], else [-1, +1]
        scale_multiplier: Multiplier to adjust final scale

    Returns:
        Normalized data with same shape as input
    """
    # Avoid division by zero
    data_range = data_max - data_min
    data_range = np.where(data_range == 0, 1.0, data_range)

    if not non_negative_only:  # [-1, +1]
        normalized = 2 * ((data - data_min) / data_range) - 1
    else:  # [0, +1]
        normalized = (data - data_min) / data_range

    # Apply scale multiplier
    normalized = scale_multiplier * normalized

    return normalized


def unnormalize_data(
    normalized_data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    non_negative_only: bool = False,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Unnormalize data from the range [-1, +1] or [0, +1] back to original scale.

    Formula for [-1, +1]:
        data = 0.5 * (normalized / scale_multiplier + 1) * (data_max - data_min) + data_min

    Formula for [0, +1]:
        data = (normalized / scale_multiplier) * (data_max - data_min) + data_min

    Args:
        normalized_data: Normalized data, shape (T, D) or (D,)
        data_min: Minimum values per dimension, shape (D,)
        data_max: Maximum values per dimension, shape (D,)
        non_negative_only: If True, data was scaled to [0, +1], else [-1, +1]
        scale_multiplier: Multiplier that was used during normalization

    Returns:
        Unnormalized data with same shape as input
    """
    # Undo scale multiplier
    scaled_data = normalized_data / scale_multiplier

    if not non_negative_only:  # From [-1, +1]
        data = 0.5 * (scaled_data + 1) * (data_max - data_min) + data_min
    else:  # From [0, +1]
        data = scaled_data * (data_max - data_min) + data_min

    return data


def load_or_compute_statistics(
    data_root: str | Path,
    episodes_data: list[dict],
    force_recompute: bool = False,
    stats_filename: str = "dataset_statistics.json",
) -> dict:
    """
    Load dataset statistics from file if it exists, otherwise compute and save.

    Args:
        data_root: Root directory of the dataset
        episodes_data: List of episode dictionaries (only used if computing)
        force_recompute: If True, recompute even if file exists
        stats_filename: Name of the statistics file

    Returns:
        Statistics dictionary
    """
    data_root = Path(data_root)
    stats_path = data_root / stats_filename

    if stats_path.exists() and not force_recompute:
        print(f"Loading existing statistics from: {stats_path}")
        return load_dataset_statistics(stats_path)
    else:
        print(f"Computing statistics from {len(episodes_data)} episodes...")
        stats = compute_dataset_statistics(episodes_data)
        save_dataset_statistics(stats, stats_path)
        return stats
