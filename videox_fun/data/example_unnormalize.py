"""
Example script demonstrating action/proprio unnormalization during inference.

This shows how to:
1. Load normalization statistics
2. Unnormalize predicted actions/proprio from model output
3. Use unnormalized actions for robot execution
"""

import numpy as np
from pathlib import Path

from normalization_utils import load_dataset_statistics, unnormalize_data


def example_inference_with_unnormalization():
    """Demonstrate the complete inference workflow with unnormalization."""
    
    # Step 1: Load normalization statistics
    stats_path = Path("/gemini/code/datasets/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json")
    
    if not stats_path.exists():
        print(f"Statistics file not found: {stats_path}")
        print("Please run the dataset with normalize_actions=True first to generate statistics.")
        return
    
    dataset_stats = load_dataset_statistics(stats_path)
    
    print("Loaded normalization statistics:")
    print(f"  Actions min: {dataset_stats['actions_min']}")
    print(f"  Actions max: {dataset_stats['actions_max']}")
    print(f"  Proprio min: {dataset_stats['proprio_min']}")
    print(f"  Proprio max: {dataset_stats['proprio_max']}")
    print()
    
    # Step 2: Simulate model predictions (normalized actions in [-1, +1])
    # In real inference, these would come from your trained model
    predicted_actions_normalized = np.array([
        [-0.523, 0.234, -0.891, 0.456, -0.123, 0.789, -0.345],  # timestep 0
        [-0.456, 0.123, -0.789, 0.345, -0.234, 0.678, -0.456],  # timestep 1
        [-0.389, 0.012, -0.687, 0.234, -0.345, 0.567, -0.567],  # timestep 2
    ], dtype=np.float32)
    
    print("Predicted actions (normalized, from model):")
    print(predicted_actions_normalized)
    print(f"Range: [{predicted_actions_normalized.min():.3f}, {predicted_actions_normalized.max():.3f}]")
    print()
    
    # Step 3: Unnormalize actions for robot execution
    predicted_actions_unnormalized = unnormalize_data(
        predicted_actions_normalized,
        dataset_stats["actions_min"],
        dataset_stats["actions_max"],
        non_negative_only=False,
        scale_multiplier=1.0,
    )
    
    print("Predicted actions (unnormalized, for robot):")
    print(predicted_actions_unnormalized)
    print(f"Range: [{predicted_actions_unnormalized.min():.3f}, {predicted_actions_unnormalized.max():.3f}]")
    print()
    
    # Step 4: Same process for proprio (if needed)
    predicted_proprio_normalized = np.array([
        [0.123, -0.456, 0.789, -0.234, 0.567, -0.891, 0.345, -0.678, 0.234],
    ], dtype=np.float32)
    
    predicted_proprio_unnormalized = unnormalize_data(
        predicted_proprio_normalized,
        dataset_stats["proprio_min"],
        dataset_stats["proprio_max"],
        non_negative_only=False,
        scale_multiplier=1.0,
    )
    
    print("Predicted proprio (normalized):")
    print(predicted_proprio_normalized)
    print()
    print("Predicted proprio (unnormalized):")
    print(predicted_proprio_unnormalized)
    print()
    
    # Step 5: Use unnormalized actions for robot control
    print("=" * 60)
    print("ROBOT EXECUTION:")
    print("=" * 60)
    for t, action in enumerate(predicted_actions_unnormalized):
        print(f"Timestep {t}: Execute action {action}")
        # In real code: robot.step(action)


def verify_normalization_inverse():
    """Verify that normalize -> unnormalize is the identity operation."""
    from normalization_utils import normalize_data
    
    stats_path = Path("/gemini/code/datasets/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json")
    
    if not stats_path.exists():
        print("Statistics file not found. Skipping verification.")
        return
    
    dataset_stats = load_dataset_statistics(stats_path)
    
    # Create some test data in the original range
    original_actions = np.array([
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
    ], dtype=np.float32)
    
    # Normalize
    normalized = normalize_data(
        original_actions,
        dataset_stats["actions_min"],
        dataset_stats["actions_max"],
    )
    
    # Unnormalize
    reconstructed = unnormalize_data(
        normalized,
        dataset_stats["actions_min"],
        dataset_stats["actions_max"],
    )
    
    # Check if they match
    max_error = np.abs(original_actions - reconstructed).max()
    
    print("\n" + "=" * 60)
    print("VERIFICATION: normalize -> unnormalize")
    print("=" * 60)
    print(f"Original actions:\n{original_actions}")
    print(f"\nNormalized actions:\n{normalized}")
    print(f"\nReconstructed actions:\n{reconstructed}")
    print(f"\nMax reconstruction error: {max_error:.10f}")
    
    if max_error < 1e-6:
        print("✓ Verification PASSED: normalize/unnormalize are correct inverses")
    else:
        print("✗ Verification FAILED: reconstruction error too large")


if __name__ == "__main__":
    print("=" * 60)
    print("Example: Action/Proprio Unnormalization for Inference")
    print("=" * 60)
    print()
    
    example_inference_with_unnormalization()
    verify_normalization_inverse()
