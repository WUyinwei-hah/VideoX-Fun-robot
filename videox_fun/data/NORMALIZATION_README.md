# Action and Proprio Normalization for LIBERO Dataset

This implementation adds action and proprioceptive state normalization to the VideoX-Fun LIBERO dataset loader, following the same approach as Cosmos-Policy.

## Features

- **Automatic statistics computation**: Calculates min/max/mean/std for actions and proprio across all episodes
- **Persistent statistics**: Saves statistics to JSON for reuse across training runs
- **Normalized training**: Actions and proprio normalized to [-1, +1] range
- **Easy inference**: Utilities for unnormalizing model predictions back to original scale

## Quick Start

### Training with Normalization

```python
from videox_fun.data.libero_dataset import LiberoDataset

# Create dataset with normalization enabled (default)
dataset = LiberoDataset(
    data_root="/path/to/LIBERO-Cosmos-Policy/success_only",
    video_sample_size=256,
    video_sample_n_frames=16,
    use_wrist_images=True,
    use_third_person_images=True,
    normalize_actions=True,     # Enable action normalization
    normalize_proprio=True,      # Enable proprio normalization
)

# First run: computes statistics and saves to dataset_statistics.json
# Subsequent runs: loads statistics from file
```

### Inference with Unnormalization

```python
from videox_fun.data.normalization_utils import (
    load_dataset_statistics,
    unnormalize_data,
)

# Load normalization statistics
stats_path = "/path/to/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json"
dataset_stats = load_dataset_statistics(stats_path)

# Get model predictions (normalized in [-1, +1])
predicted_actions_normalized = model.predict(observation)  # Shape: (T, 7)

# Unnormalize for robot execution
predicted_actions = unnormalize_data(
    predicted_actions_normalized,
    dataset_stats["actions_min"],
    dataset_stats["actions_max"],
    non_negative_only=False,
    scale_multiplier=1.0,
)

# Execute on robot
robot.step(predicted_actions[0])
```

## Normalization Details

### Formula

**Training (normalize to [-1, +1]):**
```python
normalized = 2 * ((x - x_min) / (x_max - x_min)) - 1
```

**Inference (unnormalize to original scale):**
```python
x = 0.5 * (normalized + 1) * (x_max - x_min) + x_min
```

### Statistics Computed

For both actions and proprio:
- `min`: Minimum value per dimension across all episodes
- `max`: Maximum value per dimension across all episodes
- `mean`: Mean value per dimension
- `std`: Standard deviation per dimension
- `median`: Median value per dimension

### File Structure

After first run, the dataset directory will contain:
```
LIBERO-Cosmos-Policy/
└── success_only/
    ├── libero_spatial/
    │   └── *.hdf5
    ├── libero_object/
    │   └── *.hdf5
    └── dataset_statistics.json  # ← Statistics file (auto-generated)
```

## API Reference

### `LiberoDataset.__init__`

New normalization parameters:

- **`normalize_actions`** (bool, default=True): Normalize actions to [-1, +1]
- **`normalize_proprio`** (bool, default=True): Normalize proprio to [-1, +1]
- **`dataset_stats_path`** (str, optional): Path to statistics JSON. If None, uses `{data_root}/dataset_statistics.json`
- **`force_recompute_stats`** (bool, default=False): Force recomputation even if statistics file exists

### Utility Functions

**`compute_dataset_statistics(episodes_data: list[dict]) -> dict`**
- Computes statistics from list of episodes
- Returns dict with keys: `actions_min`, `actions_max`, `proprio_min`, `proprio_max`, etc.

**`normalize_data(data, data_min, data_max, non_negative_only=False, scale_multiplier=1.0) -> ndarray`**
- Normalizes data to [-1, +1] (or [0, +1] if `non_negative_only=True`)
- Supports optional scaling by `scale_multiplier`

**`unnormalize_data(normalized_data, data_min, data_max, non_negative_only=False, scale_multiplier=1.0) -> ndarray`**
- Unnormalizes data from [-1, +1] back to original scale
- Inverse of `normalize_data`

**`load_dataset_statistics(path: str) -> dict`**
- Loads statistics from JSON file
- Returns dict with numpy arrays

**`save_dataset_statistics(stats: dict, path: str)`**
- Saves statistics to JSON file

## Examples

### Test the Dataset

```bash
cd /gemini/code/VideoX-Fun
python -m videox_fun.data.libero_dataset \
    --data-root /gemini/code/datasets/LIBERO-Cosmos-Policy \
    --mode success \
    --frames 16 \
    --normalize-actions \
    --normalize-proprio
```

### Unnormalization Example

```bash
cd /gemini/code/VideoX-Fun
python -m videox_fun.data.example_unnormalize
```

This will:
1. Load normalization statistics
2. Simulate model predictions (normalized actions)
3. Unnormalize actions for robot execution
4. Verify that normalize → unnormalize is the identity operation

## Integration with Training

When training your model:

1. **Dataset returns normalized data**: `actions` and `proprio` in [-1, +1]
2. **Model trains on normalized data**: Learns to predict in [-1, +1] range
3. **During inference**: Unnormalize predictions before sending to robot

```python
# Training loop
for batch in dataloader:
    actions = batch["actions"]      # Already normalized to [-1, +1]
    proprio = batch["proprio"]      # Already normalized to [-1, +1]
    
    # Train model
    loss = model(images, proprio, actions)
    loss.backward()

# Inference
observation = get_robot_observation()
predicted_actions = model.predict(observation)  # Output in [-1, +1]

# Unnormalize before execution
predicted_actions = unnormalize_data(
    predicted_actions,
    stats["actions_min"],
    stats["actions_max"],
)
robot.step(predicted_actions)
```

## Compatibility with Cosmos-Policy

This implementation follows the exact same normalization approach as Cosmos-Policy:
- Same formula: `2 * ((x - min) / (max - min)) - 1`
- Same range: [-1, +1]
- Same statistics: min/max per dimension
- Compatible statistics file format

You can use statistics computed by Cosmos-Policy with this implementation and vice versa.

## Performance Notes

- **First run**: Loads all episodes to compute statistics (~1-5 minutes depending on dataset size)
- **Subsequent runs**: Loads statistics from JSON file (~1 second)
- **Statistics file**: Small (~1-2 KB), stores min/max/mean/std for each dimension

## Troubleshooting

**Q: First run is slow**
- A: Normal. The dataset needs to load all episodes once to compute statistics. Subsequent runs will be fast.

**Q: How do I recompute statistics?**
- A: Set `force_recompute_stats=True` or delete the `dataset_statistics.json` file.

**Q: Can I disable normalization?**
- A: Yes, set `normalize_actions=False` and `normalize_proprio=False`. The dataset will return raw values.

**Q: What if I trained without normalization but want to normalize now?**
- A: You need to retrain. Normalization changes the data distribution significantly.
