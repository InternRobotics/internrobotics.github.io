# 📦 Add a New Dataset

This section explains how to register and add a custom dataset with the InternManip framework.
The process involves two main steps: **[ensuring the dataset format](#dataset-structure)** and **[registering it in code](#implementation-steps)**.



## Dataset Structure

All datasets must follow the [LeRobotDataset Format](https://github.com/huggingface/lerobot) to ensure compatibility with the data loaders and training pipelines.
The expected structure is:


```
<your_dataset_root>  # Root directory of your dataset
│
├── data  # Structured episode data in .parquet format
│   │
│   ├── chunk-000  # Episodes 000000 - 000999
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   │
│   ├── chunk-001  # Episodes 001000 - 001999
│   │   └── ...
│   │
│   ├── ...
│   │
│   └── chunk-00n  # Follows the same convention (1,000 episodes per chunk)
│       └── ...
│
├── meta  # Metadata and statistical information
│   ├── episodes.jsonl         # Per-episode metadata (length, subtask, etc.)
│   ├── info.json              # Dataset-level information
│   ├── tasks.jsonl            # Task definitions
│   ├── modality.json          # Key dimensions and mapping information for each modality
│   └── stats.json             # Global dataset statistics (mean, std, min, max, quantiles)
│
└── videos  # Multi-view videos for each episode
    │
    ├── chunk-000  # Videos for episodes 000000 - 000999
    │   ├── observation.images.head       # Head (main front-view) camera
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   ├── observation.images.hand_left  # Left hand camera
    │   └── observation.images.hand_right # Right hand camera
    │
    ├── chunk-001  # Videos for episodes 001000 - 001999
    │
    ├── ...
    │
    └── chunk-00n  # Follows the same naming and structure

```

> 💡 Note: For more detailed tutorials, please refer to the [Dataset](../tutorials/dataset.md) section.

This separation of raw data, video files, and metadata makes it easier to standardize transformations and modality handling across different datasets.


<!-- > 💡 Note: The `episodes_stats.jsonl` file under `meta/` is optional and can be omitted. -->

## Implementation Steps

### Register a Dataset Class

Create a new dataset class under `internmanip/datasets/`, inheriting from `LeRobotDataset`:

```python
from internmanip.datasets import LeRobotDataset

class CustomDataset(LeRobotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self):
        # Implement custom data loading logic here
        pass
```

This class defines how to read your dataset’s raw files and convert them into a standardized format for training.

### Define a Data Configuration

Each dataset needs a data configuration class that specifies modalities, keys, and transformations.
Create a new configuration file under `internmanip/configs/data_configs/`. Here’s a minimal example:

```python
class CustomDataConfig(BaseDataConfig):
    """Data configuration for the custom dataset."""
    video_keys = ["video.rgb"]
    state_keys = ["state.pos"]
    action_keys = ["action.delta_pos"]
    language_keys = ["annotation.instruction"]

    # Temporal indices
    observation_indices = [0]         # Current timestep for observations
    action_indices = list(range(16))  # Future timesteps for actions (0-15)

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Define modality configurations."""
        return {
            "video": ModalityConfig(self.observation_indices, self.video_keys),
            "state": ModalityConfig(self.observation_indices, self.state_keys),
            "action": ModalityConfig(self.action_indices, self.action_keys),
            "language": ModalityConfig(self.observation_indices, self.language_keys),
        }

    def transform(self):
        """Define preprocessing pipelines."""
        return [
            # Video preprocessing
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224),

            # State preprocessing
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.pos": "mean_std"},
            ),

            # Action preprocessing
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.delta_pos": "mean_std"},
            ),

            # Concatenate modalities
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
        ]
```

### Register Your Config

Finally, register your custom config by adding it to `DATA_CONFIG_MAP`.


```python
DATA_CONFIG_MAP = {
    ...,
    "custom": CustomDataConfig(),
}
```

> 💡 Tips: Adjust the key names (`video_keys`, `state_keys`, etc.) and `normalization_modes` based on your dataset. For multi-view video or multi-joint actions, just add more keys and update the transforms accordingly.

This config sets up how to load and process different modalities, and ensures compatibility with the training framework.

### What's Next?
After registration, you can use your dataset by passing `--dataset_path <path>` and `--data_config custom` to the training YAML file.
