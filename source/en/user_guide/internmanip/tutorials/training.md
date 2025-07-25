# Training

This document provides a detailed guide for training the policy model for robot manipulation tasks.
It covers the overall **[training workflow](#overview-of-the-training-process)**, **[model and dataset configuration](#trainer-configuration)**, and **[important hyperparameters](#important-hyperparameters)**.
The training pipeline is modular and designed to support various policies, datasets, and fine-tuning options.

## Overview of the Training Process

The entire training process includes the following steps:

1. **Model Initialization**: Load pretrained models and configure trainable parameters
2. **Dataset Loading**: Configure dataset paths, transforms, and data loaders
3. **Training Configuration**: Set up training arguments, optimizers, and schedulers
4. **Training Execution**: Run training with checkpointing and logging
5. **Evaluation**: Optional evaluation during or after training

## Trainer Configuration

### Model Configuration

We provide multiple policy models through a unified interface in `scripts/train/train.py`:

- `pi0`
- `gr00t_n1`
- `gr00t_n1_5`
- `dp_clip`
- `act_clip`

Some models support granular control over which components to fine-tune:

```python
model = model_cls.from_pretrained(
    pretrained_model_name_or_path="path/to/pretrained/model",
    tune_llm=True,  # backbone's LLM
    tune_visual=True,  # backbone's vision tower
    tune_projector=True,  # action head's projector
    tune_diffusion_model=True,  # action head's DiT
)
```

If you want to add your own model, refer to [this document](./model.md).

<!-- ### Dataset Configuration -->

<!-- #### Data Configuration System

The framework uses a modular data configuration system with predefined configs

```python
from grmanipulation.configs.dataset.data_config import DATA_CONFIG_MAP

data_config = "calvin"
data_config_cls = DATA_CONFIG_MAP[data_config]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()
```

You can also create a custom data configuration class that inherits from `BaseDataConfig` in `grmanipulation/configs/dataset/data_config.py`. This class defines how your dataset's modalities are processed and transformed.

Here is an example of a custom data configuration class:

```python
from grmanipulation.configs.dataset.data_config import BaseDataConfig
from grmanipulation.dataset.base import ModalityConfig
from grmanipulation.dataset.transform.concat import ConcatTransform
from grmanipulation.dataset.transform.state_action import (
    StateActionToTensor, StateActionTransform
)

class CustomDataConfig(BaseDataConfig):
    """Custom data configuration for your dataset."""

    # Define data modality keys from your dataset
    state_keys = ["state.joint_positions"]
    action_keys = ["action.joint_velocities"]
    # Define temporal indices
    observation_indices = [0]  # Current timestep for observations
    action_indices = list(range(16))  # Future timesteps for actions (0-15)

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Define modality configurations for dataset loading."""
        # State modality configuration
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        # Action modality configuration
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        return {
            "state": state_modality,
            "action": action_modality,
        }

    def transform(self) -> ComposedModalityTransform:
        """Define data transformations for each modality."""
        transforms = [
            # State transformations
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joint_positions": "mean_std",
                },
            ),
            # Action transformations
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joint_velocities": "mean_std",
                }
            ),
            # Concatenation transform (combines modalities)
            ConcatTransform(
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            )
        ]
        return transforms
```

Then add your configuration to the global registry:

```python
DATA_CONFIG_MAP = {
    ...,
    "custom": CustomDataConfig(),
}
``` -->

### Dataset Loading

The following code snippet demonstrates how to load a dataset using the `LeRobotSingleDataset` class, which is designed to work with the LeRobot dataset format:

```python
from grmanipulation.dataset.base import LeRobotSingleDataset
from grmanipulation.dataset.embodiment_tags import EmbodimentTag

embodiment_tag = EmbodimentTag(config.embodiment_tag)

train_dataset = LeRobotSingleDataset(
    dataset_path=config.dataset_path,
    embodiment_tags=embodiment_tag,
    modality_configs=modality_configs,
    transform=transforms,
)
```

### Important Hyperparameters


In `scripts/train/train.py`, the training scheduler is now configurable through a YAML file using `TrainingArguments` from the 🤗 `transformers` library.

This makes it easier to manage, share, and reproduce training configurations.
Example usage:

```bash
python scripts/train/train.py --config configs/train/pi0_genmanip.yaml


#### Policy and Dataset
```python
policy = ""    # Options: pi0, gr00t_n1, gr00t_n1_5, dp_clip, pi0fast, act_clip
dataset_path = "genmanip-demo"
data_config = "genmanip-v1"     # Data configuration name from DATA_CONFIG_MAP
output_dir = ""                 # Directory to save model checkpoints
```

#### Training parameters

```python
batch_size = 16                  # Batch size per GPU
gradient_accumulation_steps = 1  # Gradient accumulation steps
max_steps = 10000                # Maximum training steps
save_steps = 500                 # Save checkpoints every 500 steps
num_gpus = 1                     # Number of GPUs for training
resume_from_checkpoint = False   # Resume from a checkpoint if available
```

#### Learning Rate & Optimizer

Use cosine annealing with warm-up:

```python
learning_rate = 1e-4     # Learning rate
weight_decay = 1e-5      # Weight decay for AdamW
warmup_ratio = 0.05      # Warm-up ratio for total steps
```


#### Model Fine-tuning
```python
base_model_path = ""          # Path or HuggingFace model ID for base model
tune_llm = False              # Fine-tune language model backbone
tune_visual = True            # Fine-tune vision tower
tune_projector = True         # Fine-tune projector
tune_diffusion_model = True   # Fine-tune diffusion model
use_pretrained_model = False  # Use a pretrained model or not
```


#### LoRA Configuration
```python
lora_rank = 0         # Rank of LORA
lora_alpha = 16       # Alpha value
lora_dropout = 0.1    # Dropout rate
```

#### Data Loading
```python
embodiment_tag = "gr1"      # Embodiment tag (e.g., gr1, new_embodiment)
video_backend = "torchcodec"    # Video backend: decord, torchvision_av, opencv, or torchcodec
dataloader_num_workers = 8  # Number of workers for data loading
```
> ⚠️ Note: The default torchcodec works for most video data. Decord supports H.264 videos but cannot handle AV1 format. When processing AV1 videos, torchvision_av may cause communication deadlocks on multi-node setups. [See more video standrads](https://github.com/huggingface/lerobot/tree/main/benchmarks/video).



#### Miscellaneous
```python
augsteps = 4         # Number of augmentation steps
report_to = "wandb"  # Logging backend: wandb or tensorboard
```
> ⚠️ Note: You need to log in to your own Weights & Biases (wandb) account.




<!--

#### Optimizer Configuration

Use AdamW optimizer with specific beta values:

```python
optim = "adamw_torch"
adam_beta1 = 0.95
adam_beta2 = 0.999
adam_epsilon = 1e-8
weight_decay = 1e-5
```

#### Checkpointing Strategy

```python
save_strategy = "steps"
save_steps = 500                 # Save every 500 steps
save_total_limit = 20            # Keep last 20 checkpoints
resume_from_checkpoint = None    # Auto-resume from latest
```

#### Memory Optimization

```python
gradient_checkpointing = False        # Enable for large models
bf16 = True                           # Mixed precision training
tf32 = True                           # TensorFloat-32 on Ampere GPUs
dataloader_pin_memory = False         # Pin memory for faster transfer
dataloader_persistent_workers = True  # Keep workers alive
ddp_bucket_cap_mb = 100               # DDP bucket size
```

#### System Settings

```python
dataloader_num_workers = 4      # Data loading workers
logging_steps=10.0,             # Log every 10 steps
output_dir = "./checkpoints"    # Output directory
report_to = "wandb"       # Logging backend, tensorboard or wandb
``` -->
