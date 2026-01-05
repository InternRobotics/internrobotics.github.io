# Training

This tutorial provides a detailed guide for training both Dual System (InternVLA-N1) and System 1 (NavDP) policy models within the InterNav framework.

## Dual-System: InternVLA-N1

### 1. Environment Preparation

Ensure you have installed InterNav and its dependencies, and have access to a multi-GPU environment.

### 2. Start Training System 2
```bash
# training system2 separately
sbatch ./scripts/train/base_train/qwenvl_train/train_system2.sh 
```

#### Pretrained Model Configuration

```bash
# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct
```

#### Dataset Configuration

```bash
# Dataset configuration
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30


#Naming Convention: dataset_height_pitch1_pitch2

#- **125cm / 60cm**: Agent height  
#- **0_30**: Agent pitch starts at 0° elevation shift and shifts to 30° when output ⬇️  
#- **15_15**: Agent pitch starts at 15° elevation shift and keeps 15° when output ⬇️  
```

#### Training Hyperparameters

```bash
# Training hyperparameters
lr=2e-5                    # Global learning rate
vision_tower_lr=5e-6       # Vision encoder learning rate (slower than LLM)
batch_size=2               # Per-GPU batch size
grad_accum_steps=1         # Gradient accumulation steps
                            # Virtual batch size = batch_size × grad_accum_steps × num_gpus
max_pixels=313600          # Maximum image pixels for processing
min_pixels=3136            # Minimum image pixels
```

#### Training Architecture Parameters

```bash
# Model architecture tuning flags
tune_mm_vision=True    # Fine-tune multimodal vision encoder
tune_mm_mlp=True       # Fine-tune multimodal MLP adapter
tune_mm_llm=True       # Fine-tune language model components

# Data augmentation and temporal processing
data_augmentation=True  # Apply data augmentation
num_history=8           # Number of historical observations (frames)
sample_step=4           # Frame sampling rate (every 4th frame)
num_future_steps=4      # Number of future steps to predict
```

### 2. Start Joint Training of System 2 and System 1

```bash
# training system1 based on system2
sbatch ./scripts/train/base_train/qwenvl_train/train_dual_system.sh 
```

#### Pretrained Model Configuration

```bash
# Model configuration
system2_ckpt=checkpoints/InternVLA-N1-System2
```



#### Dataset Configuration

```bash
# Dataset configuration
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30

# %30 means using 30% of the data from each dataset
```



#### Training Architecture Parameters

```bash
# Freeze System 2 weights during joint training
tune_mm_vision=False
tune_mm_mlp=False
tune_mm_llm=False

# Planning and action configuration
predict_step_num=32      # Number of predicted waypoints
pixel_goal_only=True     # Turn and stop actions are not required at this stage

# System 1 backend selection
system1=${system1}       # Supported options: nextdit_async, nextdit, navdp_async
```




## System 1: NavDP

<!-- NavDP content start -->

This tutorial provides a detailed guide for training the NavDP policy model within the InterNav framework. It covers the **training workflow**, **configuration and parameters**, **command-line usage**, and **troubleshooting**.



### Overview of the Training Process

The NavDP training process in InterNav includes the following steps:

1. **Model Initialization**: Load NavDP configuration and initialize model structure and parameters.
2. **Dataset Loading**: Configure dataset paths and preprocessing, build the DataLoader.
3. **Training Parameter Setup**: Set batch size, learning rate, optimizer, and other hyperparameters.
4. **Distributed Training Environment Initialization**: Multi-GPU training is supported out of the box.
5. **Training Execution**: Start the main training loop, with automatic checkpointing and logging.



### Quick Start

#### 1. Environment Preparation

Ensure you have installed InterNav and its dependencies, and have access to a multi-GPU environment.

#### 2. Configuration Check

The NavDP training configuration file is located at:

```bash
InternNav/scripts/train/configs/navdp.py
```

You can modify parameters such as `batch_size`, `epochs`, and dataset path as needed.

#### 3. Start Training

Use the provided shell script for one-click startup:

```bash
cd InternNav/scripts/train
bash start_train.sh --name <experiment_name> --model navdp
```

- `<experiment_name>`: Custom name for this experiment (e.g., 20250723_navdp_train_debug).

This script will automatically allocate 8 GPUs and use torchrun to launch distributed training.

##### Core Command in the Script

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    scripts/train/train.py \
    --name "$NAME" \
    --model-name "$MODEL"
```



### Training Parameters and Configuration

The main training parameters for NavDP are set in `scripts/train/configs/navdp.py`. Common parameters include:

| Parameter         | Description                | Example |
|-------------------|---------------------------|---------|
| `epochs`          | Number of training epochs  | 1000    |
| `batch_size`      | Batch size per GPU         | 16      |
| `lr`              | Learning rate              | 1e-4    |
| `num_workers`     | DataLoader workers         | 8       |
| `dataset_navdp`   | Dataset json path          | data/datasets/navdp_dataset.json |
| `image_size`      | Input image size           | 224     |
| `memory_size`     | Number of history frames   | 8       |
| `predict_size`    | Prediction steps           | 24      |
| `temporal_depth`  | Transformer layers         | 16      |
| `token_dim`       | Feature dimension          | 384     |
| `dropout`         | Dropout probability        | 0.1     |
| `finetune`        | Whether to finetune backbone | False |

For more parameters, see the comments in the configuration file.



### Logging and Model Saving

- Logs, tensorboard files, and checkpoints are saved by default under `data/checkpoints/<experiment_name>/`.
- Tensorboard is supported for visualizing the training process.



### Troubleshooting

- **Multi-GPU training error**: Check that `CUDA_VISIBLE_DEVICES` matches the actual number of GPUs.
- **Dataset path error**: Ensure the json file at `dataset_navdp` exists and is correctly formatted.
- **Out of memory**: Try reducing `batch_size` or `image_size`.



For customizing the model structure or dataset format, see [model.md](./model.md) and [dataset.md](./dataset.md).

<!-- NavDP content end -->


<!-- ## System 2: InternVLA-N1-S2

Currently we don't support the training of InternVLA-N1-S2 in this repository. -->

## Baselines
### Create a Trainer

The Trainer manages the training loop, including data loading, forward pass, loss calculation, and backpropagation.
A custom trainer usually inherits from the [`Base Trainer`](https://github.com/InternRobotics/InternNav/blob/main/internnav/trainer/base.py) and implements:

- `train_epoch()`: Runs one training epoch (batch iteration, forward pass, loss calculation, parameter update).
- `eval_epoch()`: Evaluates the model on the validation set and records metrics.
- `save_checkpoint()`: Saves model weights, optimizer state, and training progress.
- `load_checkpoint()`: Loads pretrained models or resumes training.

Example: [`CMATrainer`](https://github.com/InternRobotics/InternNav/blob/main/internnav/trainer/cma_trainer.py) shows how to handle sequence data, compute action loss, and implement imitation learning.

### Training Data

The training data is under `data/vln_pe/traj_data`. Our dataset provides trajectory data collected from the H1 robot as it navigates through the task environment.
Each observation in the trajectory is paired with its corresponding action.

You may also incorporate external datasets to improve model generalization.

### Set the Corresponding Configuration

Refer to existing **training** configuration files for customization:

- **CMA Model Config**: [`cma_exp_cfg`](https://github.com/InternRobotics/InternNav/blob/main/scripts/train/configs/cma.py)

Configuration files should define:
- `ExpCfg` (experiment config)
- `EvalCfg` (evaluation config)
- `IlCfg` (imitation learning config)

Ensure your configuration is imported and registered in [`__init__.py`](https://github.com/InternRobotics/InternNav/blob/main/scripts/train/configs/__init__.py).

Key parameters include:
- `name`: Experiment name
- `model_name`: Must match the name used during model registration
- `batch_size`: Batch size
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `dataset_*_root_dir`: Dataset paths
- `lmdb_features_dir`: Feature storage path
