# Training

This tutorial provides a detailed guide for training both System 1 (navdp) and System 2 (rdp) policy models within the InterNav framework.

---

## System 1: Navdp

<!-- navdp content start -->

This tutorial provides a detailed guide for training the navdp policy model within the InterNav framework. It covers the **training workflow**, **configuration and parameters**, **command-line usage**, and **troubleshooting**.

---

### Overview of the Training Process

The navdp training process in InterNav includes the following steps:

1. **Model Initialization**: Load navdp configuration and initialize model structure and parameters.
2. **Dataset Loading**: Configure dataset paths and preprocessing, build the DataLoader.
3. **Training Parameter Setup**: Set batch size, learning rate, optimizer, and other hyperparameters.
4. **Distributed Training Environment Initialization**: Multi-GPU training is supported out of the box.
5. **Training Execution**: Start the main training loop, with automatic checkpointing and logging.

---

### Quick Start

#### 1. Environment Preparation

Ensure you have installed InterNav and its dependencies, and have access to a multi-GPU environment.

#### 2. Configuration Check

The navdp training configuration file is located at:

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

---

### Training Parameters and Configuration

The main training parameters for navdp are set in `scripts/train/configs/navdp.py`. Common parameters include:

| Parameter         | Description                | Example |
|-------------------|---------------------------|---------|
| `epochs`          | Number of training epochs  | 1000    |
| `batch_size`      | Batch size per GPU         | 16      |
| `lr`              | Learning rate              | 1e-4    |
| `num_workers`     | DataLoader workers         | 8       |
| `dataset_navdp`   | Dataset json path          | /path/to/multiview_dataset.json |
| `image_size`      | Input image size           | 224     |
| `memory_size`     | Number of history frames   | 8       |
| `predict_size`    | Prediction steps           | 24      |
| `temporal_depth`  | Transformer layers         | 16      |
| `token_dim`       | Feature dimension          | 384     |
| `dropout`         | Dropout probability        | 0.1     |
| `finetune`        | Whether to finetune backbone | True |

For more parameters, see the comments in the configuration file.

---

### Logging and Model Saving

- Logs, tensorboard files, and checkpoints are saved by default under `data/checkpoints/<experiment_name>/`.
- Tensorboard is supported for visualizing the training process.

---

### Troubleshooting

- **Multi-GPU training error**: Check that `CUDA_VISIBLE_DEVICES` matches the actual number of GPUs.
- **Dataset path error**: Ensure the json file at `dataset_navdp` exists and is correctly formatted.
- **Out of memory**: Try reducing `batch_size` or `image_size`.

---

For customizing the model structure or dataset format, see [model.md](./model.md) and [dataset.md](./dataset.md).

<!-- navdp content end -->

---

## System 2: InternVLA-N1-S2

*TODO
