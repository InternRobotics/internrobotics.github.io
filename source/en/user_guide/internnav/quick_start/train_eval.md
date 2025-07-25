# Training and Evaluation


This document presents how to train and evaluate models for different systems with InternNav.


## System1

### Training

```bash
    ./scripts/train/vn_train.sh --name "$NAME" --model-name navdp
```

### Evaluation

Currently, we support the evaluation of diverse System1 baselines separately in [NavDP](https://github.com/OpenRobotLab/NavDP) to make it simplest to use and deploy. Please refer to [NavDP](https://github.com/OpenRobotLab/NavDP) for more details.

For vision language navigation(VLN), please refer to [VLN Evaluation](vln_evaluation.md).

## System2

### Data Preparation

Please download the following VLN-CE datasets and insert them into the `data` folder following the same structure.

1. **VLN-CE Episodes**
   Download the VLN-CE episodes:
   - [r2r](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view) (rename R2R_VLNCE_v1/ -> r2r/)
   - [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (rename RxR_VLNCE_v0/ -> rxr/)
   - [envdrop](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) (rename R2R_VLNCE_v1-3_preprocessed/envdrop/ -> envdrop/)

   Extract them into the `data/datasets/` directory.

2. **Collected Trajectory Data**
  We provide pre-collected observation-action trajectory data for training. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment. For the **EnvDrop** subset, please refer to [DATASET.md](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/README.md) for instructions on how to collect it yourself.
  Download the observation-action trajectory data from [Hugging Face](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data), and extract it to `data/trajectory_data/`.

The final folder structure should look like this:

```bash
data/
├── datasets/
│   ├── r2r/
│   │   ├── train/
│   │   ├── val_seen/
│   │   └── val_unseen/
│   ├── rxr/
│   │   ├── train/
│   │   ├── val_seen/
│   │   └── val_unseen/
│   └── envdrop/
│       ├── envdrop.json.gz
│       └── ...
├── scene_datasets/
│   └── mp3d/
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
└── trajectory_data/
    ├── R2R/
    │   ├── images/
    │   └── annotations.json
    ├── RxR/
    │   ├── images/
    │   └── annotations.json
    └── EnvDrop/
        ├── images/
        └── annotations.json
```

### Training

Currently, we do not support directly training VLAs due to the complexity of entangling different requirements for training LLMs and simulation. Please refer to [StreamVLN](https://github.com/OpenRobotLab/StreamVLN) for training details.

### Evaluation

```bash
    # Evaluate the StreamVLN model
    sh scripts/eval/streamvln_eval_multi_gpu.sh
```

## Whole-system

### Training

Currently, we only support training small models such as Seq2Seq, CMA and RDP on VLN-PE.

```bash
    ./scripts/train/vln_pe_train.sh --name "$NAME" --model-name seq2seq # or "cma", "rdp"
```

### Evaluation

The user can either use script:

```bash
    ./scripts/eval/vln_pe_eval.sh --grutopia_assets_path path/to/grutopia_assets --config path/to/config
```

or use source code for evaluation:

```bash
    # start server
    python InternNav/agent/utils/server.py --config path/to/config
    # run evaluation
    GRUTOPIA_ASSETS_PATH=path/to/grutopia_assets python scripts/eval/eval.py --config path/to/config
```
