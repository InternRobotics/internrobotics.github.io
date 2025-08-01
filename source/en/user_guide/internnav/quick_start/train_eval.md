# Training and Evaluation


This document presents how to train and evaluate models for different systems with InternNav.

## Whole-system

### Evaluation
Before evaluation, we should download the robot assets from [InternUTopiaAssets](https://huggingface.co/datasets/InternRobotics/Embodiments) and move them to the `data/` directory. Model weights of InternVLA-N1 can be downloaded from [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1).

#### Evaluation on isaac sim
The main architecture of the whole-system evaluation adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then submits tasks to the Ray distributed framework based on the corresponding cfg file, enabling the entire evaluation process to run.

First start the ray server:
```bash
ray disable-usage-stats
ray stop
ray start --head
```

Then change the 'model_path' in the cfg file to the path of the InternVLA-N1 weights. Start the evaluation server:
```bash
python -m internnav.agent.utils.server --config scripts/eval/configs/h1_internvla_n1_cfg.py
```

Finally, start the client:
```bash
INTERNUTOPIA_ASSETS_PATH=/path/to/InternUTopiaAssets MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_cfg.py
```

The evaluation results will be saved in the `eval_results.log` file in the output_dir of the config file. The whole evaluation process takes about 10 hours at RTX-4090 graphics platform.


#### Evaluation on habitat
Evaluate on Single-GPU:

```bash
python scripts/eval/eval_habitat.py --model_path checkpoints/InternVLA-N1 --continuous_traj --output_path result/InternVLA-N1/val_unseen_32traj_8steps
```

For multi-gpu inference, currently we only support inference on SLURM.

```bash
./scripts/eval/eval_dual_system.sh
```


## System1

### Training

Download the training data from [Hugging Face](https://huggingface.co/datasets/InternRobotics/InternData-N1/), and organize them in the form mentioned in [installation](./installation.md).

```bash
./scripts/train/start_train.sh --name "$NAME" --model-name navdp
```

### Evaluation

We support the evaluation of diverse System-1 baselines separately in [NavDP](https://github.com/InternRobotics/NavDP/tree/navdp_benchmark) to make it easy to use and deploy.
To install the environment, we provide a quick start below:
#### Step 0: Create the conda environment
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
```
#### Step 1: Install Isaacsim 4.2
```bash
pip install --upgrade pip
pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
# (optional) you can check the installation by running the following
isaacsim omni.isaac.sim.python.kit
```

#### Step 2: Install IsaacLab 1.2.0
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
git checkout tags/v1.2.0
# (optional) you can check the installation by running the following
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

#### Step 3: Install the dependencies for InternVLA-N1(S1)
```bash
git clone https://github.com/OpenRobotLab/NavDP.git
cd NavDP
git checkout navdp_benchmark
pip install -r requirements.txt
```
#### Step 4: Start the InternVLA-N1(S1) server
```bash
cd system1_baselines/navdp
python navdp_server.py --port {PORT} --checkpoint {CHECKPOINT_path}
```

#### Step 5: Running the Evaluation
```bash
python eval_pointgoal_wheeled.py --port {PORT} --scene_dir {SCENE_DIR}
```


## System2

### Data Preparation

Please download the following VLN-CE datasets and insert them into the `data` folder following the same structure.

1. **VLN-CE Episodes**

   Download the VLN-CE episodes:
   - [r2r](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view) (rename R2R_VLNCE_v1/ -> r2r/)
   - [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (rename RxR_VLNCE_v0/ -> rxr/)
   - [envdrop](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) (rename R2R_VLNCE_v1-3_preprocessed/envdrop/ -> envdrop/)

   Extract them into the `data/datasets/` directory.

2. **InternData-N1**

  We provide pre-collected observation-action trajectory data for training. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment. Download the [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) and [SceneData-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1).
The final folder structure should look like this:
```bash
data/
├── scene_data/
│   ├── mp3d_pe/
│   │   ├── 17DRP5sb8fy/
│   │   ├── 1LXtFkjw3qL/
│   │   └── ...
│   ├── mp3d_ce/
│   │   ├── mp3d/
│   │   │   ├── 17DRP5sb8fy/
│   │   │   ├── 1LXtFkjw3qL/
│   │   │   └── ...
│   └── mp3d_n1/
├── vln_pe/
│   ├── raw_data/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
├── └── traj_data/
│       └── mp3d/
│           └── trajectory_0/
│               ├── data/
│               ├── meta/
│               └── videos/
├── vln_ce/
│   ├── raw_data/
│   │   ├── r2r
│   │   │   ├── train
│   │   │   ├── val_seen
│   │   │   │   └── val_seen.json.gz
│   │   │   └── val_unseen
│   │   │       └── val_unseen.json.gz
│   └── traj_data/
└── vln_n1/
    └── traj_data/
```

### Training

Currently, we only support training of small VLN models (CMA, RDP, Seq2Seq) in this repo. For the training of LLM-based VLN (Navid, StreamVLN, etc), please refer to [StreamVLN](https://github.com/OpenRobotLab/StreamVLN) for training details.

```base
# train cma model
./scripts/train/start_train.sh --name cma_train --model cma

# train rdp model
./scripts/train/start_train.sh --name rdp_train --model rdp

# train seq2seq model
./scripts/train/start_train.sh --name seq2seq_train --model seq2seq
```
### Evaluation

Currently we only support evaluate single System2 on Habitat:

Evaluate on Single-GPU:

```bash
python scripts/eval/eval_habitat.py --model_path checkpoints/InternVLA-N1-S2 --mode system2 --output_path results/InternVLA-N1-S2/val_unseen \
```

For multi-gpu inference, currently we only support inference on SLURM.

```bash
./scripts/eval/eval_system2.sh
```
