# Training and Evaluation

This document presents how to train and evaluate models for different systems with InternNav. 

## Data/Checkpoints Preparation
To get started with the training and evaluation, we need to prepare the data and checkpoints properly.
1. **InternVLA-N1 pretrained Checkpoints**
- Download our latest pretrained [checkpoint](https://huggingface.co/InternRobotics/InternVLA-N1) of InternVLA-N1 and run the following script to inference with visualization results. Move the checkpoint to the `checkpoints` directory.
2. **DepthAnything v2 Checkpoints**
- Download the depthanything v2 pretrained [checkpoint](https://huggingface.co/Ashoka74/Placement/resolve/main/depth_anything_v2_vits.pth). Move the checkpoint to the `checkpoints` directory.
3. **InternData-N1 Dataset Episodes**
- Download the [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1). You only need to download the dataset relevant to your chosen task. Download `vln_ce` for VLNCE evaluation in habitat, `vln_pe` for VLNPE evaluation in internutopia.
4. **Scene-N1**
- Download the [SceneData-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1) for `mp3d_ce` or `mp3d_pe`. Extract them into the `data/scene_data/` directory.
5. **Embodiments**
- Download the [Embodiments](https://huggingface.co/datasets/InternRobotics/Embodiments) for the `Embodiments/`

The final folder structure should look like this:

```bash
InternNav/
├── data/
|   ├── Embodiments/
│   ├── scene_data/
│   │   ├── mp3d_ce/
│   │   │   └── mp3d/
│   │   │       ├── 17DRP5sb8fy/
│   │   │       ├── 1LXtFkjw3qL/
│   │   │       └── ...
│   │   └── mp3d_pe/
│   │       ├──17DRP5sb8fy/
│   │       ├── 1LXtFkjw3qL/
│   │       └── ...
│   ├── vln_ce/
│   │   ├── raw_data/
│   │   │   ├── r2r
│   │   │   │   ├── train
│   │   │   │   ├── val_seen
│   │   │   │   │   └── val_seen.json.gz
│   │   │   │   └── val_unseen
│   │   │   │       └── val_unseen.json.gz
│   │   └── traj_data/
│   └── vln_pe/
│       ├── raw_data/    # JSON files defining tasks, navigation goals, and dataset splits
│       │   └── r2r/
│       │       ├── train/
│       │       ├── val_seen/
│       │       │   └── val_seen.json.gz
│       │       └── val_unseen/
│       └── traj_data/   # training sample data for two types of scenes
│           ├── interiornav/
│           │   └── kujiale_xxxx.tar.gz
│           └── r2r/
│               └── trajectory_0/
│                   ├── data/
│                   ├── meta/
│                   └── videos/
├── checkpoints/
│   ├── InternVLA-N1/
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── config.json
│   │   └── ...
│   ├── InternVLA-N1-S2
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── config.json
│   │   └── ...
│   ├── depth_anything_v2_vits.pth
│   ├── r2r
│   │   ├── fine_tuned
│   │   └── zero_shot
├── internnav/
│   └── ...
```

## Whole-system

### Training
The training pipeline is currently under preparation and will be open-sourced soon.

### Evaluation
Before evaluation, we should download the robot assets from [InternUTopiaAssets](https://huggingface.co/datasets/InternRobotics/Embodiments) and move them to the `data/` directory. Model weights of InternVLA-N1 can be downloaded from [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1).

#### Evaluation on isaac sim
The main architecture of the whole-system evaluation adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then submits tasks to the Ray distributed framework based on the corresponding cfg file, enabling the entire evaluation process to run.

First, start change the 'model_path' in the cfg file to the path of the InternVLA-N1 weights. Start the evaluation server:
```bash
# from one process
conda activate <model_env>
python scripts/eval/start_server.py --config scripts/eval/configs/h1_internvla_n1_cfg.py
```

Then, start the client to run evaluation:
```bash
# from another process
conda activate <internutopia>
MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_cfg.py
```

The evaluation results will be saved in the `eval_results.log` file in the output_dir of the config file. The whole evaluation process takes about 10 hours at RTX-4090 graphics platform.
The simulation can be visualized by set `vis_output=True` in eval_cfg.

<img src="../../../_static/video/nav_eval.gif" alt="My GIF">

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

#### InternVLA-N1-S2
Currently we only support evaluate single System2 on Habitat:

Evaluate on Single-GPU:

```bash
python scripts/eval/eval_habitat.py --model_path checkpoints/InternVLA-N1-S2 --mode system2 --output_path results/InternVLA-N1-S2/val_unseen \
```

For multi-gpu inference, currently we only support inference on SLURM.

```bash
./scripts/eval/eval_system2.sh
```

#### Baseline Models
We provide three small VLN baselines (Seq2Seq, CMA, RDP) for evaluation in the InterUtopia (Isaac-Sim) environment.

Download the baseline models:
```bash
# ddppo-models
$ mkdir -p checkpoints/ddppo-models
$ wget -P checkpoints/ddppo-models https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth
# longclip-B
$ huggingface-cli download --include 'longclip-B.pt' --local-dir-use-symlinks False --resume-download Beichenzhang/LongCLIP-B --local-dir checkpoints/clip-long
# download r2r finetuned baseline checkpoints
$ git clone https://huggingface.co/InternRobotics/VLN-PE && mv VLN-PE/r2r checkpoints/
```

Start Evaluation:
```bash
# seq2seq model
./scripts/eval/start_eval.sh --config scripts/eval/configs/h1_seq2seq_cfg.py
# cma model
./scripts/eval/start_eval.sh --config scripts/eval/configs/h1_cma_cfg.py
# rdp model
./scripts/eval/start_eval.sh --config scripts/eval/configs/h1_rdp_cfg.py
```

The evaluation results will be saved in the `eval_results.log` file in the `output_dir` of the config file.
