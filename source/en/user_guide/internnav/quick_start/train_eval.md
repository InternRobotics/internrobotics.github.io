# Training and Evaluation

This document presents how to train and evaluate models for different systems with InternNav. 

## Whole-system

### Training
The training pipeline is currently under preparation and will be open-sourced soon.

### Evaluation
Before evaluation, we should download the robot assets from [InternUTopiaAssets](https://huggingface.co/datasets/InternRobotics/Embodiments) and move them to the `data/` directory. Model weights of InternVLA-N1 can be downloaded from [InternVLA-N1](https://huggingface.co/InternRobotics/InternVLA-N1).

#### Evaluation on Isaac Sim
[UPDATE] We support using local model and isaac sim in one process now. Evaluate on Single-GPU:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py    
```

For multi-gpu inference, currently we support inference on environments that expose a torchrun-compatible runtime model (e.g., Torchrun or Aliyun DLC).

```bash
# for torchrun
./scripts/eval/bash/torchrun_eval.sh \
    --config scripts/eval/configs/h1_internvla_n1_async_cfg.py

# for alicloud dlc
./scripts/eval/bash/eval_vln_distributed.sh \
    internutopia \
    --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

The main architecture of the whole-system evaluation adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then submits tasks to the Ray distributed framework based on the corresponding cfg file, enabling the entire evaluation process to run.

First, change the 'model_path' in the cfg file to the path of the InternVLA-N1 weights. Start the evaluation server:
```bash
# from one process
conda activate <model_env>
python scripts/eval/start_server.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

Then, start the client to run evaluation:
```bash
# from another process
conda activate <internutopia>
MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

The evaluation results will be saved in the `eval_results.log` file in the output_dir of the config file. The whole evaluation process takes about 10 hours at RTX-4090 graphics platform.
The simulation can be visualized by set `vis_output=True` in eval_cfg.

<img src="../../../_static/video/nav_eval.gif" alt="My GIF">

#### Evaluation on Habitat Sim
Evaluate on Single-GPU:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
```

For multi-gpu inference, currently we support inference on SLURM as well as environments that expose a torchrun-compatible runtime model (e.g., Aliyun DLC).

```bash
# for slurm
./scripts/eval/bash/eval_dual_system.sh

# for torchrun
./scripts/eval/bash/torchrun_eval.sh \
    --config scripts/eval/configs/habitat_dual_system_cfg.py
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
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_cfg.py

# set config with the following fields
eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",  # inference mode: dual_system or system2
            "model_path": "checkpoints/<s2_checkpoint>",  # path to model checkpoint
        }
    )
)
```

For multi-gpu inference, currently we only support inference on SLURM.

```bash
./scripts/eval/bash/eval_system2.sh
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
# Please modify the first line of the bash file to your own conda path
# seq2seq model
./scripts/eval/bash/start_eval.sh --config scripts/eval/configs/h1_seq2seq_cfg.py
# cma model
./scripts/eval/bash/start_eval.sh --config scripts/eval/configs/h1_cma_cfg.py
# rdp model
./scripts/eval/bash/start_eval.sh --config scripts/eval/configs/h1_rdp_cfg.py
```


The evaluation results will be saved in the `eval_results.log` file in the `output_dir` of the config file.
