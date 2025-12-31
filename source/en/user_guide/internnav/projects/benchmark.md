# Extended Benchmarks in InternNav

This guide details how to use specific dataset for training a VLA model for different navigation benchmark.

## VL-LN Bench

VL-LN Bench is a large-scale benchmark for Interactive Instance Goal Navigation. VL-LN Bench provides: (1) an automatically dialog-augmented trajectory generation pipeline, (2) a comprehensive evaluation protocol for training and assessing dialog-capable navigation models, and (3) the dataset and base model used in our experiments. For full details, see our [paper](https://arxiv.org/abs/2512.22342) and the [project website](https://0309hws.github.io/VL-LN.github.io/).

- [Data Collection Pipeline](https://github.com/InternRobotics/VL-LN)
- [Training and Evaluation Code](https://github.com/InternRobotics/InternNav)
- [Dataset](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench) and [Base Model](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel)


### 1. Download Data & Assets
VL-LN Bench is built on Matterport3D (MP3D) Scene Dataset, so you need to download both the MP3D scene dataset and the VL-LN Bench dataset.
- Scene Datasets

    Download the [MP3D Scene Dataset](https://niessner.github.io/Matterport/)
- [VL-LN Data](https://huggingface.co/datasets/InternRobotics/VL-LN-Bench)
- [VL-LN Base Model](https://huggingface.co/InternRobotics/VL-LN-Bench-basemodel)
  
After unzipping the base model, scene datasets, and trajectory data, put everything under VL-LN-Bench/ in the layout below.
  ```bash
  VL-LN-Bench/
  ├── base_model/ 
  │   └── iion/
  ├── raw_data/ 
  │   └── mp3d/
  │       ├── scene_summary/
  │       ├── train/ 
  │       │   ├── train_ion.json.gz
  │       │   └── train_iion.json.gz
  │       └── val_unseen/ 
  │           ├── val_unseen_ion.json.gz
  │           └── val_unseen_iion.json.gz
  ├── scene_datasets/
  │   └── mp3d/
  │       ├── 17DRP5sb8fy/
  │       ├── 1LXtFkjw3qL/
  │       ...
  └── traj_data/
      ├── mp3d_split1/
      ├── mp3d_split2/
      └── mp3d_split3/
  ```

### 2. Environment Setup
Here we set up the Python environment for VL-LN Bench and InternVLA-N1. If you've already installed the InternNav Habitat environment, you can skip those steps and only run the commands related to VL-LN Bench.

- Get Code
  ```bash
  git clone git@github.com:InternRobotics/VL-LN.git # code for data collection
  git clone git@github.com:InternRobotics/InternNav.git # code for training and evaluation
  ```

- Create Conda Environment
  ```bash
  conda create -n vlln python=3.9 -y
  conda activate vlln
  ```
  
- Install Dependencies
  ```bash
  conda install habitat-sim=0.2.4 withbullet headless -c conda-forge -c aihabitat
  cd VL-LN
  pip install -r requirements.txt
  cd ../InternNav
  pip install -e .
  ```

### 3. Guidance for Data Collection
This step is optional. You can either use our collected data for policy training, or follow this step to collect your own training data.


- Prerequisites:
  - Get pointnav_weights.pth from [VLFM](https://github.com/bdaiinstitute/vlfm/tree/main/data)
  - Arrange the Directory Structure Like This
    ```bash
    VL-LN
    ├── dialog_generation/
    ├── images/
    ├── VL-LN-Bench/
    │   ├── base_model/ 
    │   ├── raw_data/ 
    │   ├── scene_datasets/
    │   ├── traj_data/
    │   └── pointnav_weights.pth
    ...
    ```

- Collect Trajectories
  ```bash
  # If having slurm
  sbatch generate_frontiers_dialog.sh

  # Or directly run
  python generate_frontiers_dialog.py \
      --task instance \
      --vocabulary hm3d \
      --scene_ids all \
      --shortest_path_threshold 0.1 \
      --target_detected_threshold 5 \
      --episodes_file_path VL-LN-Bench/raw_data/mp3d/train/train_iion.json.gz \
      --habitat_config_path dialog_generation/config/tasks/dialog_mp3d.yaml \
      --baseline_config_path dialog_generation/config/expertiments/gen_videos.yaml \
      --normal_category_path dialog_generation/normal_category.json \
      --pointnav_policy_path VL-LN-Bench/pointnav_weights.pth\
      --scene_summary_path VL-LN-Bench/raw_data/mp3d/scene_summary\
      --output_dir <PATH_TO_YOUR_OUTPUT_DIR> \
  ```

### 4. Guidance for Training and Evaluation
Here we show how to train your own model for the IIGN task and evaluate it on VL-LN Bench.

- Prerequisites
  ```bash
  cd InternNav
  # Link VL-LN Bench data into InternNav
  mkdir projects && cd projects
  ln -s /path/to/your/VL-LN-Bench ./VL-LN-Bench
  ```
  - Write your OpenAI API key to api_key.txt.
  ```bash
  # Your final repo structure may look like
  InternNav
  ├── assets/
  ├── internnav/
  │   ├── habitat_vlln_extensions
  │   │   ├── simple_npc
  │   │   │   ├── api_key.txt
  │   ... ... ...
  ...
  ├── projects
  │   ├── VL-LN-Bench/
  │   │   ├── base_model/ 
  │   │   ├── raw_data/ 
  │   │   ├── scene_datasets/
  │   │   ├── traj_data/
  ... ...
  ```

- Start Training
  ```bash
  # Before running, please open this script and make sure 
  # the "llm" path points to the correct checkpoint on your machine.
  sbatch ./scripts/train/qwenvl_train/train_system2_vlln.sh
  ```

- Start Evaluation
  ```bash
  # If having slurm
  sh ./scripts/eval/bash/srun_eval_dialog.sh
  
  # Or directly run
  python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_dialog_cfg.py
  ```
