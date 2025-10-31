# Dataset Preparation

We prepared high-quality data for **training** system1/system2 and **evaluation** on isaac sim and habitat sim environment. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment.


## Data and Checkpoints Checklist
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
- Download the [Embodiments](https://huggingface.co/datasets/InternRobotics/Embodiments) and place it under the `Embodiments/`. These embodiment assets are used by the Isaac Sim environment.

The final folder structure should look like this:

```bash
InternNav/
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
│   └── r2r
│       ├── fine_tuned
│       └── zero_shot
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
|   ├── vln_n1/
|   |   └── traj_data/
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
├── internnav/
│   └── ...
```