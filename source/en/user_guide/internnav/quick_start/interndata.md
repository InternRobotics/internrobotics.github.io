# InternData-N1 Dataset Preparation

We prepare high-quality data for **training** system1/system2 and **evaluation** on isaac sim environment. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment. Download the [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) and [SceneData-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1).

To set up the dataset, please follow the steps below:

1. Download Datasets
- Download the [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1) for:
   - `vln_pe/`
   - `vln_ce/`
   - `vln_n1/`

- Download the [SceneData-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1) for the `scene_data/`.

2. Directory Structure

After downloading, organize the datasets into the following structure:

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
|   └── traj_data/
│       └── mp3d/
│           └── 17DRP5sb8fy/
│           └── 1LXtFkjw3qL/
│           └── ...
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