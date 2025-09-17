# VLN Evaluation

## Overview of the Evaluation Process
The main architecture of the evaluation code adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then submits tasks to the Ray distributed framework based on the corresponding cfg file, enabling the entire evaluation process to run.


## Supported baselines
- InternVLA-N1
- CMA (Cross-Modal Attention)
- RDP (Recurrent Diffusion Policy)
- Navid (RSS2023)
- Seq2Seq Policy

## Supported Datasets
- R2R-CE
- Matterport3D

## Evaluation Metrics
The project provides comprehensive evaluation metrics:

- **Success Rate (SR)**: Proportion of episodes where the agent reaches the goal location within 3m
- **SPL**: Success weighted by Path Length
- **Trajectory Length (TL)**: Total length of the trajectory (m)
- **Navigation Error (NE)**: Euclidean distance between the agent's final position and the goal (m)
- **OS Oracle Success Rate (OSR)**: Whether any point along the predicted trajectory reaches the goal within 3m
- **Fall Rate (FR)**: Frequency of the agent falling during navigation
- **Stuck Rate (StR)**: Frequency of the agent becoming stuck during navigation


# Quick Start for Evaluation

## 1. Start the ray server
```bash
ray disable-usage-stats
ray stop
ray start --head
```

## 2. Custom your evaluation config
```bash
eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8023,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
        },
    ),
    env=EnvCfg(
        env_type='vln_multi',
        env_settings={
            'use_fabric': True,     # improve simulation efficiency
            'headless': True,       # display option: set to False will open isaac-sim interactive window
        },
    ),
    task=TaskCfg(
        task_name='test',
        task_settings={
            'env_num': 1,           # number of env in one isaac sim
            'use_distributed': False,       # Ray distributed framework
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            mp3d_data_dir='/path/to/mp3d',
        ),
        robot_name='h1',
        robot_flash=True,
        robot_usd_path='/robots/h1/h1_vln_multi_camera.usd',
        camera_resolution=[640, 480] # (W,H)
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': '/path/to/R2R_VLNCE_v1-3',
            'split_data_types': ['val_unseen'],
            'filter_stairs': True,
        },
    eval_settings={
        'save_to_json': False,      # evaluation result saved in separate json file
        'vis_output': True          # save simulation progress to video under logs/
    }
    ),
```
## 3. Launch the server
```bash
INTERNUTOPIA_ASSETS_PATH=/path/to/InternUTopiaAssets MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config path/to/cfg.py
```

## 4. Launch the client
```bash
python -m internnav.agent.utils.server --config path/to/cfg.py
```
