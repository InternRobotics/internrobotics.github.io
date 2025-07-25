# Environment

This document provides the guide of environment installation, config parameters description and I/O Specifications.

## Table of Contents
1. [Calvin](#Calvin)
2. [Simpler-env](#Simpler-env)
3. [Genmanip](#Genmanip)


# Calvin  (WIP)

# Simpler-env  (WIP)

# Genmanip
## 1. Environment Dependency Installation (Conda)

Step 1‌: Verify installation of Anaconda and Isaac Sim 4.5.0. You can download Isaac Sim 4.5.0 from [here](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip) and extract to target directory.

Step 2‌: Execute installation script:

```bash
./install.sh --genmanip

# !Note
# - When prompted, provide Isaac Sim 4.5 folder path
# - Post-installation, activate Conda environment: `conda activate genmanip`
```

## 2. Evaluation Environment Configuration Parameters

Evaluation requires configuration file declaring `evalcfg.env.env_setting` as `GenmanipEnvSettings` implementation:

```python
eval_cfg = EvalCfg(
    eval_type="genmanip",
    agent=AgentCfg(...),
    env=EnvCfg(
        env_type="genmanip",
        env_settings=GenmanipEnvSettings(
            dataset_path="path/to/genmanip/benchmark_data",
            eval_tasks=["task1", "task2", ...],
            res_save_path="path/to/save/results",
            is_save_img=False,
            camera_enable=CameraEnable(realsense=False, obs_camera=False, obs_camera_2=False),
            depth_obs=False,
            gripper_type="panda",
            env_num=1,
            max_step=500,
            max_success_step=100,
            physics_dt=1/30,
            rendering_dt=1/30,
            headless=True,
            ray_distribution=None,
        )
    ),
    ...
)
```


**Core Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | str | None | If None, it will be automatically downloaded from Hugging Face dataset [URL](https://huggingface.co/datasets/OpenRobotLab/InternBench-M1) on the first run. |
| `eval_tasks` | list[str] | ALL_EVAL_TASKS | Subset of 18 predefined tasks (validate against `internmanip.configs.env.genmanip_env.ALL_EVAL_TASKS`) |
| `res_save_path` | str | None | Evaluation results storage directory (disabled if None).|
| `is_save_img` | bool | False | Enable per-step multi-camera image capture (requires disk space) |
| `camera_enable` | CameraEnable | `CameraEnable(realsense=False, `<br>`obs_camera=False, obs_camera_2=False)` | Camera activation config:<br>• `realsense`: Ego-view gripper cam<br>• `obs_camera`: Fixed rear-view cam<br>• `obs_camera_2`: Fixed front-view cam |
| `depth_obs` | bool | False | Generate depth maps for active cameras |
| `gripper_type` | enum | "panda" | End effector selection (`"panda"` or `"robotiq"`) |
| `max_step` | int | 500 | Episode termination step threshold |
| `headless` | bool | True | Disable GUI |


**Advanced Parameters**
| Parameter | Scope |
|-----------|-------|
| `max_success_step` | Early termination upon task completion |
| `physics_dt` | Physics simulation timestep|
| `rendering_dt` | Render interval|
| `env_num` | Concurrent environment instances in one isaac sim |
| `ray_distribution` | Multi-process config (`RayDistributionCfg`):<br>• `proc_num`: Process count<br>• `gpu_num_per_proc`: GPUs per process # <br>e.g., `RayDistributionCfg(proc_num=2, gpu_num_per_proc=0.5, head_address=None, working_dir=None)`|

> **Concurrency Note**: When `env_num > 1` or `ray_distribution.proc_num > 1`, environment outputs become multi-instance tensors. Agents must process batched observations and return batched actions.

## 3. I/O Specifications: Environment Outputs and Action Data Formats
**Observation Structure**
```python
observations: List[Dict] = [
    {
        "franka_robot": {
            "robot_pose": Tuple[array, array], # (position, oritention(quaternion))
            "joints_state": {
                "positions": array,
                "velocities": array
            },
            "eef_pose": Tuple[array, array], # (position, oritention(quaternion))
            "sensors": {
                "realsense": {
                    "rgb": array, # uint8 (480, 640, 3)
                    "depth": array, # float32 (480, 640)
                },
                "obs_camera": {
                    "rgb": array,
                    "depth": array,
                },
                "obs_camera_2": {
                    "rgb": array,
                    "depth": array,
                },
            },
            "instruction": str,
            "metric": {
                "task_name": str,
                "episode_name": str,
                "episode_sr": int,
                "first_success_step": int,
                "episode_step": int
            },
            "step": int,
            "render": bool
        }
    },
    ...
]
```

**Action Space Specifications**
Agents must output `List[Union[ActionFormat1, ActionFormat2, ..., ActionFormat5]]` of the same length as the input observations.
```python
actions: List[Union[ActionFormat1, ActionFormat2, ..., ActionFormat5]] = [
    ActionFormat1,
    ActionFormat2,
    ...
]
```

Supported action formats:

**ActionFormat1**:
```python
List[float] # (9,) or (13,) -> panda or robotiq
```
**ActionFormat2**:
```python
{
    'arm_action': List[float], # (7,)
    'gripper_action': List[float], # (2,) or (9,) -> panda or robotiq
}
```
**ActionFormat3**:
```python
{
    'arm_action': List[float], # (7,)
    'gripper_action': int, # -1 or 1 -> open or close
}
```
**ActionFormat4**:
```python
{
    'eef_position': List[float], # (3,) -> (x, y, z)
    'eef_orientation': List[float], # (4,) -> (quaternion)
    'gripper_action': List[float], # (2,) or (9,) -> panda or robotiq
}
```
**ActionFormat5**:
```python
{
    'eef_position': List[float], # (3,) -> (x, y, z)
    'eef_orientation': List[float], # (4,) -> (quaternion)
    'gripper_action': int, # -1 or 1 -> open or close
}
```

**None Handling Protocol**: If observation element is `None` or invalid value, corresponding action must be `[]`.
