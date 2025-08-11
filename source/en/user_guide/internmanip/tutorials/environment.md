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
            eval_tasks=[],
            res_save_path="path/to/save/results",
            is_save_img=False,
            robot_type="aloha_split",
            gripper_type="panda",
            franka_camera_enable=FrankaCameraEnable(
                realsense=False, obs_camera=False, obs_camera_2=False
            ),
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=False, left_camera=False, right_camera=False
            ),
            depth_obs=False,
            max_step=1000,
            max_success_step=100,
            env_num=1,
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
| `eval_tasks` | list[str] | [] | The relative path to the task folder under dataset_path. <br>If `[]`, all tasks in the `dataset_path` will be automatically detected. |
| `res_save_path` | str | None | Evaluation results storage directory (disabled if None).|
| `is_save_img` | bool | False | Enable per-step multi-camera image capture (requires disk space) |
| `robote_type` | enum | "aloha_split" | robot selection (`"franka"` or `"aloha_split"`) |
| `gripper_type` | enum | "panda" | End effector selection **when robote_type is `franka`** (`"panda"` or `"robotiq"`) |
| `franka_camera_enable` | FrankaCameraEnable | `FrankaCameraEnable(`<br>`realsense=False, `<br>`obs_camera=False, `<br>`obs_camera_2=False)` | Camera activation config:<br>• `realsense`: Ego-view gripper cam<br>• `obs_camera`: Fixed rear-view cam<br>• `obs_camera_2`: Fixed front-view cam <br> Note that this only works **when robote_type is `franka`** |
| `aloha_split_camera_enable` | AlohaSplitCameraEnable | `AlohaSplitCameraEnable(`<br>`top_camera=False, `<br>`left_camera=False, `<br>`right_camera=False)` | Camera activation config:<br>• `top_camera`: Ego-view top head cam<br>• `left_camera`: Ego-view left gripper cam<br>• `right_camera`: Ego-view right gripper cam <br> Note that this only works **when robote_type is `aloha_split`** |
| `depth_obs` | bool | False | Generate depth maps for active cameras |
| `max_step` | int | 1000 | Episode termination step threshold |
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

### 3.1 when robote_type is **franka**

**Observation Structure**

```python
observations: List[Dict] = [
    {
        "robot": {
            "robot_pose": Tuple[array, array], # (position, oritention(quaternion: (w, x, y, z)))
            "joints_state": {
                "positions": array, # (9,) or (13,) -> panda or robotiq
                "velocities": array # (9,) or (13,) -> panda or robotiq
            },
            "eef_pose": Tuple[array, array], # (position, oritention(quaternion: (w, x, y, z)))
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
Agents must output `List[Union[List[float], dict]] = [action_1, action_2 , ...]` of the same length as the input observations.
```python
actions: List[Union[List[float], dict]] = [
    action_1,
    action_2,
    ...
]
```

The `action_x` supports any of the following formats:

**ActionFormat1**:
```python
List[float] # (9,) or (13,) -> panda or robotiq
```
**ActionFormat2**:
```python
{
    'arm_action': List[float], # (7,)
    'gripper_action': Union[List[float], int], # (2,) or (6,) -> panda or robotiq || -1 or 1 -> open or close
}
```
**ActionFormat3**:
```python
{
    'eef_position': List[float], # (3,) -> (x, y, z)
    'eef_orientation': List[float], # (4,) -> (quaternion: (w, x, y, z))
    'gripper_action': Union[List[float], int], # (2,) or (6,) -> panda or robotiq || -1 or 1 -> open or close
}
```

---

### 3.2 when robote_type is **aloha_split**

**Observation Structure**

```python
observations: List[Dict] = [
    {
        "robot": {
            "robot_pose": Tuple[array, array], # (position, oritention(quaternion: (w, x, y, z)))
            "joints_state": {
                "positions": array, # (28,)
                "velocities": array # (28,)
            },
            "left_eef_pose": Tuple[array, array], # (position, oritention(quaternion: (w, x, y, z))) -> left gripper eef pose
            "right_eef_pose": Tuple[array, array], # (position, oritention(quaternion: (w, x, y, z))) -> right gripper eef pose
            "sensors": {
                "top_camera": {
                    "rgb": array, # uint8 (480, 640, 3)
                    "depth": array, # float32 (480, 640)
                },
                "left_camera": {
                    "rgb": array,
                    "depth": array,
                },
                "right_camera": {
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
> **Note the following when using `observations["robot"]["joints_state"]["positions"]`.**
> 
> The aloha_split robot's dof names (in order) as follows: `['mobile_translate_x', 'mobile_translate_y', 'mobile_rotate', 'fl_steering_joint', 'fr_steering_joint', 'rl_steering_joint', 'rr_steering_joint', 'lifting_joint', 'fl_wheel', 'fr_wheel', 'rl_wheel', 'rr_wheel', 'fl_joint1', 'fr_joint1', 'fl_joint2', 'fr_joint2', 'fl_joint3', 'fr_joint3', 'fl_joint4', 'fr_joint4', 'fl_joint5', 'fr_joint5', 'fl_joint6', 'fr_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint7', 'fr_joint8']`.
> 
> Thus:
> - The left arm coordinate: `[12, 14, 16, 18, 20, 22]`
> - The left gripper coordinate: `[24, 25]`
> - The right arm coordinate: `[13, 15, 17, 19, 21, 23]`
> - The right gripper coordinate: `[26, 27]`
> 
> Example:  
> If you want to use the joint position values of the left arm, you should do this:
> ```python
> left_arm_joint_indices = [12, 14, 16, 18, 20, 22]
> left_arm_joint_positions = [observations['robot']['joints_state']['positions'][idx] for idx in left_arm_joint_indices]
> ```

---  

**Action Space Specifications**
Agents must output `List[dict] = [action_1, action_2 , ...]` of the same length as the input observations.
```python
actions: List[dict] = [
    action_1,
    action_2,
    ...
]
```

The `action_x` supports any of the following formats:

**ActionFormat1**:
```python
{
    'left_arm_action': List[float], # (6,)
    'left_gripper_action': Union[List[float], int], # (2,) || -1 or 1 -> open or close
    'right_arm_action': List[float], # (6,)
    'right_gripper_action': Union[List[float], int], # (2,) || -1 or 1 -> open or close
}
```
**ActionFormat2**:
```python
{
    'left_eef_position': List[float], # (3,) -> (x, y, z)
    'left_eef_orientation': List[float], # (4,) -> (quaternion: (w, x, y, z))
    'left_gripper_action': Union[List[float], int], # (2,) || -1 or 1 -> open or close
    'right_eef_position': List[float], # (3,) -> (x, y, z)
    'right_eef_orientation': List[float], # (4,) -> (quaternion: (w, x, y, z))
    'right_gripper_action': Union[List[float], int], # (2,)|| -1 or 1 -> open or close
}
```

**None Handling Protocol**: If observation element is `None` or invalid value, corresponding action must be `[]`.
