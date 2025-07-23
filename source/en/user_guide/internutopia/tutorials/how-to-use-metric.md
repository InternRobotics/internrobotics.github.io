# How to Use Metric

> Metric is a tool for recording statistics during task execution.

> This tutorial guides you on how to use a metric.

## Pre-defined Metric

The directory [`grutopia_extension/metrics/__init__.py`](https://github.com/OpenRobotLab/GRUtopia/blob/main/grutopia_extension/metrics/__init__.py) contains a list of all our pre-defined metrics:

```Python
from grutopia_extension.metrics import (
    traveled_distance_metric,
    recording_metric
)
```

We can also review the configuration of each metric in [`grutopia_extension/configs/metrics/__init__.py`](https://github.com/OpenRobotLab/GRUtopia/blob/main/grutopia_extension/configs/metrics/__init__.py).


## How to Use Metric

To use an existing metric within GRUtopia, you can simply use the corresponding type of metric config in the task configuration as following:

```{code-block} python
:emphasize-lines: 6,36,40

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.metrics.traveled_distance_metric import TraveledDistanceMetricCfg
from internutopia_extension.configs.robots.h1 import (
    H1RobotCfg,
    h1_camera_cfg,
    h1_tp_camera_cfg,
    move_along_path_cfg,
    move_by_speed_cfg,
    rotate_cfg,
)
from internutopia_extension.configs.tasks import FiniteStepTaskCfg

headless = False
if not has_display():
    headless = True

h1_1 = H1RobotCfg(
    position=(0.0, 0.0, 1.05),
    controllers=[
        move_by_speed_cfg,
        move_along_path_cfg,
        rotate_cfg,
    ],
    sensors=[
        h1_camera_cfg.update(name='camera', resolution=(320, 240), enable=True),
        h1_tp_camera_cfg.update(enable=False),
    ],
)

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless),
    metrics_save_path='./h1_traveled_distance_metric.jsonl',
    task_configs=[
        FiniteStepTaskCfg(
            max_steps=300,
            metrics=[TraveledDistanceMetricCfg(name='robot_movement',robot_name='h1')],
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            scene_scale=(0.01, 0.01, 0.01),
            robots=[h1_1],
        ),
    ],
)

import_extensions()

env = Env(config)
obs, _ = env.reset()
print(f'========INIT OBS{obs}=============')

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

move_action = {move_along_path_cfg.name: [path]}

while env.simulation_app.is_running():
    i += 1
    action = move_action
    obs, _, terminated, _, _ = env.step(action=action)

    if terminated:
        obs, info = env.reset()
        if info is None:  # No more episode
            break

    if i % 100 == 0:
        print(i)

env.close()

```

You can also run the [`h1_traveled_distance.py`](https://github.com/OpenRobotLab/GRUtopia/blob/main/grutopia/demo/h1_traveled_distance.py) in the demo directly.

And you can check result in `./h1_traveled_distance_metric.jsonl`, the key of output json is the `name` of metrics.

```json
{"traveled_distance_metric": 0.7508679775492055}
```

## Metric save

The `metrics_save_path` parameter in the Config has three ways to be filled:

1. `None` or `'none'`: Do not save.
2. `'console'`: Output to the console.
3. Other: Save to the specified path.

When multiple episodes are executed, the results from all episodes will be saved in a single file (in JSON Lines format).

Each time the process starts, a new file will be created at the `metrics_save_path`, and any existing content in that file will be cleared.
