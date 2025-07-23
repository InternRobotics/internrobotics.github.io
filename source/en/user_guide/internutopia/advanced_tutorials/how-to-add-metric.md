# How to Add Custom Metric

Because metrics are typically designed for specific tasks, there is rarely a completely "universal" metric.

Therefore, after customizing the task, you usually need to customize the metric to record the info you need.

## 1. Defining a New Metric
Before adding a new metric, we need to clarify the following issues:

- Metric Name: What will the metric be called?
- Metric Objective: What specific Objective will the metric achieve?
- Metric Update Method: Record the status of the metric while task is running
- Metrics Calculation: Summarize all records at the end of the task


Here is how we define TraveledDistanceMetric based on the above issues:

  - Record the total distance a robot moves from start to finish.
  - (Additional metrics as needed.)

To add a custom metric, you need to:
- Create a config class for metric config, inheriting from the `internutopia.core.config.metric.MetricCfg`.
- Create a class for metric, inheriting from the `internutopia.core.task.metric.BaseMetric`.


## 2. Create Metrics Config Class

Here's an example of a config class for a metric:

```python
# This is also the simplest configuration.
from typing import Optional

from internutopia.core.config.metric import MetricCfg


class TraveledDistanceMetricCfg(MetricCfg):
    name: Optional[str] = 'traveled_distance_metric'
    type: Optional[str] = 'TraveledDistanceMetric'
    robot_name: str
```

- Define a unique type name.
- Define new parameters needed by the metric directly in the config.


## 3. Create Metrics Class

In this doc, we demonstrate a simple metrics used to track the total distance a robot moves.

```Python
import numpy as np
from pydantic import BaseModel

from internutopia.core.config import TaskCfg
from internutopia.core.task.metric import BaseMetric
from internutopia.core.util import log
from internutopia_extension.configs.metrics.traveled_distance_metric import TraveledDistanceMetricCfg


@BaseMetric.register('TraveledDistanceMetric')
class TraveledDistanceMetric(BaseMetric):
    """
    Calculate the total distance a robot moves
    """

    def __init__(self, config: TraveledDistanceMetricCfg, task_config: TaskCfg):
        super().__init__(config, task_config)
        self.distance: float = 0.0
        self.position = None
        self.robot_name = config.robot_name

    def update(self, obs: dict):
        """
        This function is called at each world step.
        """
        if self.position is None:
            self.position = obs[self.robot_name]['position']
            return
        self.distance += np.linalg.norm(self.position - obs[self.robot_name]['position'])
        self.position = obs[self.robot_name]['position']
        return

    def calc(self):
        """
        This function is called to calculate the metrics when the episode is terminated.
        """
        log.info('TraveledDistanceMetric calc() called.')
        return self.distance

```
- The `update` method will be invoked after every step.
   - Do not use computationally intensive operations. This is performed by a For loop.
   - The received `obs` is a dict, where the key is the robot name and the value corresponds to the obs output from gym_env. It's similar to the vec_env observation format.
- The `calc` method will be invoked at the end of an episode.


## 4. Metric Usage Preview

To use the custom metrics, you can simply include them in the configuration settings as follows

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
            metrics=[TraveledDistanceMetricCfg(robot_name='h1')],
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
        if info is None:
            break

env.close()
```

And you can check result in `./h1_traveled_distance_metric.jsonl`, the key of output json is the `name` of metrics.

```json
{"traveled_distance_metric": 0.7508679775492055}
```
