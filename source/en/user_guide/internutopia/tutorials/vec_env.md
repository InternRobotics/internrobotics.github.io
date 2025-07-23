# Vec Env

> Before reading this document, please read the [Gym Env tutorial](gym_env.md).

> The Vec Env provides speed-up in the steps taken per second by running multiple independent environments in parallel.

In most cases, we use `internutopia.core.gym_env` as the simulation execution environment (i.e., single environment), and tasks are executed by looping through all the `episodes`. However, in certain situations, we can achieve parallel simulation and improve efficiency through **vectorization**.


## Usage

An example of running two environments in parallel looks like following:
```{code-block} python
:emphasize-lines: 2,35,36,66,67

from internutopia.core.config import Config, SimConfig
from internutopia.core.vec_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.h1 import (
    H1RobotCfg,
    h1_camera_cfg,
    h1_tp_camera_cfg,
    move_along_path_cfg,
    move_by_speed_cfg,
    rotate_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

headless = False
if not has_display():
    headless = True

h1 = H1RobotCfg(
    position=(0.0, 0.0, 1.05),
    controllers=[
        move_by_speed_cfg,
        move_along_path_cfg,
        rotate_cfg,
    ],
    sensors=[
        h1_camera_cfg.update(name='camera', resolution=(320, 240), enable=False),
        h1_tp_camera_cfg.update(enable=False),
    ],
)

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless),
    env_num=2,
    env_offset_size=10,
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            scene_scale=(0.01, 0.01, 0.01),
            robots=[h1],
        ),
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            scene_scale=(0.01, 0.01, 0.01),
            robots=[h1],
        ),
    ],
)

print(config.model_dump_json(indent=4))

import_extensions()

env = Env(config)
obs, _ = env.reset()
print(f'========INIT OBS{obs}=============')

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

move_action = {move_along_path_cfg.name: [path]}

while env.simulation_app.is_running():
    i += 1
    action = {'h1': move_action}
    obs, _, terminated, _, _ = env.step(action=[action, action])
    if i % 100 == 0:
        print(i)

env.close()
```

The code above behaves as follows:

1. The simulation will run two environments.
2. The distance between the two environments is 10 units (default is meters).

> By default, the environments will be distributed as close to a square layout as possible on the plane (with the side length not exceeding env_offset_size * sqrt(env_num)).

## Env Reset

Similar to `gym_env`, the outputs of `env.reset(reset_list)` in `vec_env` are `obs` and `info`, but both are **lists**.

### Input
`reset_list`: a list of environment IDs (`env_id`) to reset.

If `env_num = 4`, valid values in `reset_list` can be `0, 1, 2, 3`. Any other value will raise an error.

### Example

For example, if `reset_list = [3, 0]`, the returned `obs` and `info` both have length 2:
- `obs[0]` and `info[0]` correspond to the environment with `env_id = 3`.
- `obs[1]` and `info[1]` correspond to the environment with `env_id = 0`.

### Edge Case

If there is only **one episode** left in `task_configs`, `env.reset([3, 0])` still returns lists of length 2：
- `obs[0]` and `info[0]` will contain valid data for `env_id = 3`.
- `obs[1]` and `info[1]` (for `env_id = 0`) will be `None`.

## Env Step

Similar to `gym_env`'s `step(action)`, the return values are `obs`, `reward`, `terminated`, `truncated`, and `info`. The difference is that in `vec_env`, these are all **lists** of length `env_num`, where each index corresponds to the return value for the respective environment (`env_id`).

If `task_configs` is exhausted but some environments are still running (e.g., `env_num = 4` and `env_1`, `env_3` have finished):
- The length of each returned list remains `env_num`.
- The values at the positions of the finished environments (e.g., `env_1` and `env_3`) will be `None`.
