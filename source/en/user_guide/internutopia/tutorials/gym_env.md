# Gym Env

> This is an `Env` class that implements the same interface as `gym.Env`.

# Usage

```{code-block} python
:emphasize-lines: 2,53

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
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

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

move_action = {move_along_path_cfg.name: [path]}

while env.simulation_app.is_running():
    i += 1
    action = move_action
    obs, reward, terminated, truncated, info = env.step(action=action)
    if i % 100 == 0:
        print(i)

env.close()
```

## Env Reset

Unlike the standard `gym.Env`, the `internutopia.core.gym_env.Env` class provides a `reset()` method that does **not** reset the environment to its initial state.

In InternUtopia, `task_configs` is a list where each element is referred to as an `episode`.

The purpose of `reset()` is to **load the next episode**.

The return values of `reset()` are `obs` and `info`:

1. When there are episodes remaining, `reset()` returns the `obs` and `task_config` of the newly loaded episode.
2. When all episodes have been executed, `reset()` returns `None` and `None`.

You can break the `while env.simulation_app.is_running()` loop when `info` is `None` to stop the simulation.

## Env Step

The `env.step(action)` runs one timestep of the environment’s dynamics using the actions.

There are five outputs from `env.step(action)`:
1. `obs`: The observation of the robot in the env.
2. `reward`: The reward of the env.
3. `terminated`: Whether the episode has ended.
4. `truncated`: Not used yet.
5. `info`: Not used yet.

When the current episode ends, `env.reset()` is **not called automatically**. Its return values will all be `None` (i.e., the return from `env.step(action)` will be `(None, None, None, None, None)`). You must **manually call** `env.reset()` to load a new episode.
