# How to Use Task

> This tutorial guides you on how to run a task.

## Pre-defined Tasks

The directory [`grutopia_extension/tasks/__init__.py`](https://github.com/OpenRobotLab/GRUtopia/blob/main/grutopia_extension/tasks/__init__.py) contains a list of all our pre-defined tasks:

```Python
from grutopia_extension.tasks import (
    manipulation_task,
    finite_step_task,
    single_inference_task,
)
```

We can also review the configuration of each task in [`grutopia_extension/configs/tasks/__init__.py`](https://github.com/OpenRobotLab/GRUtopia/blob/main/grutopia_extension/configs/tasks/__init__.py).


## How to Use Task

To use an existing task within GRUtopia, you can simply use the corresponding type of task config in the runtime configuration as following:

```{code-block} python
:emphasize-lines: 14-18

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

headless = False
if not has_display():
    headless = True

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
        )
    ],
)

import_extensions()

env = Env(config)
obs, _ = env.reset()

i = 0

while env.simulation_app.is_running():
    i += 1
    obs, _, terminated, _, _ = env.step(action={})

    if i % 1000 == 0:
        print(i)

env.simulation_app.close()
```

<video width="720" height="405" controls>
    <source src="../../../_static/video/tutorial_use_task.webm" type="video/webm">
</video>

> In InternUtopia, `task_configs` is a list in which each element represents one episode. This is one of the key points that distinguish InternUtopia with other platforms.
