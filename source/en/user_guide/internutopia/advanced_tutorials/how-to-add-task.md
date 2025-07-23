# How to Add Custom Task

> This tutorial explains how to add a custom task.

## 1. Defining a New Task
Before adding a new task, we need to clarify the following issues:

- Task Name: What will the task be called?
- Task Objective: What specific Objective will the task achieve?
- Task Termination: Will the task end, and how will we determine that?


Here is how we define FiniteStepTask based on the above issues:
- Name: FiniteStepTask
- Objective: Demo
- Termination:
  - The Task will end.
  - End Condition: The task will end either after finite steps.


To add a custom task, we need to:
- Create a config class for task config, inheriting from the `internutopia.core.config.task.TaskCfg`.
- Create a class for task, inheriting from the `internutopia.core.task.BaseTask`.


## 2. Create Task Config Class

Here's how we define the `FiniteStepTask` based on the above considerations:

```{code-block} python
:emphasize-lines: 8-9

from typing import Optional

from internutopia.core.config.task import TaskCfg


class FiniteStepTaskCfg(TaskCfg):
    type: Optional[str] = 'FiniteStepTask'
    max_steps: Optional[int] = 500

```

Respect the following rules when defining a task config class:

- Define a unique `type` name
- Define new task-specific parameters directly in the config (e.g., `max_steps`)
- Avoid overriding other parameters defined in `internutopia.core.config.task.TaskCfg`


## 3. Create Task Class

```Python
from internutopia.core.scene.scene import IScene
from internutopia.core.task import BaseTask
from internutopia_extension.configs.tasks.finite_step_task import FiniteStepTaskCfg


@BaseTask.register('FiniteStepTask')
class FiniteStepTask(BaseTask):
    def __init__(self, config: FiniteStepTaskCfg, scene: IScene):
        super().__init__(config, scene)
        self.stop_count = 0
        self.max_steps = config.max_steps

    def is_done(self) -> bool:
        self.stop_count += 1
        return self.stop_count > self.max_steps
```

The `is_done` method has been overridden based on the End Condition defined above.


## 4. Task Usage Preview
To use the custom task, we can simply include them in the configuration settings as follows

```{code-block} python
:emphasize-lines: 17-20

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

import_extensions()

headless = False
if not has_display():
    headless = True

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless),
    task_configs=[
        FiniteStepTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            max_steps = 300,
        ),
    ]
)


env = Env(config)
obs, _ = env.reset()

i = 0

while env.simulation_app.is_running():
    obs, _, terminated, _, _ = env.step(action={})

    if i % 1000 == 0:
        print(i)

    i += 1

env.simulation_app.close()

env = Env(config)
...
```

This task will terminate after 300 steps.
