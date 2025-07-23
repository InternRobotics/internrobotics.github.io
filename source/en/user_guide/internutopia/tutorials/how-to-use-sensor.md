# How to Use Sensor

> This tutorial guides you on how to use a sensor to retrieve corresponding observation data.

## Pre-defined Sensors

The directory [`internutopia_extension/sensors/`](https://github.com/InternRobotics/InternUtopia/tree/main/internutopia_extension/sensors) contains a list of all available sensors:

```
internutopia_extension/
└── sensors
    ├── mocap_controlled_camera.py
    └── rep_camera.py
    ...
```

For each robot, we provide some ready-to-use sensor configurations for each robot in `internutopia_extension/configs/robots/{robot_name}.py`.

## How to Use a Sensor

Typically, a sensor should be used with a robot. So first of all, the sensor configuration should be added to the sensor list in robot configuration:

```{code-block} python
:emphasize-lines: 9,27,46-47

from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia.core.util import has_display
from internutopia.macros import gm
from internutopia_extension import import_extensions
from internutopia_extension.configs.robots.jetbot import (
    JetbotRobotCfg,
    camera_cfg,
    move_by_speed_cfg,
)
from internutopia_extension.configs.tasks import SingleInferenceTaskCfg

import_extensions()

headless = not has_display()

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=headless, webrtc=headless),
    task_configs=[
        SingleInferenceTaskCfg(
            scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
            robots=[
                JetbotRobotCfg(
                    position=(0.0, 0.0, 0.0),
                    scale=(5.0, 5.0, 5.0),
                    controllers=[move_by_speed_cfg],
                    sensors=[camera_cfg],
                )
            ],
        ),
    ],
)

env = Env(config)
obs, _ = env.reset()

i = 0

while env.simulation_app.is_running():
    i += 1
    action = {move_by_speed_cfg.name: [0.5, 0.5]}
    obs, _, terminated, _, _ = env.step(action=action)

    if i % 1000 == 0:
        print(i)
        for k, v in obs['sensors'][camera_cfg.name].items():
            print(f'key: {k}, value: {v}')

env.close()
```

<video width="720" height="405" controls>
    <source src="../../../_static/video/tutorial_use_sensor.webm" type="video/webm">
</video>

In the above example, first we import the `move_by_speed_cfg` for jetbot. It'a a ready-to-use sensor config for jetbot to use the `Camera` to get observations:

[`internutopia_extension/configs/robots/jetbot.py`](https://github.com/InternRobotics/InternUtopia/blob/main/internutopia_extension/configs/robots/jetbot.py)

```python
camera_cfg = RepCameraCfg(
    name='camera',
    prim_path='chassis/rgb_camera/jetbot_camera',
    resolution=(640, 360),
)
```

The sensor config is then added to the robot config to declare it as an available sensor for the robot in that episode. In each step, we can read the observations from the `obs` dict returned by `env.step()`. Observations from certain sensor are stored in `obs['sensors'][{sensor_name}]`. The data structure of observation is defined by the `get_data` method of the specific sensor. For the above example, we can check it in the `RepCamera` class:

[`internutopia_extension/sensors/rep_camera.py`](https://github.com/InternRobotics/InternUtopia/blob/main/internutopia_extension/sensors/rep_camera.py)

```python
class Camera(BaseSensor):
    def get_data(self) -> OrderedDict:
        return self.get_camera_data()

    def get_camera_data(self) -> OrderedDict:
        ...
        if self.config.rgba:
            output_data['rgba'] = self._camera.get_rgba()
        ...
        return self._make_ordered(output_data)
```

So the rgba data captured would be printed every 1000 steps in our example.
