# How to Add Custom Robot

> This tutorial guides you on how to add a custom robot.

To add a custom robot, you need to:
- Create a config class for robot config, inheriting from the `internutopia.core.config.robot.RobotCfg`.
- Create a class for robot, inheriting from the `internutopia.core.robot.BaseRobot`.

In this tutorial we take Unitree G1 as an example to show how to add a custom robot.

## Prepare USD of robot

We use [USD](https://openusd.org/release/index.html) file to represent a robot. If you have the URDF/MJCF file of the robot, you can refer to the links below to convert it to a USD file:

- Import URDF: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_urdf.html
- Import MJCF: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_mjcf.html

## Create Config Class

Here's an example of a config class for a robot:

```Python
from typing import Optional

from internutopia.core.config import RobotCfg

class G1RobotCfg(RobotCfg):
    name: Optional[str] = 'g1'
    type: Optional[str] = 'G1Robot'
    prim_path: Optional[str] = '/g1'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/g1/g1_29dof_color.usd'
```

- name: name of the robot, must be unique in one episode
- type: type of the robot, must be unique and same with the type of corresponding robot class
- prim_path: prim path of the robot, relative to the env root path
- usd_path: USD file path, absolute path is preferred to run your code from any working directory

More fields can be added if more attributes are configurable.

Generally, when creating a new config class for robot, reasonable default values for required fields should be specified to avoid validating error, and robot specific config fields can be added when necessary.

## Create Robot Class

In the simplest scenario, the following methods are required to be implemented in your robot class:

```python
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene


@BaseRobot.register('G1Robot')  # Register this robot to internutopia
class G1Robot(BaseRobot):
    def __init__(self, config: G1RobotCfg, scene: IScene):
        """Initialize the robot with the given config.

        Args:
            config (G1RobotCfg): config for the robot, should be a instance of corresponding config class.
            scene (IScene): current scene.
        """

    def post_reset(self):
        """Set up things after the env resets."""

    def apply_action(self, action: dict):
        """Apply actions of controllers to robot.

        Args:
            action (dict): action dict.
              key: controller name.
              value: corresponding action array.
        """

    def get_obs(self) -> OrderedDict:
        """Get observation of robot, including controllers, sensors, and world pose. OrderedDict is used to ensure the order of observations.
        """
```

The `apply_action` method are used to apply the provided actions, and `get_obs` to obtain the robot's current observations in each step.

For complete list of robot methods, please refer to the [Robot API documentation](../../../api/robot.rst).

Please note that the registration of the robot class is done through the `@BaseRobot.register` decorator, and the registered name should match the value of `type` field within the corresponding robot config class (here is `G1Robot`).

[`IArticulation`](../../../api/articulation.rst) is an interface class to deal with any articulated object in InternUtopia. Robot is one kind of articulated object, so here we use it to control the robot.

For users who want to get state of a certain rigid link of robot, we provide the `self._rigid_body_map` containing all rigid links of the robot with prim path as key. All the links within are of [`IRigidBody`](../../../api/rigidbody.rst) type which is an interface class to deal with rigid bodies.

An example of g1 robot class implementation is shown as following:

```python
import numpy as np

from internutopia.core.config.robot import RobotUserConfig as Config
from internutopia.core.robot.articulation import IArticulation
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log

from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene


@BaseRobot.register('G1Robot')  # Register this robot to internutopia
class G1Robot(BaseRobot):

        def __init__(self, config: G1RobotCfg, scene: IScene):  # Use the config class for this robot
        super().__init__(config, scene)
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'G1 {config.name}: position    : ' + str(self._start_position))
        log.debug(f'G1 {config.name}: orientation : ' + str(self._start_orientation))

        usd_path = config.usd_path

        log.debug(f'G1 {config.name}: usd_path         : ' + str(usd_path))
        log.debug(f'G1 {config.name}: config.prim_path : ' + str(config.prim_path))

        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
        # Create articulation handler
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=usd_path,
            scale=self._robot_scale,
        )

        # More initialization here...

    def post_reset(self):
        super().post_reset()
        self._robot_base = self._rigid_body_map[self.config.prim_path + '/base']

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warning(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.articulation.apply_action(control)

    def get_obs(self) -> OrderedDict:
        position, orientation = self._robot_base.get_pose()

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'joint_positions': self.articulation.get_joint_positions(),
            'joint_velocities': self.articulation.get_joint_velocities(),
            'controllers': {},
            'sensors': {},
        }

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs[c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs[sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)
```


You can check the implementations of our robots under [`internutopia_extension/robots/g1.py`](https://github.com/InternRobotics/InternUtopia/tree/main/internutopia_extension/robots/g1.py).
