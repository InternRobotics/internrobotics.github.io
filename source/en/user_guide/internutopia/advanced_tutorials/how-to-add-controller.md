# How to Add Custom Controller

> This tutorial guides you on how to add a custom controller for a robot.

Note that the controller cannot be operated independently. It must be used with robot to enable robot to act in the environment.


To add a custom controller, you need to:
- Create a config class for controller config, inheriting from the `internutopia.core.config.robot.ControllerCfg`.
- Create a class for controller, inheriting from the `internutopia.core.robot.controller.BaseController`.

In this tutorial we take Differential Drive Controller as an example to show how to add a custom controller.

## Create Config Class

Here's an example of a config class for a controller:

```Python
from typing import Optional

from internutopia.core.config.robot import ControllerCfg


class DifferentialDriveControllerCfg(ControllerCfg):

    type: Optional[str] = 'DifferentialDriveController'
    wheel_radius: float
    wheel_base: float
```

For differential drive controller, we specify the type and add new fields `wheel_radius` and `wheel_base` to specify the radius and base of the wheels.

## Create Controller Class

In the simplest scenario, the following methods are required to be implemented in your controller class:

```python
import numpy as np
from typing import List, Union

from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot


@BaseController.register('DifferentialDriveController')
class DifferentialDriveController(BaseController):
    def __init__(self, config: DifferentialDriveControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        """Initialize the controller with the given configuration and its owner robot.

        Args:
            config (DifferentialDriveControllerCfg): controller configuration.
            robot (BaseRobot): robot owning the controller.
            scene (IScene): scene interface.
        """

    def action_to_control(self, action: Union[np.ndarray, List]) -> ArticulationAction:
        """Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (Union[np.ndarray, List]): input control action.

        Returns:
            ArticulationAction: joint signals to apply
        """
```

The `action_to_control` method translates the input action into joint signals to apply in each step.

For complete list of controller methods, please refer to the [Controller API documentation](../../api/robot.rst#module-internutopia.core.robot.controller).

Please note that the registration of the controller class is done through the `@BaseController.register` decorator, and the registered name **MUST** match the value of `type` field within the corresponding controller config class (here is `DifferentialDriveController`).

Sometimes the calculation logic is defined in a method named `forward` to show the input parameters the controller accepts (which is common in our implementations), making it more human-readable. In this case, the `action_to_control` method itself only expands the parameters, and invokes `forward` method to calculate the joint signals.

An example of controller class implementation is shown as following:

```python
from typing import List

import numpy as np

from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.controllers import DifferentialDriveControllerCfg


@BaseController.register('DifferentialDriveController')
class DifferentialDriveController(BaseController):
    def __init__(self, config: DifferentialDriveControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        self._robot_scale = self.robot.get_robot_scale()[0]
        self._wheel_base = config.wheel_base * self._robot_scale
        self._wheel_radius = config.wheel_radius * self._robot_scale

    def forward(
        self,
        forward_speed: float = 0,
        rotation_speed: float = 0,
    ) -> ArticulationAction:
        left_wheel_vel = ((2 * forward_speed) - (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        right_wheel_vel = ((2 * forward_speed) + (rotation_speed * self._wheel_base)) / (2 * self._wheel_radius)
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=[left_wheel_vel, right_wheel_vel])

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): n-element 1d array containing:
              0. forward_speed (float)
              1. rotation_speed (float)
        """
        assert len(action) == 2, 'action must contain 2 elements'
        return self.forward(
            forward_speed=action[0],
            rotation_speed=action[1],
        )

```

You can check implementations of our controllers under [`internutopia_extension/controllers/`](https://github.com/InternRobotics/InternUtopia/tree/main/internutopia_extension/controllers), within which you can find:

- Rule-based controllers: such as `ik_controller`, `dd_controller`.
- Learning-based controllers: such as `g1_move_by_speed_controller`, `h1_move_by_speed_controller`.
- Composite controllers: controllers chained with other controller, such as `move_to_point_by_speed_controller`, `rotate_controller`.
