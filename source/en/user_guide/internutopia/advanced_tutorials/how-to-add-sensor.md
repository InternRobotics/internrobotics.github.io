# How to Add Custom Sensor

> This tutorial guides you on how to add a sensor for a robot.

The sensor serves as an abstraction layer that can either encapsulate native sensors provided by simulator or generate synthetic data outputs to emulate a sensor's behavior.

Currently a sensor instance belongs to a robot in the scene, and its data is included in the observation of that robot.

To add a custom sensor, you need to:
- Create a config class for sensor config, inheriting from the `internutopia.core.config.robot.SensorCfg`.
- Create a class for sensor, inheriting from the `internutopia.core.sensor.sensor.BaseSensor`.

## Create Config Class

Let's assume we need a depth camera.

The camera accepts the following config parameters:

- Flag to enable/disable the sensor.
- Camera resolution.

Here's an example of a config class for the sensor:

```Python
class DepthCameraCfg(SensorCfg):

    type: Optional[str] = 'DepthCamera'
    resolution: Optional[Tuple[int, int]] = (640, 480)
```

For depth camera, we specify the type and add a new field `resolution` to specify the resolution of the camera.

## Create Sensor Class

In the simplest scenario, the following methods are required to be implemented in your sensor class:

```python
from internutopia.core.robot.robot import BaseRobot, Scene
from internutopia.core.sensor.sensor import BaseSensor


@BaseSensor.register('DepthCamera')
class DepthCamera(BaseSensor):
    def __init__(self, config: DepthCameraCfg, robot: BaseRobot, scene: Scene):
        """Initialize the sensor with the given config.

        Args:
            config (DepthCameraCfg): sensor configuration.
            robot (BaseRobot): robot owning the sensor.
            scene (Scene): scene from isaac sim.
        """

    def get_data(self) -> Dict:
        """Get data from sensor.

        Returns:
            Dict: data dict of sensor.
        """
```

The `get_data` method gets the sensor data in each step.

For complete list of sensor methods, please refer to the [Sensor API documentation](../../api/robot.rst#module-internutopia.core.robot.sensor).

Please note that the registration of the sensor class is done through the `@BaseSensor.register` decorator, and the registered name should match the value of `type` field within the corresponding sensor config class (here is `DepthCamera`).

A interface class [`ICamera`](../../../api/camera.rst) is defined to wrap native camera implementation provided by simulator.

An example of depth camera implementation is shown as following:

```python
from typing import Dict

from internutopia.core.robot.robot import BaseRobot
from internutopia.core.sensor.camera import ICamera
from internutopia.core.scene.scene import IScene
from internutopia.core.sensor.sensor import BaseSensor
from internutopia.core.util import log
from internutopia_extension.configs.sensors import DepthCameraCfg


@BaseSensor.register('DepthCamera')
class DepthCamera(BaseSensor):
    """
    wrap of isaac sim's Camera class
    """
     def __init__(self, config: DepthCameraCfg, robot: BaseRobot, name: str = None, scene: IScene = None):
        super().__init__(config, robot, scene)
        self.name = name
        self._camera = self.create_camera()

    def __init__(self,
                 config: DepthCameraCfg,
                 robot: BaseRobot,
                 name: str = None,
                 scene: Scene = None):
        super().__init__(config, robot, scene)

    def post_reset(self):
        prim_path = self._robot.config.prim_path + '/' + self.config.prim_path
        if self._camera:
            self._camera.cleanup()
        self._camera = ICamera.create(
            name=self.name,
            prim_path=self.camera_prim_path,
            distance_to_image_plane=True,
            resolution=self.resolution,
        )

    def get_data(self) -> Dict:
        depth = self._camera.get_distance_to_image_plane()
        return {'depth': depth}
```
