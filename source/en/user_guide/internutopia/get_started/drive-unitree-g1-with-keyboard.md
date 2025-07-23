# 🚶‍➡️Drive Unitree G1 with Keyboard

> This tutorial guides you to drive Unitree G1 Robot with keyboard.

```bash
$ python -m internutopia.demo.g1_locomotion
```

You can control the g1 robot with keyboard command:

- I: Move Forward
- K: Move Backward
- J: Move Left
- L: Move Right
- U: Turn Left
- O: Turn Right

<video width="720" height="405" controls>
    <source src="../../../_static/video/g1_locomotion.webm" type="video/webm">
</video>

## Brief Explanation

The keyboard is abstracted as an interaction device. A vector is used to denote which key is being pressed, and this vector is then translated into the robot's actions at each step.

```python
from internutopia_extension.interactions.keyboard import KeyboardInteraction

keyboard = KeyboardInteraction()

while env.simulation_app.is_running():
    i += 1
    command = keyboard.get_input()
    x_speed = command[0] - command[1]
    y_speed = command[2] - command[3]
    z_speed = command[4] - command[5]
    env_action = {
        move_by_speed_cfg.name: (x_speed, y_speed, z_speed),
    }
    obs, _, terminated, _, _ = env.step(action=env_action)
    ...
```

You can refer to [`InternUtopia/internutopia/demo/g1_locomotion.py`](https://github.com/InternRobotics/InternUtopia/blob/main/internutopia/demo/g1_locomotion.py) for a complete example.
