# Env
## Create Your Env and Task in InternNav
This tutorial provided a step-by-step guide to define a new environment and a new navigation task within the InternNav framework.

---

## Overview
InternNav separates **navigation logic / policy** from **where the agent actually lives** (simulator vs real robot). The key ideas are:

- `Env`: a unified interface. All environments must behave like an `Env`.

- `Task`: a high-level navigation objective exposed to the agent, like "go to the kitchen sink" or "follow this instruction".

- `Agent`: consumes observations from `Env`, predicts an action, and sends that action back to `Env`.

Because of this separation:

- We can run the same agent in simulation (Isaac / InternUtopia) or on a real robot, as long as both environments implement the same API.

- We can benchmark different tasks (VLN, PointGoalNav, etc.) in different worlds without rewriting the agent.

InternNav already ships with two major environment backends:

- **InternUtopiaEnv**:
Simulated environment built on top of InternUtopia / Isaac Sim. This supports complex indoor scenes, object semantics, RGB-D sensing, and scripted evaluation loops.
- **HabitatEnv** (WIP): Simulated environment built on top of Habitat Sim.

- **RealWorldEnv**:
Wrapper around an actual robot platform and its sensors (e.g. RGB camera, depth, odometry). This lets you deploy the same agent logic in the physical world.

Both of these are children of the same base [`Env`](https://github.com/InternRobotics/InternNav/blob/main/internnav/env/base.py) class.

## Evaluation Task (WIP)
For the vlnpe benchmark, we build the task based on internutopia. Here is a diagram.

![img.png](../../../_static/image/agent_definition.png)


## Evaluation Metrics (WIP)
For the vlnpe benchmark in internutopia, InternNav provides comprehensive evaluation metrics:
- **Success Rate (SR)**: Proportion of episodes where the agent reaches the goal location within 3m
- **SPL**: Success weighted by Path Length
- **Trajectory Length (TL)**: Total length of the trajectory (m)
- **Navigation Error (NE)**: Euclidean distance between the agent's final position and the goal (m)
- **OS Oracle Success Rate (OSR)**: Whether any point along the predicted trajectory reaches the goal within 3m
- **Fall Rate (FR)**: Frequency of the agent falling during navigation
- **Stuck Rate (StR)**: Frequency of the agent becoming stuck during navigation

The implementation is under `internnav/env/utils/internutopia_extensions`, we highly suggested follow the guide of [Internutopia](../../internutopia).
