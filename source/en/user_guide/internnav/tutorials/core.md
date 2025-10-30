# Core Concepts
## Overview



The main architecture of the evaluation code adopts a client-server model. In the client, we specify the corresponding configuration (*.cfg), which includes settings such as the scenarios to be evaluated, robots, models, and parallelization parameters. The client sends requests to the server, which then submits tasks to the Ray distributed framework based on the corresponding cfg file, enabling the entire evaluation process to run.
## Main Process (WIP)

![img.png](../../../_static/image/internnav_process.png)

**Learn the Modules**
1. [Dataset](./dataset.md)
2. [Model](./model.md)
3. [Training](./training.md)
4. [Agent](./agent.md)
5. [Env](./env.md)
