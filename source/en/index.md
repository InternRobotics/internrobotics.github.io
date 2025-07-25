---
sidebar: false
myst:
  html_meta:
    "description lang=en": |
      Documentation for users who wish to build sphinx sites with
      pydata-sphinx-theme.
---

# Welcome to Intern Robotics Documentation
**Intern Robotics** provides a framework for facilitating robotics research and development, providing a set of tools and libraries that features:

 - **Model Training**: Train embodied models for various robotics tasks.
 - **Evaluation**: Evaluate the performance of trained models, reproducing results across different environments.
 - **Benchmarking**: Benchmark models against standard datasets and tasks.
 - **Data Generation**: Collect training data inside various simulation environments.
 - **Development**: Develop various robotics systems, tasks and applications.

The framework comprises a general-purpose simulation platform [InternUtopia](user_guide/internutopia/index.md) that provides realistic and immersive environments, along with toolkits that support research across various embodied AI domains:
- [InternSR](user_guide/internsr/index.md): Embodied spatial reasoning.
- [InternNav](user_guide/internnav/index.md): Interactive multimodal embodied navigation.
- [InternManip](user_guide/internmanip/index.md): Multimodal manipulation.
- [InternHumanoid](#) (WIP): Agile humanoid robot control.
- [InternWorldModel](#) (WIP): World modeling and dynamic prediction.



```{toctree}
:maxdepth: 1

user_guide/index
api/index

```
