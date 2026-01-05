# Training

This document provides instructions for training models in **InternNav**.  

## Overview

InternNav supports training models under three system paradigms:

- **Dual-System VLN Models**: integrated System2 + System1 architectures  
- **Single-System VLN Models**: end-to-end vision-and-language navigation models  
- **VN System (System1) Models**: low-level visual navigation and control models  


Each paradigm follows a different training protocol, which is detailed below.


## Dual-System VLN Models
Dual-System VLN Models integrates **System2** (high-level reasoning and planning) with  
**System1** (low-level action control), supporting both modular integration and joint training.


### Supported Systems
- **InternVLA-N1 (System2)**  
- **InternVLA-N1 (Dual System) w/ NavDP***
  (*NavDP** indicates joint tuning with System2)
- **InternVLA-N1 (Dual System) DualVLN**


### 1. Training for InternVLA-N1 (System2)

**InternVLA-N1 (System2)** is trained independently to predict 2D pixel goals for navigation.

It can be used with any compatible System1 model capable of executing 2D pixel goals or point goals (given depth and pose).  
Alternatively, it can be jointly trained together with a System1 model for end-to-end multi-system optimization.


#### Training Command

```bash
# training system2 separately
sbatch ./scripts/train/base_train/qwenvl_train/train_system2.sh 
```

---

### 2. Joint Training for InternVLA-N1 (Dual System)

After completing training of **InternVLA-N1 (System2)**, joint training is supported with a pixel-goal navigation System1, using either the **NavDP** or **NextDiT** architecture.

- **InternVLA-N1 (Dual System) w/ NavDP**: preserves **NavDP**'s model design and uses **RGB-D** input.  
- **InternVLA-N1 (Dual System) DualVLN**: uses only **RGB** input, resulting in a smaller model footprint.

#### Training Command

```bash
# training system1 based on system2
sbatch ./scripts/train/base_train/qwenvl_train/train_dual_system.sh 
```

- For **w/ NavDP** model variant, set `system1=navdp_async`. Optimal performance is typically observed after **30,000 iterations**.  
- For **DualVLN** model variant, set `system1=nextdit_async`. Optimal performance is typically observed after **15,000 iterations**.

## Single-System VLN Models

Single-System VLN Models directly map **visual observations and language instructions** to navigation actions in an end-to-end manner.


### Supported Models

The following Single-System VLN Models are currently supported:

- Seq2Seq  
- CMA  
- RDP  

For our VLM-based VLN model **StreamVLN**, please refer to the following repository for training details:  
https://github.com/InternRobotics/StreamVLN  

Support for StreamVLN within InternNav is planned for future releases.


### Training Command

Training is performed through a unified training entry script.  
Below are example commands for each supported model.

**Seq2Seq**
```
./scripts/train/base_train/start_train.sh --name seq2seq_train --model seq2seq
```

**CMA**
```
./scripts/train/base_train/start_train.sh --name cma_train --model cma
```

**RDP**
```
./scripts/train/base_train/start_train.sh --name rdp_train --model rdp
```


## VN System (System1) Models

VN System (System1) focuses on **low-level visual navigation and motion control**.  


### Supported Methods

The following visual navigation methods are included in the System1 benchmark:

- DD-PPO  
- iPlanner  
- ViPlanner  
- GNM  
- ViNT  
- NoMaD  
- NavDP (**InternVLA-N1 System1**)

Among them, **only NavDP is currently supported for training** in InternNav.  
All other methods are provided for **evaluation and comparison purposes only**.


### Training Command

**NavDP**


```bash
./scripts/train/base_train/start_train.sh --name navdp_train --model-name navdp
```

