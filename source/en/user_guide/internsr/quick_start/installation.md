<style>
  .mytable {
    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 8px;        /* 四个角统一 8px 圆角 */
    border-collapse: separate; /* 必须设 separate，否则圆角会被 collapse 吃掉 */
    border-spacing: 0;
  }
  .mytable th, .mytable td {
    border: 1px solid rgba(128, 128, 128, 0.5);
    padding: 6px 12px;
  }
</style>

# 🛠️ Installation Guide

😄 Don’t worry — both [Quick Installation](#quick-installation) and [Dataset Preparation](#dataset-preparation) are beginner-friendly.


<!-- > 💡NOTE \
> 🙋 **[First-time users:](#-lightweight-installation-recommended-for-beginners)** Skip GenManip for now — it requires installing NVIDIA [⚙️ Isaac Sim](#), which can be complex.
Start with **CALVIN** or **SimplerEnv** for easy setup and full training/eval support.\
> 🧠 **[Advanced users:](#-full-installation-advanced-users)** Feel free to use all benchmarks, including **GenManip** with Isaac Sim support. -->

<!-- > For 🙋**first-time** users, we recommend skipping the GenManip benchmark, as it requires installing NVIDIA [⚙️ Isaac Sim](#) for simulation (which can be complex).
Instead, start with **CALVIN** or **SimplerEnv** — both are easy to set up and fully support training and evaluation. -->

<!-- This guide provides comprehensive instructions for installing and setting up the InternManip robot manipulation learning suite. Please read through the following prerequisites carefully before proceeding with the installation. -->


## Quick Installation

```shell
git clone https://github.com/InternRobotics/InternSR.git
cd InternSR
pip install -e .
```

## Dataset Preparation

We recommend placing all data under `data/`. The expected directory structure under `data/` is as follows :

```shell
data/
├── images/ # `images/` folder stores all image modality files from the datasets
├── videos/ # `videos/` folder contains all video modality files from the datasets
├── annotations/ # `annotations/` folder holds all text annotation files from the datasets
```

### MMScan
1. Download the image zip files from [Hugging Face](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main) (~56G), combine and unzip them under `./data/images/mmscan`.
2. Download the annotations from [Hugging Face](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main) and place them under `./data/annotations`.

```shell
data/
├── images/
│   ├── mmscan/
│   │   ├── 3rscan
│   │   ├── 3rscan_depth
│   │   ├── matterport3d
│   │   ├── scannet
├── annotations/
│   ├── embodiedscan_video_meta/
│   ├── ├── image.json
│   ├── ├── depth.json
│   ├── ├── ...
│   ├── mmscan_qa_val_0.1.json
│   ├── ...
```
**Note**: The file `mmscan_qa_val_{ratio}.json` contains the validation data at the specified ratio.


### OST-Bench
Download the images from [Hugging Face](https://huggingface.co/datasets/rbler/OST-Bench)/[Kaggle](https://www.kaggle.com/datasets/jinglilin/ostbench/)(~5G) and download the [`.tsv` file](https://opencompass.openxlab.space/utils/VLMEval/OST.tsv) , place them as follows:

```shell
data/
├── images/
│   ├── OST/
│   │   ├── <scan_id>
│   │   ├── ...
├── annotations/
│   ├── OST.tsv
```

### MMSI-Bench
Download the [`.tsv` file](https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_bench.tsv) (~1G, including images) , place it as follows:

```shell
data/
├── annotations/
│   ├── MMSI_Bench.tsv
```
### EgoExo-Bench
1. Download the processed video data from the [Hugging Face](https://huggingface.co/datasets/onlyfaces/EgoExoBench/tree/main).
2. Due to license restrictions, data from the [Ego-Exo4D](https://ego-exo4d-data.org/) project is not included. Users should acquire it separately by following the official Ego-Exo4D guidelines.
3. Download the [`.tsv` file](https://drive.google.com/file/d/1pRGd9hUgwCzMU6JSPFxpjGAtCChwIB9G/view?usp=sharing) , place them as follows:

```shell
data/
├── videos/
│   ├── EgoExo4D/tasks
│   ├── processed_frames
│   ├── processed_video
├── annotations/ EgoExoBench_MCQ.tsv
```
