# Model Finetuning
## Finetune a built-in model

To finetune a built-in model, you can use the `train.py` script provided in the `scripts/train` directory. This script allows you to specify various parameters such as dataset path, number of GPUs, batch size, output directory, and more. Following is an example command to finetune `Pi-0` model on the `Genmanip` dataset:

```bash
python3 scripts/train/train.py \
   --dataset-path genmanip \ # use the name of the registered dataset, e.g., Genmanip, or a custom dataset path
   --num-gpus 1 \
   --batch_size 16 \
   --output-dir ./Checkpoints/runs/${run_name}  \
   --max-steps 20000 \
   --tune_visual \
   --tune_llm \
   --save_steps 80000 \
   --data-config sweep_joint \ # registered data config
   --video-backend torchvision_av \
   --report_to tensorboard \
   --base_model_path lerobot/pi0 # model checkpoint
```
We also provide bash scripts under `bash_scripts/` for finetuning models on slurm clusters. For instance, to finetune `Pi-0` model on the `Genmanip` dataset using 8 GPUs, you can use the following command:

```bash
srun --gpus-per-task=8 --cpus-per-task=8 --ntasks=1 --job-name=pi0_genmanip \
   bash bash_scripts/train_pi0_genmanip.sh
```

For a quick validation of the finetuned model, you can perform an open-loop evaluation on the training set using the `eval_open_loop.py` script:

```bash
python3 scripts/eval/eval_open_loop.py \
   --dataset-path genmanip \ # use the name of the registered dataset, e.g, Genmanip, or a custom dataset path
   --model-path ./Checkpoints/runs/${run_name} \ # path to the finetuned model
   --video-backend torchvision_av \ # video backend for rendering
   --data-config sweep_joint \ # registered data config
```

## Available Models and Datasets

At present, we have the following built-in models and datasets that can be used for finetuning:
- **Models**:
  - `pi0`: A pre-trained model for general-purpose manipulation tasks.
  - `gr00t-n1/1.5`: An advanced model with additional capabilities.
  - `DP+CLIP`: A model that combines diffusion policy with CLIP for instruction-guided manipulation.
  - `AcT+CLIP`: A model that integrates Action-chunking Transformer with CLIP for instruction-guided manipulation.
- **Datasets**:
  - `Genmanip`: A dataset for general manipulation tasks.
  - `CALVIN`: A dataset for complex manipulation tasks.
  - `Simpler-Env`: A dataset for simpler environments.

To finetune your own model, you can follow the same command structure as above, replacing the `--base_model_path` with your custom model path and adjusting other parameters as needed. Please refer to [`How to customize your dataset`](../tutorials/dataset.md) and [`How to add a model`](../tutorials/model.md) to learn how to prepare your model and dataset for finetuning.
