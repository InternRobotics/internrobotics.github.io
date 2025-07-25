# Evaluation

## Spatial Reasoning Benchmarks: MMSI-Bench, OST-Bench, EgoExo-Bench
Our evaluation framework for these benchmarks is built on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The system supports evaluation of multiple model families including: o1/o3, GPT series, Gemini series, Claude series, InternVL series, QwenVL series and LLaVA series. You need to first configure the environment variables in `.env`:
```shell
OPENAI_API_KEY= 'XXX'
GOOGLE_API_KEY = "XXX"
LMUData = "./data" # the relative/absolute path of the `data` folder.
```
Available models and their configurations can be modified in `eval_tool/config.py`. To evaluate models on MMSI-Bench/OST-Bench/EgoExo-Bench, execute the following commands:
```shell
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=1 scripts/run.py --data mmsi_bench/ost_bench/egoexo_bench --model model_name

# for very large VLMs
python scripts/run.py --data mmsi_bench/ost_bench/egoexo_bench --model model_name
```
**Note**:
- When evaluating QwenVL-7B on EgoExo-Bench, use model_name "Qwen2.5-VL-7B-Instruct-ForVideo" instead of "Qwen2.5-VL-7B-Instruct".
- We support the interleaved evaluation version of OST-Bench. For the multi-round version, please refer to [the official repository](https://github.com/OpenRobotLab/OST-Bench).

## Spatial Understanding Benchmark: MMScan

We provide two versions of the MMScan benchmark. For the original 3D version, we supply RGB videos with depth information, along with camera parameters for each frame as input. The corresponding object prompts are provided in the form of 3D bounding boxes. For the newly introduced 2D version, we provide RGB videos, with the corresponding object prompts given as the projected center of the object in each image.

(1) Original 3D Version: We mainly support LLaVA-3D as an example for this version. To run LLaVA-3D on MMScan, download the [model checkpoints](https://huggingface.co/ChaimZhu/LLaVA-3D-7B) and execute the following command:
```shell
# Single Process
bash scripts/llava3d/llava_mmscan_qa.sh --model-path path_of_ckpt --question-file ./data/annotations/mmscan_qa_val_{ratio}.json --question-file path_to_save --num-chunks 1 --chunk_idx 1

# Multiple Processes
bash scripts/llava3d/multiprocess_llava_mmscan_qa.sh
```
(2) New 2D Version: For the 2D version, execute the following command to generate the results in `.xlsx` format:
```shell
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=4 scripts/run.py --data mmscan2d --model model_name

# for very large VLMs
python scripts/run.py --data mmscan2d --model model_name
```

After obtaining results, use MMScan evaluators:
```shell
# Traditional Metrics (3D Version)
python -m scripts.eval_mmscan_qa --answer-file path_of_result

# GPT Evaluator (2D/3D Version)
python -m scripts.eval_mmscan_gpt --answer-file path_of_result --api_key XXX --tmp_path tmp_path_to_save
```
