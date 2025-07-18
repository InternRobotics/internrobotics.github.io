# Benchmarking
This section provides a quick start guide for evaluating models in the GRManipulation framework. The evaluation process is designed to be straightforward, allowing users to quickly assess the performance of their models on various tasks.

## Evaluation in a single process
By default, the inference of model will be running in the main loop sharing the same process with the `env`. You can evaluate `pi0` on the `Genmanip` benchmark in a single process using the following command:

```bash
python3 scripts/eval/eval.py \
   --dataset-path genmanip \ # use the name of the registered dataset, e.g
   --model-path lerobot/pi0 \ # model checkpoint
```


## Evaluation with a Client-Server Architecture
 However, the hardware (e.g., RTX series v.s. A100) or package requirements for model and benchmark simulator can potentially be conflicting. To solve this, we use a **client-server** architecture to separate the model inference from the simulation. The **server** will run the model inference and the **client** will run the simulator. The communication between them is done through a socket connection.

Specifically, you should first launch the model server with the following command:

```bash
python3 scripts/eval/start_model_server.py \
   --port 12345 \ # port for the server to listen on
   --policy lerobot/pi0 \ # model
   --model-path lerobot/pi0 \ # model checkpoint
   --data-config sweep_joint \ # registered data config
```

Then you should run the benchmark client with the following command:

```bash
python3 scripts/eval/start_evaluator_client.py \
    --server-ip <server_ip> \ # IP address of the server
    --port 12345 \ # port for the server to connect to
    --benchmark genmanip \ # use the name of the registered benchmark
    --output-dir evaluation_results/runs/${run_name} \ # output directory for evaluation results
```

We provide an example bash script `bash_scripts/eval_genmanip.sh` for running the evaluation on the `Genmanip` benchmark. You can run it with the following command:

```bash
bash bash_scripts/eval_genmanip.sh
```

You can modify the bash script according to your resource availability and requirements.


## Available Benchmarks
The following benchmarks are currently available for evaluation:
- **Genmanip**: A benchmark for general manipulation tasks.
- **CALVIN**: A benchmark for complex manipulation tasks.
- **Simpler-Env**: A benchmark for simpler environments.

We have the finetuned weights for the following built-in models that can be used for evaluation:
- `pi0`: A pre-trained model for general-purpose manipulation tasks.
- `gr00t-n1/1.5`: An advanced model with additional capabilities.
- `DP+CLIP`: A model that combines diffusion policy with CLIP for instruction-guided manipulation.
- `AcT+CLIP`: A model that integrates Action-chunking Transformer with CLIP for instruction-guided manipulation.

To evaluate models on your own benchmark, you should first implement your benchmark and register it, then follow the same command structure as above, replacing the `--benchmark` with your custom benchmark name and adjusting other parameters as needed. Please refer to [`How to add your benchmark`](../tutorials/evaluation.md) to learn how to prepare your model and benchmark for evaluation.
