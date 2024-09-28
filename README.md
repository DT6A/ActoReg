# The Role of Deep Learning Regularizations on Actors in Offline RL
[![arXiv](https://img.shields.io/badge/arXiv-2409.07606-b31b1b.svg)](https://arxiv.org/abs/2409.07606)

The repository organisation is inspired by [this](https://github.com/DT6A/ClORL/tree/main) repository.
## Dependencies & Docker setup
To set up a python environment (with dev-tools of your taste, in our workflow, we use conda and python 3.8), just install all the requirements:

```commandline
python3 install -r requirements.txt
```

However, in this setup, you must install mujoco210 binaries by hand. Sometimes this is not super straightforward, but this recipe can help:
```commandline
mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```
You may also need to install additional dependencies for mujoco_py. 
We recommend following the official guide from [mujoco_py](https://github.com/openai/mujoco-py).

### Docker

We also provide a more straightforward way with a dockerfile that is already set up to work. All you have to do is build and run it :)
```commandline
docker build -t actoreg .
```
To run, mount current directory:
```commandline
docker run -it \
    --gpus=all \
    --rm \
    --volume "<PATH_TO_THE_REPO>:/workspace/" \
    --name actoreg \
    actoreg bash
```

## How to reproduce experiments

### Training

Configs for reproducing results of original algorithms are stored in the `configs/<algorithm_name>/<task_type>`. All avaialable hyperparameters are listed in the `src/algorithms/<algorithm_name>.py`. Implemented algorithms are: `rebrac` and `iql` with various regularizations in actors.

For example, to start ReBRAC with the best combination of regularizations we report in our paper (LN+DO+GrN) training process with D4RL `halfcheetah-medium-v2` dataset, run the following:
```commandline
PYTHONPATH=. python3 src/algorithms/rebrac_cl.py --config_path="configs/rebrac-mt-comb/halfcheetah/medium_expert_v2.yaml"
```

### Targeted Reproduction

[//]: # (For better transparency and replication, we release all the experiments in the form of [Weights & Biases reports]&#40;https://wandb.ai/tlab/ReBRAC/reportlist&#41;.)
We provide Weights & Biases logs for all of our experiments [here](https://wandb.ai/tarasovd/ActoReg/sweeps).

If you want to replicate results from our work, you can use the configs for [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps/quickstart) provided at [https://wandb.ai/tarasovd/ActoReg/sweeps](https://wandb.ai/tarasovd/ActoReg/sweeps).

### Reliable Reports

We also provide a script and binary data for reconstructing the graphs and tables from our paper: `plotting/plotting.py`. We repacked the results into .pickle files, so you can re-use them for further research and head-to-head comparisons.

# Citing
If you use this code for your research, please consider the following bibtex:
```
@article{tarasov2024role,
  title={The Role of Deep Learning Regularizations on Actors in Offline RL},
  author={Tarasov, Denis and Surina, Anja and Gulcehre, Caglar},
  journal={arXiv preprint arXiv:2409.07606},
  year={2024}
}
```