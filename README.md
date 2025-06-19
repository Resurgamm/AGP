# Adaptive Graph Pruning for Multi-Agent Communication

## Overview 📒

We provided the code of the paper. The algorithm implementation code is located in the `AGP` folder, and the experimental code is located in the `experiments` folder.


## Quick Start 🚀

### Download the Codes

```bash
git clone
cd AGP
```

### Install packages

```bash
conda create -n gdesigner python=3.10
conda activate AGP
pip install -r requirements.txt
```

### Call API

Add API keys in `AGP/llm/gpt_chat.py`.

```python
MINE_BASE_URL = "" # the BASE_URL of OpenAI LLM backend
MINE_API_KEYS = "" # for OpenAI LLM backend
```

### Stage I

We provide all the datasets (`train_general_reasoning.json`, `train_math_reasoning.json`, `train_coding.json`) from Stage I. You can skip this part.

Or, you can run Stage I on general reasoning by running the following scripts:

```bash
python experiments/general_reasoning_collector.py --mode FullConnected --batch_size 10 --agent_nums 9 --num_iterations 10 --num_rounds 1 --optimized_spatial --resume True
python experiments/get_general_reasoning_dataset.py --mode FullConnected --batch_size 10 --agent_nums 9 --num_iterations 10 --num_rounds 1 --optimized_spatial --resume True
```

And you will get `train_general_reasoning`.

The same applies to running Stage I in the other two fields.

## Stage II

We provide models that have been trained separately in three domains `*.pth`, and you can directly use them for evaluation by modifying the name of any model parameter to `model.pth`.

Or you can run Stage II on general reasoning by running the following scripts:

```bash
python experiments/train_general_reasoning.py --mode FullConnected --batch_size 10 --agent_nums 9 --num_iterations 10 --num_rounds 1 --optimized_spatial --resume True
```

to get the model parameter.

The same applies to running Stage II in the other two fields.
