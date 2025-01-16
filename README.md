# PyTorch Distributed Data Parallel (DDP) Training Examples

This repository provides examples of implementing Distributed Data Parallel (DDP) training in PyTorch using two different approaches: using `torchrun` and using manual process spawning. These examples demonstrate how to scale deep learning training across multiple GPUs efficiently.

## Features

- Simple linear model implementation for demonstration purposes
- Support for both `torchrun` and manual multi-processing approaches
- Configurable parameters via command-line arguments
- Comparison between normal training and DDP training
- Custom dataset implementation for testing

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA-capable GPU(s)
- python-dotenv (for `ddp.py`)

## Installation

```bash
# Clone the repository
git clone git@github.com:SushOS/Distributed-Training.git
cd Distributed-Training

# Install requirements
pip install torch python-dotenv
```

## Files Description

1. `torchrun_ddp.py`: Implementation using `torchrun` for process management
   - Simpler setup using PyTorch's built-in distributed training launcher
   - Recommended approach for most use cases

2. `ddp.py`: Implementation using manual process spawning
   - More control over the training process
   - Uses environment variables for configuration
   - Includes comparison with normal (non-distributed) training

## Usage

### Using torchrun (Recommended)

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 torchrun_ddp.py --backend nccl --epochs 5 --batch_size 64 --input_size 100
```

Parameters:
- `--nproc_per_node`: Number of processes (GPUs) to use per node
- `--nnodes`: Total number of nodes
- `--node_rank`: Rank of the current node
- `--master_addr`: Address of the master node
- `--master_port`: Port for communication

### Using Manual Process Spawning

```bash
python ddp.py --backend nccl --epochs 10 --batch_size 128 --input_size 200 --world_size 2
```

Parameters:
- `--backend`: Backend for DDP (nccl or gloo)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--input_size`: Input feature size of the model
- `--world_size`: Number of GPUs to use

## Environment Variables

For `ddp.py`, create a `.env` file with the following variables:
```env
MASTER_ADDR=localhost
MASTER_PORT=12355
```

## Model Architecture

The example uses a simple linear model (`MyModel`) with:
- Input size: Configurable via command line
- Output size: 2 (binary classification)
- Single fully connected layer

## Dataset

The example includes a custom `MyDataset` class that generates:
- Random input tensors
- Random binary labels
- Configurable dataset length and feature size

## Notes

- The NCCL backend is recommended for multi-GPU training on a single machine
- The Gloo backend can be used for CPU training or when NCCL is not available
- Make sure all GPUs are on the same CUDA device (same machine) when using NCCL
- The examples use `DistributedSampler` to ensure proper data distribution across processes
