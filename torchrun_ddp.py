import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

class MyDataset(Dataset):
    def __init__(self, length, size):
        self.data = torch.randn(length, size)
        self.labels = torch.randint(0, 2, (length,))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_size, 2)
    
    def forward(self, x):
        return self.fc(x)

def train_loop(dataloader, model, device, optimizer, criterion):
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), label)
        loss.backward()
        optimizer.step()

def DDP_training(args):
    # Initialize the process group
    init_process_group(backend=args.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Prepare dataset and dataloader
    dataset = MyDataset(length=1000, size=args.input_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Define the model, wrap it in DDP, and set up optimizer and loss
    model = MyModel(input_size=args.input_size).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loop(dataloader, model, device, optimizer, criterion)
    end_time = time.time()

    if rank == 0:
        print(f"DDP training time: {end_time - start_time:.2f} seconds")

    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Example with torchrun")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for DDP (e.g., nccl, gloo)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--input_size", type=int, default=100, help="Input feature size of the model")
    args = parser.parse_args()

    DDP_training(args)


# Script to run the code
"""
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 torchrun_ddp.py --backend nccl --epochs 5 --batch_size 64 --input_size 100
"""