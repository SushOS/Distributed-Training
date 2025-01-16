import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader, Dataset

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

def normal_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel(100).to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = MyDataset(1000, 100)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    start = time.time()
    for _ in range(10):
        train_loop(dataloader, model, device, optimizer, criterion)
    print(f"Normal training time: {time.time()-start:.2f} seconds")

def DDP_training(rank, world_size):
    # nccl -> nvidia collective communication library used by the replicas to communicate 
    # with each other in a multiGPU scenario
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    dataset = MyDataset(length = 1000, size = 100)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = MyModel(input_size=100).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(5):  # Train for 5 epochs
        train_loop(model, dataloader, criterion, optimizer, device)
    end_time = time.time()
    
    if rank == 0:
        print(f"DDP training time: {end_time - start_time:.2f} seconds")
    
    destroy_process_group()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="PyTorch DDP vs Normal Training Example")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for DDP (e.g., nccl, gloo)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--input_size", type=int, default=100, help="Input feature size of the model")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Load environment variables for DDP
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    
    # Check GPU availability
    if args.world_size < 2:
        print("This script requires at least 2 GPUs!")
    else:
        print(f"Running normal training with {args.epochs} epochs...")
        normal_training(args.input_size, args.batch_size, args.epochs)
        
        print(f"\nRunning DDP training with {args.epochs} epochs and backend '{args.backend}'...")
        torch.multiprocessing.spawn(
            DDP_training,
            args=(args.world_size, args.backend, args.input_size, args.batch_size, args.epochs),
            nprocs=args.world_size
        )

"""
To run the script, you can use the following command:
python ddp_training.py --backend nccl --epochs 10 --batch_size 128 --input_size 200 --world_size 2
"""