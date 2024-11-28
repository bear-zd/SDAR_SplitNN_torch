import os
import numpy as np
import time
import argparse
import sys
from data import *
from sdar import SDARAttacker
from config import load_config

parser = argparse.ArgumentParser(description='Run SDAR experiment with specified configurations.')
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
parser.add_argument("--level", type=int, choices=[4,5,6,7,8,9])

parser.add_argument("--aux_data_frac", type=float, default=1.0)
parser.add_argument("--num_class_to_remove", type=int, default=0)
parser.add_argument("--diff_simulator", action="store_true")

# number of heterogenous distributions of the client data
parser.add_argument("--num_hetero_client", type=int, default=1)

# the following arguments are used to specify the defense method
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.0)

parser.add_argument("--print_to_stdout", action="store_true")
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

dataset = LoadDataset(args.dataset)
num_class = dataset.num_class
num_iters = 40000 if args.level > 7 else 20000
config = load_config(args.dataset)

client_ds, server_ds = dataset.get_dataset(args.aux_data_frac, args.num_class_to_remove)

print(f"Start experiments on {args.dataset} at l{args.level} with {config}")

sdar_attacker = SDARAttacker(client_loader=client_ds, server_loader=server_ds, num_classes=dataset.num_class, device=f"cuda:{args.gpu}")

sdar_attacker.preprocess(level=args.level, num_iters=num_iters, p_config=config)

start_time = time.time()
history = sdar_attacker.train_pipeline()
end_time = time.time()
