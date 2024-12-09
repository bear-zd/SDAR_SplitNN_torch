import os
import numpy as np
import time
import argparse
import sys
import torch
from data import *
from sdar import SDARAttacker
from config import load_config
import random


def init():
    parser = argparse.ArgumentParser(
        description="Run SDAR experiment with specified configurations."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"]
    )
    parser.add_argument("--level", type=int, choices=[4, 5, 6, 7, 8, 9])

    parser.add_argument("--aux_data_frac", type=float, default=1.0)
    parser.add_argument("--num_class_to_remove", type=int, default=0)
    parser.add_argument("--diff_simulator", action="store_true")
    parser.add_argument("--run", type=int, help="Run number.", default=0)
    parser.add_argument("--model", type=str, help="Model structure.", default="resnet20", choices=["resnet20", "plainnet"])

    # number of heterogenous distributions of the client data
    parser.add_argument("--num_hetero_client", type=int, default=1)

    # the following arguments are used to specify the defense method
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)

    parser.add_argument("--print_to_stdout", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    dir_name = os.path.join("logs", f"{args.dataset}_{args.model}_l{args.level}_run{args.run}")
    args.dir_name = dir_name
    os.makedirs(dir_name, exist_ok=True)
    if not args.print_to_stdout:
        sys.stdout = open(os.path.join(dir_name, "logs.txt"), "wt")

    def random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random_seed(args.run)
    return args


def main(args):
    dataset = LoadDataset(args.dataset)
    # num_class = dataset.num_class
    num_iters = 40000 if args.level > 7 else 20000
    if args.test:
        num_iters //= 100
    config = load_config(args.dataset, args.model)

    client_ds, server_ds = dataset.get_dataset(
        args.aux_data_frac, args.num_class_to_remove, evaluate=False
    )
    client_eval_ds, _ = dataset.get_dataset(
        args.aux_data_frac, args.num_class_to_remove, evaluate=True
    )

    print(f"Start experiments on {args.dataset} at l{args.level} with {config} and model {args.model}")
    sdar_attacker = SDARAttacker(
        client_loader=client_ds,
        server_loader=server_ds,
        num_classes=dataset.num_class,
        device=f"cuda:{args.gpu}",
        dataset_name=args.dataset,
    )
    sdar_attacker.preprocess(level=args.level, num_iters=num_iters, p_config=config)
    start_time = time.time()
    history = sdar_attacker.train_pipeline()
    end_time = time.time()
    print(
        f"Training time: {end_time - start_time}. Periter time: {(end_time - start_time) / num_iters}"
    )

    # evaluate
    sdar_attacker.evaluate(client_eval_ds)

    # try attack
    x, y = list(client_eval_ds)[0]
    x_recon, mse = sdar_attacker.attack(x, y)
    t = 6
    plot_attack_results(
        x[t : t + 20],
        x_recon[t : t + 20],
        os.path.join(args.dir_name, "attack_results.png"),
    )
    print(f"Attack MSE on batch: {mse}")

    np.save(os.path.join(args.dir_name, "history.npy"), history)
    print(f"History saved to {os.path.join(args.dir_name, 'history.npy')}")


if __name__ == "__main__":
    args = init()
    main(args)
