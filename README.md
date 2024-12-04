# SDAR_torch

This repository is a simple pytorch version of the original repository in TF-2.

Please give credit to the original repository [github](https://github.com/zhxchd/SDAR_SplitNN/?tab=readme-ov-file) and Paper: [arxiv](https://arxiv.org/pdf/2310.10483) that is recently accepted by NDSS 2025.

Not implement:
- U-shape SL
- SDAR hetero client number

# how to run

## installation

just install pytorch, matplotlib, imageio
```python3
pip install -y matplotlib imageio
# pytorch installation should do according to the pytorch official.
```

## dataset preparation

Not tinyimagenet dataset need to download. Cifar10, Cifar100, STL10 datasets will be placed in the `data` dir

```bash
mkdir data

cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

```

## run

```bash
./run.sh
``` 