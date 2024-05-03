# SonicTriad_SaShiMi

# [It's Raw! Audio Generation with State-Space Models](https://arxiv.org/abs/2202.09729)

Karan Goel, Albert Gu, Chris Donahue, Christopher Ré

[*ICML 2022*](https://icml.cc/virtual/2022/poster/17773)

This repository provides a simple implementation of this paper in PyTorch


## Installation

PyTorch is required. See [PyTorch installation page](https://pytorch.org/get-started/locally/) for more info.
Here's how to install PyTorch with `pip`:
```bash
pip3 install torch torchvision torchaudio
```

Following libraries are required for dataset handling:
```bash
pip3 install numpy scipy
```

**Optional:** [PyKeOps](https://www.kernel-operations.io/keops/index.html) can be installed for more memory-efficient Cauchy kernel computation.
Install PyKeOps using `pip`:
```bash
pip3 install pykeops
```

If that doesn't work, try:
```bash
pip3 install pykeops[full]
```


## Downloading the Data

Download the pre-trained models together with the YouTube Mix dataset, which are not included in this repository due to their large file sizes.
To download these, you can simply run `download_data.sh`.
`wget` and `unzip` programs must be installed.

```bash
./download_data.sh
```

## Training Script

Since the training on the full dataset takes a lot of time, we have created a dedicated training script for that instead of putting it into the Jupyter Notebooks.

The script named `train.py` will train the SaShiMi model on the Youtube Mix dataset located at `./datasets/youtube-mix/train/` and save the model checkpoints at `./models/ym-8l/` after each epoch:
```bash
python3 train.py
```

For example, you can interrupt the training at any time and change the learning rate by modifying the following part:
```python
# Update LR
for group in optimizer.param_groups:
    group["lr"] = 0.001
```

## Testing Script

Since the training on the full dataset takes a lot of time, we have created a dedicated training script for that instead of putting it into the Jupyter Notebooks.

The script named `test.py` will generate audio sample from SaShiMi model from the model checkpoints at `./models/ym-8l/` with best epoch of mention duration:
```bash
python3 test.py duration
```

For example, you can interrupt the training at any time and change the learning rate by modifying the following part:
```python
# Update LR
for group in optimizer.param_groups:
    group["lr"] = 0.001
```

Audio is saved