# Hierarchical Fine-Grained Image Recognition

A PyTorch implementation of hierarchical fine-grained image classification
with a multi-branch ResNet-50. The model predicts every taxonomic level for an
image and combines those predictions into a hierarchy-aware species score.

This project originated as an EE3108 Design & Innovation Project and uses the
datasets introduced with *Fine-Grained Representation Learning and Recognition
by Exploiting Hierarchical Semantic Embedding* (Chen et al., ACM MM 2018).

> **Scope:** the network in this repository follows the paper's shared-trunk,
> multi-branch baseline architecture. It does not currently implement the full
> HSE semantic-guidance modules, KL regularization, or staged training procedure.

## Architecture

An ImageNet-pretrained ResNet-50 is split after `layer3`. Four independent
copies of `layer4` predict increasingly fine labels:

```text
image -> shared ResNet-50 trunk
                  |-> branch 1 -> order/family
                  |-> branch 2 -> family/subfamily
                  |-> branch 3 -> genus
                  `-> branch 4 -> species
```

The first three probability distributions are projected into the species space
using dataset-provided transition matrices. Their projected scores are added to
the species branch probability distribution to produce the fused prediction.

The default configuration preserves the behavior of the original project: the
shared feature tensor is detached before entering the first three branches, so
only the species loss updates the shared trunk. This is configurable with
`--no-detach-auxiliary-branches`.

## Supported datasets

- **Extended CUB-200-2011:** 13 orders, 37 families, 122 genera, and 200 species.
- **Butterfly-200:** 5 families, 23 subfamilies, 116 genera, and 200 species.

The image data and hierarchical annotations are not redistributed in this
repository. Obtain them from the
[official HSE project](https://sysu-hcp.net/projects/cv/35.html) and respect the
dataset's terms of use.

## Installation

Python 3.10 or newer is required. Install PyTorch using the command appropriate
for your operating system and CUDA version, then install this project:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Data layout

The default paths are:

```text
data/
├── CUB_200_2011/CUB_200_2011/
│   ├── images/
│   └── CUB_200_2011_train_test_multi_level_info.pkl
└── Butterfly200/Butterfly200/
    ├── images/
    └── Butterfly_train_test_multi_level_info.pkl
```

The metadata file is the sequential 13-object pickle stream used by the
original project. It contains train/test filenames and labels, four auxiliary
objects, class counts, and four adjacent-level transition matrices. The loader
validates the lengths and matrix dimensions before training.

Python pickle files can execute arbitrary code. Only load metadata from a
trusted source.

## Training

Each invocation trains one dataset and writes a timestamped experiment:

```bash
hse-fgir-train --dataset cub --data-root data
hse-fgir-train --dataset butterfly --data-root data
```

Important settings are exposed as command-line options:

```bash
hse-fgir-train --help
hse-fgir-train \
  --dataset cub \
  --epochs 20 \
  --batch-size 32 \
  --image-size 448 \
  --learning-rate 0.005 \
  --device cuda \
  --device-ids 0 1
```

Training produces:

- `last-model.pth`, including model, optimizer, scheduler, and configuration;
- `metrics.jsonl`, with one machine-readable record per epoch;
- a copy of the exact hierarchy metadata used by the experiment.

The legacy metadata provides train/test splits but no validation split. Test
accuracy is logged for compatibility with the original experiment, but it is
not used to select a “best” checkpoint.

Resume an interrupted run with `--resume PATH_TO_LAST_MODEL`.

## Inference

Predict individual images or every supported image in a directory:

```bash
hse-fgir-predict test/ \
  --dataset cub \
  --checkpoint experiments/cub/EXPERIMENT_ID/last-model.pth
```

The command emits one JSON object per image containing predictions at all four
levels and the hierarchy-fused species prediction. Predictions are zero-based
class indices because the legacy metadata schema does not document a reliable
index-to-name field.

A small, output-free example is also available in
`load_model_predict.ipynb`.

## Development

The project uses a `src` package layout and keeps training, inference, data
loading, hierarchy operations, and model definitions independent:

```text
src/hse_fgir/
├── config.py
├── data.py
├── engine.py
├── hierarchy.py
├── model.py
├── predict.py
└── train.py
```

Run the static checks and unit tests with:

```bash
ruff check .
python -m unittest discover -s tests
```

The test suite uses synthetic hierarchy data and does not require either image
dataset.

## Reference

Tianshui Chen, Wenxi Wu, Yuefang Gao, Le Dong, Xiaonan Luo, and Liang Lin.
“Fine-Grained Representation Learning and Recognition by Exploiting
Hierarchical Semantic Embedding.” ACM Multimedia, 2018.
[Paper](https://arxiv.org/abs/1808.04505) ·
[Official implementation](https://github.com/HCPLab-SYSU/HSE)
