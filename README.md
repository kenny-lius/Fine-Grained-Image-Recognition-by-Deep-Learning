# Fine-Grained Image Recognition by Deep Learning

EE3108 - Design & Innovation Project

Hierarchical fine-grained image classification with a multi-branch ResNet-50:
a shared trunk (conv1-3) feeds four branches, one per label granularity
(coarse to fine), and their predictions are combined at inference time using
the dataset's coarse-to-fine label mapping. Supports [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
(birds) and Butterfly200 (butterflies).

## Layout

- `common.py` — shared dataset, transform, and model code used by both
  training and inference.
- `train.py` — trains a model on CUB-200-2011 or Butterfly200.
- `load_model_predict.ipynb` — loads a trained checkpoint and runs inference
  on images in `./test`.

## Setup

```bash
pip install -r requirements.txt
```

Each dataset needs its images plus a pickled train/test split file (file
lists, hierarchical labels, and coarse-to-fine label transition matrices),
laid out under `./data/` as referenced in `common.py`'s `DATASET_CONFIGS`:

- `./data/CUB_200_2011/CUB_200_2011/images/` and
  `./data/CUB_200_2011/CUB_200_2011/CUB_200_2011_train_test_multi_level_info.pkl`
- `./data/Butterfly200/Butterfly200/images/` and
  `./data/Butterfly200/Butterfly200/Butterfly_train_test_multi_level_info.pkl`

## Training

Edit the hyperparameters at the top of `train.py` (or in the `__main__`
block, which runs one job per dataset) and run:

```bash
python train.py
```

Checkpoints and per-epoch accuracy logs are written to
`./experiments/<setname>/<model_name>/code-<timestamp>/`.

## Inference

Open `load_model_predict.ipynb`, point `model_path` at a trained checkpoint,
and run it against images placed in `./test`.
