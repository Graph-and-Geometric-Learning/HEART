# HEART: Learning better representation of EHR data with a heterogeneous relation-aware transformer

This is our PyTorch implementation for the paper:

> Tinglin Huang, Syed Asad Rizvi, Rohan Krishna Thakur, Vimig Socrates, Meili Gupta, David van Dijk, R. Andrew Taylor, Rex Ying (2024). HEART: Learning better representation of EHR data with a heterogeneous relation-aware transformer. [Paper Link](https://www.sciencedirect.com/science/article/pii/S153204642400159X). In Journal of Biomedical Informatics 159 (2024): 104741.

## Dataset Preparation

We have provided the preprocessing scripts in `dataset/` for MIMIC-III and eICU datasets respectively. Please first download the datasets from the following link:

* MIMIC-III: https://mimic.mit.edu/docs/iii/
* eICU: https://eicu-crd.mit.edu/about/eicu/


## Requirements

The code has been tested running under Python 3.10.14. The required packages are as follows:
- pytorch == 2.3.0
- torch_geometric == 2.5.3
- einops == 0.8.0

Once you finished these installation, please run install the package by running:
```
pip install -e .
```

## Organization

The code is organized as follows:
- `app/`: the main code for training and testing the model
    - `finetune.py`: the pipeline for finetuning the model on downstream tasks
    - `pretrain.py`: the pipeline for pretraining the model on the pretraining task
- `dataset/`: the code for data processing
    - `eICU.ipynb`: dataset preprocessing for eICU
    - `MIMIC-III.ipynb`: dataset preprocessing for MIMIC-III
- `models/`
    - `gnn.py`: implementation of the graph attention for the encounter-level attention
    - `HEART.py`: implementation of the pretraining and finetuning model
    - `transformer_rel.py`: implementation of the transformer with heterogeneous relations
    - `transformer.py`: implementation of the transformer
- `utils/`: utility functions including data loading pipeline
