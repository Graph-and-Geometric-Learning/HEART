We provided the datasets and preprocessing scripts used in the paper, including:
- MIMIC-III
    - mimic_pretrain.pkl: the pretraining dataset for the MIMIC-III dataset
    - mimic_downstream.pkl: this dataset includes the finetuning, validation and test datasets for the Death, PLOS, and Readmission prediction tasks
    - mimic_nextdiag_6m.pkl: this dataset includes the finetuning, validation and test datasets for the next diagnosis prediction in 6 months tasks
    - mimic_nextdiag_12m.pkl: this dataset includes the finetuning, validation and test datasets for the next diagnosis prediction in 12 months tasks
    - MIMIC-III.ipynb: the preprocessing script for the MIMIC-III dataset
- eICU
    - eicu_pretrain.pkl: the pretraining dataset for the eICU dataset
    - eicu_downstream.pkl: this dataset includes the finetuning, validation and test datasets for the Death and PLOS prediction tasks
    - eICU.ipynb: the preprocessing script for the eICU dataset

## Example

```
import pickle

train_data, val_data, test_data = pickle.load(open('./dataset/mimic_downstream.pkl', 'rb'))
print(train_data.head(5))
```