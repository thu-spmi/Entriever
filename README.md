# Code for Entriever
This repository contains the code for the paper “Entriever: Energy-based Retriever for Knowledge-Grounded Dialog Systems”.

## Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```

Then, unzip the data files:
```Shell
unzip data_e2e.zip
```
## Data Description
The data for the MobileCS dataset is under the directory of data/seretod.
As the access to the data for seretod challenge needs to be authentized by China Mobile, you need to apply for the data usage. More details for the seretod challenge data can be referred to this repo: https://github.com/SereTOD/SereTOD2022. 
The data for the other 3 datasets is under the directory of data_e2e.

## Training and Testing

The training and evaluation results of all three methods can be got by the python file retrieve_kb.py.
Set the dataset to select the experiments, and it should be in the following 4 datasets: 
```Python
['woz2.1', 'seretod', 'camrest', 'incar']
```

### Training baseline retrievers
To train the baseline retrievers, the config should be set to:
```Python
cfg.train_ebm = False
cfg.train_retrieve = True
```

### Training entrievers

After training the baseline retrievers, you can train the entriever. To train the entriever, the config should be set to:
```Python
cfg.train_ebm = True
cfg.train_retrieve = False
```

Use "cfg.train_ebm_mis" to select whether to train with mis sampling or train with is sampling.
Use "cfg.residual" to select whether to train with residual modelling.

### Testing

To test the baseline retrievers, the config should be set to:
```Python
cfg.train_ebm = False
cfg.train_retrieve = False
test_retrieve = True
```

To test the retrievers, the config should be set to:
```Python
cfg.train_ebm = False
cfg.train_retrieve = False
test_retrieve = False
```

### Using Entriever in semi-supervised dialog systems
Building a semi-supervised dialog system is complicated, we employ our entriever upon the origin JSA-KRTOD code (https://github.com/thu-spmi/JSA-KRTOD/).
To run the JSA-KRTOD system with entriever, you need to substitute main.py for JSA-KRTOD with the main.py in this repo.
