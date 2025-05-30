# Code for Entriever
This repository contains the code for the paper “Entriever: Energy-based Retriever for Knowledge-Grounded Dialog Systems”.

## Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```

## Data Description
The data for the MobileCS dataset is under the directory of data/seretod.
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