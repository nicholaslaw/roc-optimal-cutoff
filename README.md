# Determining Optimal Probability Cutoffs with ROC Curve
## Installation
Open terminal in this directory and run either:
1. pip install the module itself
```
pip install .
```
2. setup.py (same as 1)
```
python setup.py install
```
3. docker-compose to set up an environment with module installed
```
docker-compose up -d
```
## Getting Started
```
import thresholder

proba_thresholder = thresholder.ROC_Thresholder()
proba_thresholder.fit(label_prob_path, label_idx_dic_path, true_labels_field, "./ROC_output/")
roc_preds = proba_thresholder.transform(save=False, indices=True) # don't save output as a column in dataframe containing label probabilities, roc predictions are contained in an array and they are indexed
```

## Jupyter Notebook Server
To set up a notebook server, follow step 3 of **Installation** and assuming default settings are applied, head to *http://localhost:8889/tree* to view existing or create new notebooks to perform experiments with the module.