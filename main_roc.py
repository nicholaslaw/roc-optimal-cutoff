import numpy as np, pandas as pd, joblib, os, logging
from roc_funcs import determine_optimal_cutoffs, roc_predictions
logging.basicConfig(filename="roc.log", filemode="w", level=logging.INFO)

# Define Variables
DF_PATH = "" # file path to csv containing labels and predicted probabilities for test set
TRUE_LABELS_FIELD = "" # name of column containing true labels
LABEL_INDEX_DIC = ""
MULTICLASS = True # True if multiclass and False if multilabel
SAVE_FOLDER = "./ROC_FOLDER/"

# Create Folder to Contain Results if it does not exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Load Necessary Items
print("Loading Necessary Items...")
df = pd.read_csv(DF_PATH)
lab_idx_dic = joblib.load(open(LABEL_INDEX_DIC, "rb"))
print("Done...\n")

# Obtain Optimal Probability Cutoffs for Each Class
print("Obtaining Optimal Probability Cutoffs...")
cutoffs = determine_optimal_cutoffs(df, lab_idx_dic, TRUE_LABELS_FIELD, SAVE_FOLDER)
print("Done...\n")

# Obtain Predictions
print("Obtaining Predictions Based on Optimal Probability Cutoffs...")
roc_predictions(df, lab_idx_dic, cutoffs, SAVE_FOLDER, MULTICLASS)
print("Completed!")