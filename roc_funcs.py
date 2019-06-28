import matplotlib, pandas as pd, numpy as np, logging
from sklearn.metrics import roc_curve, roc_auc_score
matplotlib.use("PS")
from matplotlib import pyplot as plt

def determine_optimal_cutoffs(lab_prob_df, lab_idx_dic, true_lab, save_folder):
    """
    lab_prob_df: Pandas DataFrame
        contains label columns with their probabilities
    lab_idx_dic: dictionary
        contains distinct labels and their corresponding indices
    true_lab: string
        name of column in the pandas dataframe containing true labels
    save_folder: string
        path to folder to save results
    
    Saves dataframe containing distinct labels, counts, true positive rates, false positive rates, roc auc scores and optimal probability cutoffs
    """
    labels = []
    counts = []
    true_positive_rates = []
    false_positives_rates = []
    roc_auc_scores = []
    optimal_cutoffs = []
    true_lab_idx = [lab_idx_dic[lab] for lab in lab_prob_df[true_lab].tolist()]

    for lab, idx in lab_idx_dic.items():
        lab_prob = lab_prob_df[lab].tolist()
        binary_true_lab = np.array([1 if i==idx else 0 for i in true_lab_idx])
        false_pos_rate, true_pos_rate, proba = roc_curve(binary_true_lab, lab_prob)
        auc_score = roc_auc_score(binary_true_lab, lab_prob)

        # Save ROC Curve for Each Class
        plt.figure()
        plt.plot([0,1], [0,1], linestyle="--") # save random curve
        plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {auc_score}")
        plt.title(f"ROC Curve for {lab}")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(save_folder + str(lab) + "_roc.png")

        # Determine Optimal Probability Cutoffs Using Difference Between True Positive and False Positive Rates
        optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]

        labels.append(lab)
        counts.append(sum(binary_true_lab))
        true_positive_rates.append(true_pos_rate)
        false_positives_rates.append(false_pos_rate)
        roc_auc_scores.append(auc_score)
        optimal_cutoffs.append(optimal_proba_cutoff)
    
    df = pd.DataFrame({"Labels": labels, "Counts": counts, "True Positive Rate": true_positive_rates, "False Positive Rate": false_positives_rates, "ROC AUC": roc_auc_scores, "Optimal Probability Cutoff": optimal_cutoffs})
    df.to_csv(save_folder + "roc_stats.csv", index=False)
    cutoffs = {lab: prob for lab, prob in zip(labels, optimal_cutoffs)}
    return cutoffs

def roc_predictions(lab_prob_df, lab_idx_dic, cutoffs, save_folder, multiclass=True):
    """
    lab_prob_df: Pandas DataFrame
        contains all labels' probabilities
    lab_idx_dic: dictionary
        contains distinct labels and their corresponding indices
    cutoffs: dictionary
        contains labels and their corresponding optimal probability cutoffs
    save_folder: string
        path to folder to save predictions
    multiclass: boolean
        True if multiclass and False if multilabel
    """
    labels = list(lab_idx_dic.keys())
    proba_cutoffs = np.array([cutoffs[lab] for lab in labels])
    predictions = []
    for _, row in lab_prob_df.iterrows():
        proba_diffs = row[labels].toarray() - proba_cutoffs
        proba_diffs = list(zip(labels, proba_diffs))
        proba_cutoffs.sort(key=lambda i: i[1], reverse=True)
        temp = []
        for idx, lab, prob in enumerate(proba_cutoffs):
            if multiclass == True and idx > 0:
                break
            if prob >= 0:
                temp.append(lab)
        predictions.append("||".join(temp))
    lab_prob_df["Prediction"] = predictions
    lab_prob_df.to_csv(save_folder + "roc_predictions.csv", index=False)