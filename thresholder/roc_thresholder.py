import pandas as pd, numpy as np, os, matplotlib, joblib
from sklearn.metrics import roc_curve, roc_auc_score
matplotlib.use("PS")
from matplotlib import pyplot as plt

class ROC_Thresholder:
    def __init__(self):
        self.lab_prob_df = None
        self.lab_idx_dic = None
        self.save_folder = None
        self.cutoffs = None

    def fit(self, lab_prob_df_path, lab_idx_dic_path, true_lab_col, save_folder="./ROC/"):
        """
        lab_prob_df_path: string
            file path to dataframe containing columns with true labels, other labels with predicted probabilities
        lab_idx_dic_path: string
            file path to pickle file containing labels and their corresponding indices
        true_lab_col: string
            name of column containing true labels
        save_folder: string
            path to folder containing outputs

        Import dataset and construct label index dictionary, save dataframe containing results of optimal cutoffs
        """
        if not isinstance(lab_prob_df_path, str):
            raise TypeError("lab_prob_df_path must be a string")
        if not isinstance(lab_idx_dic_path, str):
            raise TypeError("lab_idx_dic_path must be a string")
        if not isinstance(true_lab_col, str):
            raise TypeError("true_lab_col must be a string")
        if not isinstance(save_folder, str):
            raise TypeError("save_folder must be a string")

        if lab_prob_df_path.endswith(".xlsx"):
            self.lab_prob_df = pd.read_excel(lab_prob_df_path)
        elif lab_prob_df_path.endswith(".csv"):
            self.lab_prob_df = pd.read_csv(lab_prob_df_path)
        self.lab_idx_dic = joblib.load(open(lab_idx_dic_path, "rb"))
        self.save_folder = save_folder
        self.true_lab_idx = [self.lab_idx_dic[lab] for lab in self.lab_prob_df[true_lab_col].tolist()]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder

        self.cutoffs = self.determine_optimal_cutoffs()

    def transform(self, lab_prob_df_path=None, save=False, indices=True):
        """
        lab_prob_df_path: string
            path to csv or excel containing true labels and predicted probabilities,
            this data file can be the same as the one called in fit or can be a new one
            which is similar to the one called in fit, if None then it will be the previous
            dataframe loaded in the fit method
        save: boolean
            True if want to save predictions in a column in dataframe

        returns predictions based on optimal cutoffs
        """
        if lab_prob_df_path is None:
            lab_prob_df = self.lab_prob_df
        else:
            if lab_prob_df_path.endswith(".xlsx"):
                lab_prob_df = pd.read_excel(lab_prob_df_path)
            elif lab_prob_df_path.endswith(".csv"):
                lab_prob_df = pd.read_csv(lab_prob_df_path)
        labels = list(self.lab_idx_dic.keys())
        proba_cutoffs = np.array([self.cutoffs[lab] for lab in labels])
        predictions = []
        for _, row in lab_prob_df.iterrows():
            proba_diffs = np.array(row[labels]) - proba_cutoffs
            proba_diffs = list(zip(labels, proba_diffs))
            proba_diffs.sort(key=lambda i: i[1], reverse=True)
            predictions.append(proba_diffs[0][0])
        if save:
            lab_prob_df["ROC_Prediction"] = predictions
            lab_prob_df.to_csv(self.save_folder + "roc_predictions.csv", index=False)
        if indices:
            predictions = [self.lab_idx_dic[lab] for lab in predictions]
        return predictions

    def fit_transform(self, lab_prob_df_path, lab_idx_dic_path, true_lab_col, save_folder="./ROC/", save=False, indices=True):
        self.fit(lab_prob_df_path, lab_idx_dic_path, true_lab_col, save_folder="./ROC/")
        return self.transform(save=save, indices=indices)

    def determine_optimal_cutoffs(self):
        labels = []
        counts = []
        true_positive_rates = []
        false_positives_rates = []
        roc_auc_scores = []
        optimal_cutoffs = []

        for lab, idx in self.lab_idx_dic.items():
            lab_prob = self.lab_prob_df[lab].tolist()
            binary_true_lab = np.array([1 if i==idx else 0 for i in self.true_lab_idx])
            false_pos_rate, true_pos_rate, proba = roc_curve(binary_true_lab, lab_prob)
            auc_score = roc_auc_score(binary_true_lab, lab_prob)

            # Save ROC Curve Plots
            plt.figure()
            plt.plot([0,1], [0,1], linestyle="--") # save random curve
            plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {auc_score}")
            plt.title(f"ROC Curve for {lab}")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend(loc="lower right")
            plt.savefig(self.save_folder + str(lab) + "_roc.png")

            # Determine Optimal Probability Cutoffs Using Difference Between True Positive and False Positive Rates
            optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]

            labels.append(lab)
            counts.append(sum(binary_true_lab))
            true_positive_rates.append(true_pos_rate)
            false_positives_rates.append(false_pos_rate)
            roc_auc_scores.append(auc_score)
            optimal_cutoffs.append(optimal_proba_cutoff)

        df = pd.DataFrame({"Labels": labels, "Counts": counts, "True Positive Rate": true_positive_rates,
                        "False Positive Rate": false_positives_rates, "ROC AUC": roc_auc_scores, "Optimal Probability Cutoff": optimal_cutoffs})
        df.to_csv(self.save_folder + "roc_stats.csv", index=False)
        cutoffs = {lab: prob for lab, prob in zip(labels, optimal_cutoffs)}
        return cutoffs
