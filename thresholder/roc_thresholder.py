import pandas as pd, numpy as np, os, matplotlib, joblib
from sklearn.metrics import roc_curve, roc_auc_score
matplotlib.use("PS")
from matplotlib import pyplot as plt

class ROC_Thresholder:
    def __init__(self):
        self.idx_label_dic = None
        self.cutoffs = None

    def fit(self, predict_proba, true_Y, save_folder=None):
        """
        predict_proba: Numpy Array
            Numpy array with shape (n_samples, n_features), each column contains probabilities for one feature
        true_Y: array
            array containing true labels in form of indices
        idx_lab_dic: dictionary
            dictionary containing labels (values) and their corresponding indices (keys)

        Create dataframe containing results of optimal cutoffs
        """
        if not isinstance(predict_proba, np.ndarray):
            raise TypeError("predict_proba must be a Numpy Array")
        if not isinstance(true_Y, np.ndarray):
            if not isinstance(true_Y, list):
                raise TypeError("true_Y must be a Numpy Array or a List")
        self.num_classes_ = predict_proba.shape[-1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.determine_optimal_cutoffs(predict_proba, true_Y, save_folder)

    def determine_optimal_cutoffs(self, predict_proba, true_Y, save_folder=None):
        labels = []
        counts = []
        true_positive_rates = []
        false_positives_rates = []
        roc_auc_scores = []
        optimal_cutoffs = []

        for idx in range(self.num_classes_):
            lab_prob = predict_proba[:, idx]
            binary_true_lab = np.array([1 if i==idx else 0 for i in true_Y])
            false_pos_rate, true_pos_rate, proba = roc_curve(binary_true_lab, lab_prob)
            auc_score = roc_auc_score(binary_true_lab, lab_prob)
            
            if save_folder:
                # Save ROC Curve Plots
                plt.figure()
                plt.plot([0,1], [0,1], linestyle="--") # save random curve
                plt.plot(false_pos_rate, true_pos_rate, marker=".", label=f"AUC = {auc_score}")
                plt.title(f"ROC Curve for {idx}")
                plt.ylabel("True Positive Rate")
                plt.xlabel("False Positive Rate")
                plt.legend(loc="lower right")
                plt.savefig(save_folder + str(idx) + "_roc.png")

            # Determine Optimal Probability Cutoffs Using Difference Between True Positive and False Positive Rates
            optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]

            labels.append(idx)
            counts.append(sum(binary_true_lab))
            true_positive_rates.append(true_pos_rate)
            false_positives_rates.append(false_pos_rate)
            roc_auc_scores.append(auc_score)
            optimal_cutoffs.append(optimal_proba_cutoff)
        if save_folder:
            df = pd.DataFrame({"Labels": labels, "Counts": counts, "True Positive Rate": true_positive_rates,
                            "False Positive Rate": false_positives_rates, "ROC AUC": roc_auc_scores, "Optimal Probability Cutoff": optimal_cutoffs})
            df.to_csv(save_folder + "roc_stats.csv", index=False)
        self.cutoffs = {lab: prob for lab, prob in zip(labels, optimal_cutoffs)}

    def transform(self, predict_proba):
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
        if not isinstance(predict_proba, np.ndarray):
            raise TypeError("predict_proba must be a Numpy Array")
        proba_cutoffs = np.array([self.cutoffs[i] for i in range(self.num_classes_)])
        labels = list(range(self.num_classes_))
        predictions = []
        for row in predict_proba:
            proba_diffs = row - proba_cutoffs
            proba_diffs = list(zip(labels, proba_diffs))
            proba_diffs.sort(key=lambda i: i[1], reverse=True)
            predictions.append(proba_diffs[0][0])
        return predictions

    def fit_transform(self, predict_proba, true_Y, save_folder=None):
        self.fit(predict_proba, true_Y, save_folder=save_folder)
        return self.transform(predict_proba)