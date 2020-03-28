import pandas as pd, numpy as np, os, matplotlib, joblib
from sklearn.metrics import roc_curve, roc_auc_score
matplotlib.use("PS")
from matplotlib import pyplot as plt
import typing

class ROC_Thresholder:
    def __init__(self):
        self.idx_label_dic = None
        self.cutoffs = None

    def fit(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], method: str="youden", save_folder: typing.Optional[str]=None) -> None:
        """
        predict_proba: Numpy Array
            Numpy array with shape (n_samples, n_features), each column contains probabilities for one feature
        true_Y: array
            array containing true labels in form of indices
        method: str
            string can be either youden or euclidean to be used to obtain optimal cutoffs
            youden would refer to youden j statistic
            euclidean would refer to obtaining threshold with point on ROC curve closest to the top left, i.e. (0, 1)

        Create dataframe containing results of optimal cutoffs
        """
        method = method.strip().lower()
        if method not in ["youden", "euclidean"]:
            raise ValueError("method must be either youden or euclidean")
        self.num_classes_ = predict_proba.shape[-1]
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        self.determine_optimal_cutoffs(predict_proba, true_Y, method, save_folder)

    def determine_optimal_cutoffs(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], method: str="youden", save_folder: typing.Optional[str]=None) -> None:
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
            if method == "youden":
                optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
            elif method == "euclidean":
                coordinates_array = np.array(list(zip(true_pos_rate, false_pos_rate)))
                coordinates_array -= np.array([0, 1]) # coordinates (0, 1) is the top left corner of the roc plot
                coordinates_array **= 2
                coordinates_array = np.sum(coordinates_array, axis=1)
                coordinates_array **= 0.5
                optimal_proba_cutoff = sorted(list(zip(coordinates_array, proba)), key=lambda i: i[0], reverse=False)[0][1]


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

    def transform(self, predict_proba: typing.Union[np.ndarray, list]) -> list:
        """
        predict_proba: Numpy Array
            Numpy array with shape (n_samples, n_features), each column contains probabilities for one feature

        returns predictions based on optimal cutoffs
        """
        proba_cutoffs = np.array([self.cutoffs[i] for i in range(self.num_classes_)])
        labels = list(range(self.num_classes_))
        predictions = []
        for row in predict_proba:
            proba_diffs = row - proba_cutoffs
            proba_diffs = list(zip(labels, proba_diffs))
            proba_diffs.sort(key=lambda i: i[1], reverse=True)
            predictions.append(proba_diffs[0][0])
        return predictions

    def fit_transform(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], method: str="youden", save_folder: typing.Optional[str]=None) -> list:
        self.fit(predict_proba, true_Y, method, save_folder=save_folder)
        return self.transform(predict_proba)