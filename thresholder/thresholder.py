import pandas as pd, numpy as np, os, matplotlib
from sklearn.metrics import roc_curve, precision_recall_curve, auc
matplotlib.use("PS")
from matplotlib import pyplot as plt
import typing

class Thresholder:
    def __init__(self, idx_label_dic=None):
        self.idx_label_dic = idx_label_dic
        self.cutoffs = None
        self.curve_methods = {"roc": ["youden", "euclidean"], "pr": ["f1", "euclidean"]}
        self.num_classes_ = None

    def fit(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], curve: str="roc", method: str="youden", save_folder: typing.Optional[str]=None) -> None:
        """
        PARAMS
        ==========
        predict_proba: Numpy Array
            Numpy array with shape (n_samples, n_features), each column contains probabilities for one feature
        true_Y: array
            array containing true labels in form of indices
        curve: str
            string can be either roc or pr
        method: str
            if curve is roc
            string can be either youden or euclidean to be used to obtain optimal cutoffs
            youden would refer to youden j statistic
            euclidean would refer to obtaining threshold with point on ROC curve closest to the top left, i.e. (0, 1)

            if curve is pr
            string can be either f1 or euclidean
            f1 would refer to f1 score
            euclidean would refer to obtaining threshold with point on PR curve closest to the top right, i.e. (1,1)

        Create dataframe containing results of optimal cutoffs
        """
        method = method.strip().lower()
        curve = curve.strip().lower()
        if method not in self.curve_methods[curve]:
            raise ValueError("method must be either {}".format(" or ".join(self.curve_methods[curve])))
        
        self.num_classes_ = predict_proba.shape[-1]
        self.idx_label_dic = self.idx_label_dic if self.idx_label_dic is not None else {idx: idx in range(self.num_classes_)}
        
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

        self.determine_optimal_cutoffs(predict_proba, true_Y, curve, method, save_folder)

    def determine_optimal_cutoffs(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], curve: str="roc", method: str="youden", save_folder: typing.Optional[str]=None) -> None:
        """
        PARAMS
        ==========
        SAME AS fit()

        """
        labels = []
        counts = []

        x_scores = []
        y_scores = []
        
        auc_scores = []
        optimal_cutoffs = []

        curve_op = roc_curve if curve == "roc" else precision_recall_curve
        x_label = "False Positive Rate" if curve == "roc" else "Recall"
        y_label = "True Positive Rate" if curve == "roc" else "Precision"

        for idx in range(self.num_classes_):
            lab_prob = predict_proba[:, idx]
            binary_true_lab = np.array([1 if i==idx else 0 for i in true_Y])

            x_score, y_score, auc_score, optimal_proba_cutoff = -1, -1, -1, -1
            if len(np.unique(binary_true_lab)) != 1:

                x_score, y_score, proba = curve_op(binary_true_lab, lab_prob) # false_pos_rate, true_pos_rate if roc_curve, precision, recall if pr curve
                
                if curve == "roc":
                    x_score, y_score, proba = x_score[1:], y_score[1:], proba[1:] # thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
                else:
                    x_score, y_score = y_score, x_score # precision is y, recall is x

                auc_score = auc(x_score, y_score)
                
                if save_folder:
                    # Save ROC Curve Plots
                    plt.figure()
                    plt.plot([0,1], [0,1], linestyle="--") # save random curve
                    plt.plot(x_scores, y_scores, marker=".", label=f"AUC = {auc_score}")
                    plt.title(f"{curve} Curve for {self.idx_label_dic[idx]}")
                    plt.ylabel(y_label)
                    plt.xlabel(x_label)
                    plt.legend()
                    plt.savefig(save_folder + str(self.idx_label_dic[idx]) + "_{}.png".format(curve))

                # Determine Optimal Probability Cutoffs Using Difference Between True Positive and False Positive Rates
                if method == "youden":
                    optimal_proba_cutoff = sorted(list(zip(y_score - x_score, proba)), key=lambda i: i[0], reverse=True)[0][1]

                elif method == "euclidean":
                    coordinates_array = np.array(list(zip(x_score, y_score)))
                    top_coord = [0, 1] if curve == "roc" else [1, 1]
                    coordinates_array -= np.array(top_coord) # coordinates (0, 1) is the top left corner of the roc plot, (1,1) is the top right of the pr plot
                    coordinates_array **= 2
                    coordinates_array = np.sum(coordinates_array, axis=1)
                    coordinates_array **= 0.5
                    optimal_proba_cutoff = sorted(list(zip(coordinates_array, proba)), key=lambda i: i[0], reverse=False)[0][1]
                else:
                    optimal_proba_cutoff = sorted(list(zip(2 * (y_score * x_score / (y_score + x_score + 1e-7)), proba)), key=lambda i: i[0], reverse=True)[0][1]

            labels.append(idx)
            counts.append(sum(binary_true_lab))

            x_scores.append(x_score)
            y_scores.append(y_score)
            auc_scores.append(auc_score)
            optimal_cutoffs.append(optimal_proba_cutoff)

        if save_folder:
            df = pd.DataFrame({"Labels": labels, "Counts": counts, x_label: x_scores,
                            y_label: y_scores, "{} AUC".format(curve.capitalize()): auc_scores, "Optimal Probability Cutoff": optimal_cutoffs})
            df.to_csv(save_folder + "{}_stats.csv".format(curve), index=False)
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

    def fit_transform(self, predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], curve: str="roc", method: str="youden", save_folder: typing.Optional[str]=None) -> list:
        self.fit(predict_proba, true_Y, curve, method, save_folder=save_folder)
        return self.transform(predict_proba)