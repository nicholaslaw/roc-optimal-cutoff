import pandas as pd, numpy as np, os, matplotlib, joblib
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
matplotlib.use("PS")
from matplotlib import pyplot as plt
import typing

class PR_Thresholder:
    def __init__(self):
        self.idx_label_dic = None
        self.cutoffs = None

    def fit(self, model, X_test: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], method: str="prec_rec", save_folder: typing.Optional[str]=None) -> None:
        """
        model: Sklearn's model object
            Fitted model object
        X_test: Numpy Array
            Numpy array with shape (n_samples, n_features), i.e. feature matrix for test features
        true_Y: array
            array containing true labels in form of indices
        method: str
            string can be either prec_rec or euclidean to be used to obtain optimal cutoffs
            prec_rec would refer to obtaining probability threshold based on the point where precision and recall scores are the closest to each other
            euclidean would refer to obtaining threshold with point on PR curve closest to the top right, i.e. (1,1)

        Create dataframe containing results of optimal cutoffs
        """
        method = method.strip().lower()
        if method not in ["prec_rec", "euclidean"]:
            raise ValueError("method must be either prec_rec or euclidean")
        predict_proba = model.predict_proba(X_test)
        self.num_classes_ = predict_proba.shape[-1]
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        self.determine_optimal_cutoffs(model, X_test, predict_proba, true_Y, method, save_folder)

    def determine_optimal_cutoffs(self, model, X_test: typing.Union[np.ndarray, list], predict_proba: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], method: str="prec_rec", save_folder: typing.Optional[str]=None) -> None:
        labels = []
        counts = []
        precision_scores = []
        recall_scores = []
        optimal_cutoffs = []

        for idx in range(self.num_classes_):
            lab_prob = predict_proba[:, idx]
            binary_true_lab = np.array([1 if i==idx else 0 for i in true_Y])
            precision_, recall_, proba = precision_recall_curve(binary_true_lab, lab_prob)
            
            if save_folder:
                # Save ROC Curve Plots
                plt.figure()
                plt.plot(recall_, precision_, marker=".")
                plt.title(f"PR Curve for {idx}")
                plt.ylabel("Recall")
                plt.xlabel("Precision")
                plt.savefig(save_folder + str(idx) + "_pr.png")

            # Determine Optimal Probability Cutoffs Using Difference Between True Positive and False Positive Rates
            if method == "prec_rec":
                optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
            elif method == "euclidean":
                coordinates_array = np.array(list(zip(recall_, precision_)))
                coordinates_array -= np.array([1, 1]) # coordinates (0, 1) is the top left corner of the roc plot
                coordinates_array **= 2
                coordinates_array = np.sum(coordinates_array, axis=1)
                coordinates_array **= 0.5
                optimal_proba_cutoff = sorted(list(zip(coordinates_array, proba)), key=lambda i: i[0], reverse=False)[0][1]


            labels.append(idx)
            counts.append(sum(binary_true_lab))
            precision_scores.append(precision_)
            recall_scores.append(recall_)
            optimal_cutoffs.append(optimal_proba_cutoff)
        if save_folder:
            df = pd.DataFrame({"Labels": labels, "Counts": counts, "Precision Scores": precision_scores,
                            "Recall Scores": recall_scores, "Optimal Probability Cutoff": optimal_cutoffs})
            df.to_csv(save_folder + "pr_stats.csv", index=False)
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

    def fit_transform(self, model, X_test: typing.Union[np.ndarray, list], true_Y: typing.Union[np.ndarray, list], predict_proba: typing.Union[np.ndarray, list], method: str="prec_rec", save_folder: typing.Optional[str]=None) -> list:
        self.fit(model, X_test, true_Y, method, save_folder=save_folder)
        return self.transform(predict_proba)