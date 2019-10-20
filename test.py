from config import *
import thresholder

proba_thresholder = thresholder.ROC_Thresholder()
proba_thresholder.fit(test_configs["label_prob_path"], test_configs["label_idx_dic_path"],
                        test_configs["true_labels_field"], test_configs["results_folder"])
print(proba_thresholder.transform(save=False, indices=True))
print(proba_thresholder.fit_transform(test_configs["label_prob_path"], test_configs["label_idx_dic_path"],
                        test_configs["true_labels_field"], test_configs["results_folder"], save=False, indices=True))