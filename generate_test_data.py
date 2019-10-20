import pandas as pd, numpy as np, joblib, os

if not os.path.exists("./data/"):
    os.makedirs("./data/")
proba = [0.35, 0.1, 0.7]
true_labels = np.random.choice(["green", "red", "blue"], 100, replace=True)
label_proba = []
for _ in range(100):
    np.random.shuffle(proba)
    temp = proba.copy()
    label_proba.append(temp)

pd.DataFrame({"label": true_labels, "green": [i[0] for i in label_proba],
"red": [i[1] for i in label_proba], "blue": [i[2] for i in label_proba]}).to_csv("./data/test_data.csv", index=False)
joblib.dump(dict({"green": 0, "red": 1, "blue": 2}), open("./data/label2idx.p", "wb"))