import unittest
import numpy as np
from thresholder import PR_Thresholder
from sklearn.ensemble import RandomForestClassifier

X_train = np.random.randn(1000, 4)
X_test = np.random.randn(1000, 4)
while True:
    train_Y = np.random.randint(3, size=1000)
    true_Y = np.random.randint(3, size=1000)
    if [len(np.unique(train_Y)), len(np.unique(true_Y))] != [3,3]:
        continue
    else:
        break

model = RandomForestClassifier()
model.fit(X_train, train_Y)
predict_proba = model.predict_proba(X_test)

thres = PR_Thresholder()
thres.fit(model, X_test, true_Y)
preds = thres.transform(predict_proba)

class Thresholder_Test(unittest.TestCase):
    def test_cutoff_dic_type(self):
        """
        Test that it can sum a list of integers
        """
        self.assertIsInstance(thres.cutoffs, dict)
        for val in thres.cutoffs.values():
            self.assertIsInstance(val, float)

    def test_transform(self):
        """
        Test that it can sum a list of fractions
        """
        self.assertIsInstance(preds, list)
        for i in preds:
            self.assertIsInstance(i, int)

if __name__ == '__main__':
    unittest.main()