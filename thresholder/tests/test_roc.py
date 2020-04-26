import unittest
import numpy as np
from thresholder import ROC_Thresholder

thres = ROC_Thresholder()
predict_proba = []
for _ in range(1000):
    temp = np.abs(np.random.normal(size=3))
    temp /= np.sum(temp)
    predict_proba.append(temp)
predict_proba = np.array(predict_proba)
true_Y = np.random.randint(3, size=1000)
thres.fit(predict_proba, true_Y)
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