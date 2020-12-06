import unittest
import numpy as np
from thresholder import Thresholder

y = np.array([0,0,1,1])
scores = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])

idx_label_dic = {idx: "dummy_{}".format(idx) for idx in range(2)}
thres = Thresholder(idx_label_dic)
thres.fit(scores, y, curve="pr", method="f1", save_folder=None)
preds = thres.transform(scores)

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