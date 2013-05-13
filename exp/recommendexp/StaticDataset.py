
"""
Wrap a single static dataset.
"""

from scipy.io import mmread

class StaticMatrixMarketDataset(object):
    def __init__(self,train_file,test_file):
        """
        Read datasets from the specified files.
        """
        train = mmread(train_file)
        test = mmread(test_file)

        self.trainXList = [train]
        self.testXList = [test]

    def getTrainIteratorFunc(self):
        def trainIteratorFunc():
            return iter(self.trainXList)

        return trainIteratorFunc

    def getTestIteratorFunc(self):
        def testIteratorFunc():
            return iter(self.testXList)

        return testIteratorFunc

