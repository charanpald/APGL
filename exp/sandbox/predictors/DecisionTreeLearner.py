from apgl.predictors.AbstractPredictor import AbstractPredictor


class DecisionTreeLearner(AbstractPredictor): 
    def __init__(self, criterion="gini", maxDepth=10, minSplit=30, type="class"):
        super(DecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        
        self.maxDepths = numpy.arange(1, 10)
        self.minSplits = numpy.arange(10, 51, 10)
        
    def learnModel(self, X, y):
        
        
        
        
        