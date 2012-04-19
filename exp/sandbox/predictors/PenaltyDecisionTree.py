
class PenaltyDecisionTree(AbstractPredictor): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="reg", pruneType="none", alphaThreshold=0.0, folds=5):
        """
        Need a minSplit for the internal nodes and one for leaves. 
        """
        super(DecisionTreeLearner, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        self.pruneType = pruneType 
        self.alphaThreshold = alphaThreshold
        self.folds = 5