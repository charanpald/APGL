import numpy 
from apgl.util.Util import Util
from apgl.predictors.AbstractWeightedPredictor import AbstractWeightedPredictor

class MajorityPredictor(AbstractWeightedPredictor):
    def learnModel(self, X, y):
        """
        Basically figure out the majority label
        """
        self.majorLabel = Util.mode(y)

    def predict(self, X):
        predY = numpy.ones(X.shape[0])*self.majorLabel
        return predY
