from apgl.util import *
from apgl.predictors.AbstractPredictor import AbstractPredictor

class AbstractKernelPredictor(AbstractPredictor):
    """
    An abstract kernel predictor
    """
    
    def learnModel(self, X, y):
        Util.abstract()

    def predict(self, X):
        Util.abstract()