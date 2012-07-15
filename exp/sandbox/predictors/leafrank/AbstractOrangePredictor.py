import numpy 
from apgl.predictors.AbstractWeightedPredictor import AbstractWeightedPredictor

class AbstractOrangePredictor(AbstractWeightedPredictor):
    def labelsToInds(self, y):
        """
        Convert labels to indices.
        """
        newY = numpy.zeros(y.shape[0])
        newY[y==self.worstResponse] = 0
        newY[y==self.bestResponse] = 1
        return newY

    def indsToLabels(self, newY):
        y = numpy.zeros(newY.shape[0])
        y[newY==0] = self.worstResponse
        y[newY==1] = self.bestResponse
        return y

    def __str__(self):
        return str(self.learner)