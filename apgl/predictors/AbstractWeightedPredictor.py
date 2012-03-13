from apgl.util.Parameter import Parameter
from apgl.predictors.AbstractPredictor import AbstractPredictor

class AbstractWeightedPredictor(AbstractPredictor):
    def __init__(self):
        super(AbstractWeightedPredictor, self).__init__()
        #Weight of positive examples
        self.bestResponse = 1
        self.weight = 0.5

    def setWeight(self, weight):
        """
        :param weight: the weight on the positive examples between 0 and 1 (the negative weight is 1-weight)
        :type weight: :class:`float`
        """
        Parameter.checkFloat(weight, 0.0, 1.0)
        self.weight = weight

    def getWeight(self):
        """
        :return: the weight on the positive examples between 0 and 1 (the negative weight is 1-weight)
        """
        return self.weight

    def setBestResponse(self, bestResponse):
        """
        :param bestResponse: the label corresponding to "positive"
        :type bestResponse: :class:`int`
        """
        Parameter.checkInt(bestResponse, -float('inf'), float('inf'))
        self.bestResponse = bestResponse

    def getBestResponse(self):
        """
        :return: the label corresponding to "positive"
        """
        return self.bestResponse