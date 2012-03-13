import numpy 
from apgl.metabolomics.leafrank.AbstractWeightedPredictor import AbstractWeightedPredictor

class AbstractFunctionalPredictor(AbstractWeightedPredictor):
    def __init__(self):
        super(AbstractWeightedPredictor, self).__init__()
        self.waveletInds = None
        self.candidatesN = numpy.array([5, 10, 25, 50, 75, 100, 150], numpy.int)
        #self.candidatesN = 2**numpy.arange(1, 9)
        self.featureInds = None 

    def setWaveletInds(self, waveletInds):
        self.waveletInds = waveletInds

    def getWaveletInds(self):
        return self.waveletInds

    def setCandidatesN(self, candidatesN):
        self.candidatesN = candidatesN

    def getCandidatesN(self):
        return self.candidatesN

    def getFeatureInds(self):
        return self.featureInds
