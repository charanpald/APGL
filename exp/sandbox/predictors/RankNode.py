import numpy 
from apgl.util.Parameter import Parameter 

class RankNode(object):
    """
    A simple class to model a node in a TreeRank tree.
    """
    def __init__(self, trainInds, featureInds):
        self.trainInds = trainInds
        self.featureInds = featureInds

        self.pure = False
        self.leafNode = False
        #This is the classifier for the node for prediction
        self.leafRank = None
        #A score for the leaf nodes
        self.score = None
        #The indices of the test examples 
        self.testInds = None

    def getFeatureInds(self):
        return self.featureInds

    def getTestInds(self):
        return self.testInds

    def setTestInds(self, testInds):
        Parameter.checkClass(testInds, numpy.ndarray)
        self.testInds = testInds

    def getTrainInds(self):
        return self.trainInds

    def setPure(self, pure):
        Parameter.checkBoolean(pure)
        self.pure = pure

    def isPure(self):
        return self.pure

    def setIsLeafNode(self, leafNode):
        Parameter.checkBoolean(leafNode)
        self.leafNode = leafNode

    def isLeafNode(self):
        return self.leafNode

    def setLeafRank(self, leafRank):
        self.leafRank = leafRank

    def getLeafRank(self):
        return self.leafRank

    def setScore(self, score):
        Parameter.checkFloat(score, 0.0, float("inf"))
        self.score = score

    def getScore(self):
        return self.score

    def __str__(self):
        outputStr = "Size: " + str(self.trainInds.shape[0]) + ", pure: " + str(self.pure) + ", isLeafNode: " + str(self.leafNode)
        outputStr += ", score: " + str(self.score)
        return outputStr