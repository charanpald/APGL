
import orange
import orngTree
import numpy
from apgl.util.Parameter import Parameter
from exp.sandbox.predictors.leafrank.AbstractOrangePredictor import AbstractOrangePredictor

class DecisionTree(AbstractOrangePredictor):
    def __init__(self):
        super(DecisionTree, self).__init__()
        self.maxDepth = 10
        self.minSplit = 30
        #Post-pruned using m-error estimate pruning method with parameter m 
        self.m = 2 

    def setM(self, m):
        self.m = m

    def getM(self):
        return self.m 

    def setMinSplit(self, minSplit):
        Parameter.checkInt(minSplit, 0, float('inf'))
        self.minSplit = minSplit

    def getMinSplit(self):
        return self.minSplit 

    def setMaxDepth(self, maxDepth):
        Parameter.checkInt(maxDepth, 1, float('inf'))
        self.maxDepth = maxDepth

    def getMaxDepth(self):
        return self.maxDepth 

    def learnModel(self, X, y):
        if numpy.unique(y).shape[0] != 2:
            raise ValueError("Can only operate on binary data")

        classes = numpy.unique(y)
        self.worstResponse = classes[classes!=self.bestResponse][0]

        #We need to convert y into indices
        newY = self.labelsToInds(y)

        XY = numpy.c_[X, newY]
        attrList = []
        for i in range(X.shape[1]):
            attrList.append(orange.FloatVariable("X" + str(i)))

        attrList.append(orange.EnumVariable("y"))
        attrList[-1].addValue(str(self.bestResponse))
        attrList[-1].addValue(str(self.worstResponse))

        self.domain = orange.Domain(attrList)
        eTable = orange.ExampleTable(self.domain, XY)

        #Weight examples and equalise
        #Equalizing computes such weights that the weighted number of examples
        #in each class is equivalent.
        preprocessor = orange.Preprocessor_addClassWeight(equalize=1)
        preprocessor.classWeights = [1-self.weight, self.weight]
        eTable, weightID = preprocessor(eTable)
        eTable.domain.addmeta(weightID, orange.FloatVariable("w"))

        self.learner = orngTree.TreeLearner(m_pruning=self.m, measure="gainRatio")
        self.learner.max_depth = self.maxDepth
        self.learner.stop = orange.TreeStopCriteria_common()
        self.learner.stop.minExamples = self.minSplit
        self.classifier = self.learner(eTable, weightID)

    def getLearner(self):
        return self.learner

    def getClassifier(self):
        return self.classifier 

    def predict(self, X):
        XY = numpy.c_[X, numpy.zeros(X.shape[0])]
        eTable = orange.ExampleTable(self.domain, XY)
        predY = numpy.zeros(X.shape[0])

        for i in range(len(eTable)):
            predY[i] = self.classifier(eTable[i])

        predY = self.indsToLabels(predY)
        return predY
    
    @staticmethod
    def generate(maxDepth=10, minSplit=30, m=2):
        def generatorFunc():
            decisionTree = DecisionTree()
            decisionTree.setMaxDepth(maxDepth)
            decisionTree.setMinSplit(minSplit)
            decisionTree.setM(m)
            return decisionTree
        return generatorFunc

    @staticmethod
    def depth(tree):
        """
        Find the depth of a tree
        """
        if tree == None:
            return 0
        if not tree.branches:
            return 1
        else:
            return max([DecisionTree.depth(branch) for branch in tree.branches]) + 1