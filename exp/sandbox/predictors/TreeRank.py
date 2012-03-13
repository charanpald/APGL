"""
A Python implementation of TreeRank. 
"""
import numpy
import scikits.learn.cross_val as cross_val
from apgl.graph.DictTree import DictTree
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.metabolomics.RankNode import RankNode
from apgl.metabolomics.AbstractTreeRank import AbstractTreeRank
from apgl.metabolomics.leafrank.MajorityPredictor import MajorityPredictor
from apgl.util.Evaluator import Evaluator 

class TreeRank(AbstractTreeRank):
    def __init__(self, generateLeafRank):
        """
        Create a new TreeRank object and initialise with a function that
        generates leaf rank objects, for example LinearSVM or DecisionTree. The
        left node is more positive than the right for each split.

        :param generateLeafRank: A function which generates leafranks
        :type generateLeafRank: :class:`function`
        """
        super(TreeRank, self).__init__(generateLeafRank)
        

    def getTree(self):
        """
        :return: The learned tree as DictTree object.
        """
        return self.tree 

    def classifyNode(self, tree, X, d, k):
        """
        Take a node indexed by (d, k) and perform classification using the
        leafrank and then propogate the examples the child nodes. 
        """

        node = tree.getVertex((d, k))
        testInds = node.getTestInds()
        featureInds = node.getFeatureInds()

        if not node.isLeafNode():
            leafRank = node.getLeafRank()

            predY = leafRank.predict(X[testInds, :][:, featureInds])

            leftInds = testInds[predY == self.bestResponse]
            leftNode = tree.getVertex((d+1, 2*k))
            leftNode.setTestInds(leftInds)

            rightInds = testInds[predY != self.bestResponse]
            rightNode = tree.getVertex((d+1, 2*k+1))
            rightNode.setTestInds(rightInds)

    def splitNode(self, tree, X, Y, d, k):
        """
        Take a node in a tree and classify in order to split it into 2 
        """
        node = tree.getVertex((d, k))
        inds = node.getTrainInds()
        featureInds = node.getFeatureInds()
        alpha =  numpy.sum(Y[inds]==self.bestResponse)/float(inds.shape[0])

        #Now classify

        #We have the following condition if we need to do cross validation within the node
        if Util.histogram(Y[inds])[0].min() > self.minLabelCount:
            leafRank = self.generateLeafRank()
            leafRank.setWeight(1-alpha)
        else:
            leafRank = MajorityPredictor()

        node.setLeafRank(leafRank)
        leafRank.learnModel(X[inds, :][:, featureInds], Y[inds])
        predY = leafRank.predict(X[inds, :][:, featureInds])
        
        if numpy.unique(predY).shape[0] == 2 and inds.shape[0] >= self.minSplit:
            leftInds = inds[predY == self.bestResponse]
            featureInds = numpy.sort(numpy.random.permutation(X.shape[1])[0:numpy.round(X.shape[1]*self.featureSize)])
            leftNode = RankNode(leftInds, featureInds)
            leftNode.setPure(numpy.unique(Y[leftInds]).shape[0] <= 1)
            leftNode.setIsLeafNode(d==self.maxDepth-1 or leftNode.isPure())
            leftNode.setScore((1 - float(2*k)/2**(d+1))*2**self.maxDepth)
            tree.addEdge((d, k), (d+1, 2*k))
            tree.setVertex((d+1, 2*k), leftNode)

            rightInds = inds[predY != self.bestResponse]
            featureInds = numpy.sort(numpy.random.permutation(X.shape[1])[0:numpy.round(X.shape[1]*self.featureSize)])
            rightNode = RankNode(rightInds, featureInds)
            rightNode.setPure(numpy.unique(Y[rightInds]).shape[0] <= 1)
            rightNode.setIsLeafNode(d==self.maxDepth-1 or rightNode.isPure())
            rightNode.setScore((1 - float(2*k+1)/2**(d+1))*2**self.maxDepth)
            tree.addEdge((d, k), (d+1, 2*k+1))
            tree.setVertex((d+1, 2*k+1), rightNode)
        else:
            node.setIsLeafNode(True)
            node.setScore((1 - float(k)/2**d)*2**self.maxDepth)
            
        return tree 

    def learnModel(self, X, Y):
        """
        Learn a model for a set of examples given as the rows of the matrix X,
        with corresponding labels given in the elements of 1D array Y.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :param Y: A vector of binary labels as a 1D array
        :type Y: :class:`ndarray`
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkArray(X)
        Parameter.checkArray(Y)
        if numpy.unique(Y).shape[0] != 2:
            raise ValueError("Can only accept binary labelled data")

        tree = DictTree()
        trainInds = numpy.arange(Y.shape[0])
        featureInds = numpy.sort(numpy.random.permutation(X.shape[1])[0:numpy.round(X.shape[1]*self.featureSize)]) 

        #Seed the tree
        node = RankNode(trainInds, featureInds)
        tree.setVertex((0, 0), node)

        for d in range(self.maxDepth):
            for k in range(2**d):
                if tree.vertexExists((d, k)):
                    node = tree.getVertex((d, k))

                    if not node.isPure() and not node.isLeafNode():
                        self.splitNode(tree, X, Y, d, k)

        self.tree = tree 

    def predict(self, X):
        """
        Make a prediction for a set of examples given as the rows of the matrix X.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :return: A vector of scores corresponding to each example. 
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkArray(X)

        scores = numpy.zeros(X.shape[0])
        root = self.tree.getVertex((0, 0))
        root.setTestInds(numpy.arange(X.shape[0]))

        #We go down the tree making predictions at each stage 
        for d in range(self.maxDepth+1):
            for k in range(2**d):
                if self.tree.vertexExists((d, k)):
                    self.classifyNode(self.tree, X, d, k)

                    node = self.tree.getVertex((d,k))
                    if node.isLeafNode():
                        inds = node.getTestInds()
                        scores[inds] = node.getScore()

        return scores 

    @staticmethod
    def cut(tree, d):
        """
        Cut the tree up to the depth d and return a new tree making sure the
        leaf nodes are labelled as such.

        :param X: A tree as generated using TreeRank.
        :type X: :class:`DictTree`

        :param d: The depth of the new tree. 
        :type d: :class:`int`

        :return: A new tree of depth d. 
        """
        Parameter.checkInt(d, 1, float("inf"))
        newTree = tree.cut(d)

        #Now just rename the leaf nodes
        leaves = newTree.leaves()
        for leaf in leaves: 
            vertex = newTree.getVertex(leaf)
            vertex.setIsLeafNode(True)

        return newTree

    def learnModelCut(self, X, Y, folds=4):
        """
        Perform model learning with tree cutting in order to choose a maximal
        depth. The best tree is chosen using cross validation and depths are
        selected from 0 to maxDepth. The best depth corresponds the maximal
        AUC obtained using cross validation. 

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :param Y: A vector of binary labels as a 1D array
        :type Y: :class:`ndarray`

        :param folds: The number of cross validation folds.
        :type folds: :class:`int`
        """

        indexList = cross_val.StratifiedKFold(Y, folds)
        depths = numpy.arange(1, self.maxDepth)
        meanAUCs = numpy.zeros(depths.shape[0])

        for trainInds, testInds in indexList:
            trainX, trainY = X[trainInds, :], Y[trainInds]
            testX, testY = X[testInds, :], Y[testInds]

            self.learnModel(trainX, trainY)
            fullTree = self.tree

            for i in range(fullTree.depth()):
                d = depths[i]
                self.tree = TreeRank.cut(fullTree, d)
                predTestY = self.predict(testX)

                meanAUCs[i] += Evaluator.auc(predTestY, testY)/float(folds)

        bestDepth = depths[numpy.argmax(meanAUCs)]
        self.learnModel(X, Y)
        self.tree = TreeRank.cut(self.tree, bestDepth)
