
import numpy 
from apgl.util import *
from apgl.graph import * 
from exp.egograph.EgoGenerator import EgoGenerator 
from apgl.io import * 

class AbstractDiffusionSimulator():
    """
    An abstract class which learns from real survey data and then runs a simulation
    over a network to model information diffusion. 
    """

    def generateRandomGraph(self, egoFileName, alterFileName, infoProb, graph):
        """
        Generate numVertices vertices according to the distributions of egos and
        alters found in egoFileName, and alterFileName. Augment with an additional
        indicator variable with probabability infoProb. Also, use graph structure
        as given by graph. 
        """
        Parameter.checkFloat(infoProb, 0.0, 1.0)

        #Learn the distribution of the egos and alters
        eCsvReader = EgoCsvReader()

        self.egoQuestionIds = eCsvReader.getEgoQuestionIds()
        (X1, _) = eCsvReader.readFile(egoFileName, self.egoQuestionIds)

        alterQuestionIds = eCsvReader.getAlterQuestionIds()
        (X2, _) = eCsvReader.readFile(alterFileName, alterQuestionIds)

        X = numpy.r_[X1, X2]

        egoGenerator = EgoGenerator()
        (mu, sigmaSq) = Util.computeMeanVar(X)
        vList = egoGenerator.generateIndicatorVertices2(graph.getNumVertices(), mu, sigmaSq, infoProb, X.min(0), X.max(0))

        graph.setVertexList(vList)
        self.graph = graph

        return self.graph