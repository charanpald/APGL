

"""
A class to compute a series of graph statistics over a sequence of subgraphs using 
igraph graph objects. 
"""
import gc 
import numpy
import logging 
import igraph 

from apgl.util.Util import Util
from apgl.util.Parameter import Parameter 

class GraphStatistics(object):
    def __init__(self):
        self.numVerticesIndex = 0
        self.numEdgesIndex = 1
        self.numDirEdgesIndex = 2
        self.maxComponentSizeIndex = 3
        self.numComponentsIndex = 4
        self.meanComponentSizeIndex = 5
        self.meanDegreeIndex = 6
        self.diameterIndex = 7
        self.effectiveDiameterIndex = 8
        self.densityIndex = 9
        self.powerLawIndex = 10
        self.geodesicDistanceIndex = 11
        self.harmonicGeoDistanceIndex = 12
        self.geodesicDistMaxCompIndex = 13
        self.meanTreeSizeIndex = 14
        self.meanTreeDepthIndex = 15
        self.numNonSingletonComponentsIndex = 16
        self.maxComponentEdgesIndex = 17
        self.numTriOrMoreComponentsIndex = 18
        self.secondComponentSizeIndex = 19
        self.maxTreeSizeIndex = 20
        self.maxTreeDepthIndex = 21
        self.secondTreeSizeIndex = 22
        self.secondTreeDepthIndex = 23
        self.maxCompMeanDegreeIndex = 24

        self.numTreesIndex = 25
        self.numNonSingletonTreesIndex = 26 

        self.numStats = 27
        self.q = 0.9
        self.printStep = 5
        self.vectorPrintStep = 1

        self.useFloydWarshall = False 

    def getNumStats(self):
        return self.numStats 

    def strScalarStatsArray(self, statsArray):
        """
        Get a string representation of the scalar statastics array.
        """
        s = "Num vertices: " + str(statsArray[self.numVerticesIndex]) + "\n"
        s += "Num edges: " + str(statsArray[self.numEdgesIndex]) + "\n"
        s += "Num directed edges: " + str(statsArray[self.numDirEdgesIndex]) + "\n"
        s += "Max component size: " + str(statsArray[self.maxComponentSizeIndex]) + "\n"
        s += "Max component edges: " + str(statsArray[self.maxComponentEdgesIndex]) + "\n"
        s += "Max component mean degree: " + str(statsArray[self.maxCompMeanDegreeIndex]) + "\n"
        s += "Max component geodesic disance: " + str(statsArray[self.geodesicDistMaxCompIndex]) + "\n"
        s += "2nd largest component size: " + str(statsArray[self.secondComponentSizeIndex]) + "\n"
        s += "Num components: " + str(statsArray[self.numComponentsIndex]) + "\n"
        s += "Mean component size: " + str(statsArray[self.meanComponentSizeIndex]) + "\n"
        s += "Mean degree: " + str(statsArray[self.meanDegreeIndex]) + "\n"
        s += "Diameter: " + str(statsArray[self.diameterIndex]) + "\n"
        s += "Effective diameter: " + str(statsArray[self.effectiveDiameterIndex]) + "\n"
        s += "Density: " + str(statsArray[self.densityIndex]) + "\n"
        s += "Power law exponent: " + str(statsArray[self.powerLawIndex]) + "\n"
        s += "Geodesic distance: " + str(statsArray[self.geodesicDistanceIndex]) + "\n"
        s += "Harmonic geodesic distance: " + str(statsArray[self.harmonicGeoDistanceIndex]) + "\n"
        s += "Mean tree size: " + str(statsArray[self.meanTreeSizeIndex]) + "\n"
        s += "Mean tree depth: " + str(statsArray[self.meanTreeDepthIndex]) + "\n"
        s += "Num non-singleton components: " + str(statsArray[self.numNonSingletonComponentsIndex]) + "\n"
        s += "Num tri-or-more components: " + str(statsArray[self.numTriOrMoreComponentsIndex]) + "\n"

        return s 

    def scalarStatistics(self, graph, slowStats=True, treeStats=False):
        """
        Find a series of statistics for the given input graph which can be represented
        as scalar values. Return results as a vector.
        """
        if graph.is_directed(): 
           raise ValueError("Only works on undirected graphs")     
        
        #This method is a bit of a mess 
        Parameter.checkBoolean(slowStats)
        Parameter.checkBoolean(treeStats)
        
        statsArray = numpy.ones(self.numStats)*-1
        statsArray[self.numVerticesIndex] = graph.vcount()
        statsArray[self.numEdgesIndex] = graph.ecount()
        statsArray[self.numDirEdgesIndex] = graph.as_directed().ecount()
        statsArray[self.densityIndex] = graph.density()

        logging.debug("Finding connected components")
        subComponents = graph.components()
        logging.debug("Done")
        statsArray[self.numComponentsIndex] = len(subComponents)
        
        nonSingletonSubComponents = [c for c in subComponents if len(c) > 1]
        statsArray[self.numNonSingletonComponentsIndex] = len(nonSingletonSubComponents)

        triOrMoreSubComponents = [c for c in subComponents if len(c) > 2]
        statsArray[self.numTriOrMoreComponentsIndex] = len(triOrMoreSubComponents)
        
        componentSizes =  numpy.array([len(c) for c in subComponents])
        inds = numpy.flipud(numpy.argsort(componentSizes))

        logging.debug("Studying max component")
        if len(subComponents) != 0:
            maxCompGraph = graph.subgraph(subComponents[inds[0]])
            statsArray[self.maxComponentSizeIndex] = len(subComponents[inds[0]])

            if len(subComponents) >= 2:
                statsArray[self.secondComponentSizeIndex] = len(subComponents[inds[1]])

            statsArray[self.maxComponentEdgesIndex] = maxCompGraph.ecount()
            statsArray[self.meanComponentSizeIndex] = componentSizes.mean()
            statsArray[self.maxCompMeanDegreeIndex] = numpy.mean(maxCompGraph.degree(mode=igraph.OUT))
        else:
            statsArray[self.maxComponentSizeIndex] = 0
            statsArray[self.maxComponentEdgesIndex] = 0 
            statsArray[self.meanComponentSizeIndex] = 0
            statsArray[self.geodesicDistMaxCompIndex] = 0

        if graph.vcount() != 0:
            statsArray[self.meanDegreeIndex] = numpy.mean(graph.degree(mode=igraph.OUT))
        else:
            statsArray[self.meanDegreeIndex] = 0
            
        if slowStats:
            logging.debug("Computing diameter")
            statsArray[self.diameterIndex] = graph.diameter()
            #statsArray[self.effectiveDiameterIndex] = graph.effectiveDiameter(self.q, P=P)
            #statsArray[self.powerLawIndex] = graph.fitPowerLaw()[0]
            logging.debug("Computing geodesic distance")
            statsArray[self.geodesicDistanceIndex] = graph.average_path_length()

            if len(subComponents) != 0:
                statsArray[self.geodesicDistMaxCompIndex] = graph.average_path_length(P=P, vertexInds=list(subComponents[inds[0]]))

        return statsArray
        
    def vectorStatistics(self, graph): 
        """
        Find vector statistics over the graph 
        """
        if graph.is_directed(): 
           raise ValueError("Only works on undirected graphs")             
        
        statsList = []
        #Degree distribution, component distribution 
        
        statsList.append(graph.degree_distribution())
        statsList.append(graph.components().size_histogram())
        
        return statsList 
        