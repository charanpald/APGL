import datetime
import numpy
import logging 
import itertools
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.DictGraph import DictGraph

class CitationIterGenerator(object):
    """
    A class to load the high energy physics date and generate an iterator.
    """
    def __init__(self):
        dataDir = PathDefaults.getDataDir() + "cluster/"
        edgesFilename = dataDir + "Cit-HepTh.txt"
        dateFilename = dataDir + "Cit-HepTh-dates.txt"

        #Note the IDs are integers but can start with zero so we prefix "1" to each ID 
        edges = []
        file = open(edgesFilename, 'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()

        for line in file:
            (vertex1, sep, vertex2) = line.partition("\t")
            vertex1 = vertex1.strip()
            vertex2 = vertex2.strip()
            edges.append([int("1" + vertex1), int("1" + vertex2)])

        file.close()
        edges = numpy.array(edges, numpy.int)
        logging.info("Loaded edge file " + str(edgesFilename) + " with " + str(edges.shape[0]) + " edges")

        #Keep an edge graph 
        graph = DictGraph(False)
        graph.addEdges(edges)
        logging.info("Created directed citation graph with " + str(graph.getNumEdges()) + " edges and " + str(graph.getNumVertices()) + " vertices")

        edgeVertexSet = set(graph.getAllVertexIds())

        #Read in the dates articles appear in a dict which used the year and month
        #as the key and the value is a list of vertex ids. For each month we include
        #all papers uploaded that month and those directed cited by those uploads. 
        vertexIds = {}

        file = open(dateFilename, 'r')
        file.readline()
        ind = 0

        for line in file:
            (id, sep, date) = line.partition("\t")
            id = id.strip()
            dt = datetime.datetime.strptime(date.strip(), "%Y-%m-%d")
            intId = int("1" + id)

            if (dt.month, dt.year) not in vertexIds:
                vertexIds[(dt.month, dt.year)] = [intId]
            else:
                vertexIds[(dt.month, dt.year)].append(intId)

            if intId in edgeVertexSet:
                vertexIds[(dt.month, dt.year)].extend(graph.neighbours(intId))
            ind += 1

        file.close()
        logging.info("Loaded date file " + str(dateFilename) + " with " + str(ind) + " dates")

        self.vertexIds = vertexIds

        graph = DictGraph()
        graph.addEdges(edges)
        logging.info("Created undirected citation graph with " + str(graph.getNumEdges()) + " edges and " + str(graph.getNumVertices()) + " vertices")
        self.graph = graph
        self.edges = edges

        self.endYear = -1
        self.endMonth = -1
        self.startYear = 3000
        self.startMonth = 13

        for (month, year) in vertexIds.keys():
            if (month <= self.startMonth and year <= self.startYear) or year < self.startYear:
                self.startMonth = month
                self.startYear = year

            if (month >= self.endMonth and year >= self.endYear) or year > self.endYear:
                self.endMonth = month
                self.endYear = year

        logging.info("Starting date: " + str((self.startMonth, self.startYear)))
        logging.info("Ending date: " + str((self.endMonth, self.endYear)))

    def getIterator(self):
        """
        Return an iterator which outputs the citation graph for each month. Note
        that the graphs are undirected but we make them directed.
        """
        class CitationIterator():
            def __init__(self, graph, vertexIds, startMonth, startYear, endMonth, endYear):
                """
                We start counting at startMonth,startYear and end counting at endMonth, endYear
                """
                self.graph = graph
                self.vertexIds = vertexIds

                self.W = graph.getSparseWeightMatrix().tocsr()
                self.vertexNames = numpy.array(graph.getAllVertexIds(), numpy.int)

                self.minGraphSize = 50

                self.currentVertexIds = numpy.array([], numpy.int)
                self.currentMonth = startMonth
                self.currentYear = startYear

                self.lastSubgraphInds = numpy.array([])

                #We increment the end time to make the termination check easier
                (self.endMonth, self.endYear) = self.incrementMonth(endMonth, endYear)

            def __iter__(self):
                return self

            def incrementMonth(self, month, year):
                """
                Increment the month and year by 1 month and return new values.
                """
                month += 1
                if month == 13:
                    month = 1
                    year += 1

                return (month, year)

            def next(self):
                if self.currentMonth == self.endMonth and self.currentYear == self.endYear:
                    raise StopIteration
                else:
                    subgraphSize = 0

                    while subgraphSize < self.minGraphSize:
                        newVertexIds = numpy.intersect1d(self.vertexIds[(self.currentMonth, self.currentYear)], self.vertexNames)
                        self.currentVertexIds = numpy.union1d(self.currentVertexIds, newVertexIds)
                        (self.currentMonth, self.currentYear) = self.incrementMonth(self.currentMonth, self.currentYear)

                        #We need to make sure the order of vertexIds is consistent
                        sortedInds = numpy.argsort(self.vertexNames)
                        sortedVertexNames = self.vertexNames[sortedInds]

                        currentVertexInds = numpy.searchsorted(sortedVertexNames, self.currentVertexIds)
                        subgraphInds = sortedInds[currentVertexInds]
                        subgraphSize = subgraphInds.shape[0]

                        #Just make sure we add new indices at the end
                        additionalInds = numpy.setdiff1d(subgraphInds, self.lastSubgraphInds)
                        subgraphInds = numpy.r_[self.lastSubgraphInds, additionalInds]

                    self.lastSubgraphInds = subgraphInds

                    subW = self.W[subgraphInds, :][:, subgraphInds]
                    return subW

        #Note: creating an undirected graph
        iterator = CitationIterator(self.graph, self.vertexIds, self.startMonth, self.startYear, self.endMonth, self.endYear)

        #Check how many vertices have dates
        #vertexIdsDate = numpy.array([])
        #for k in self.vertexIds.keys():
        #    vertexIdsDate = numpy.union1d(vertexIdsDate, self.vertexIds[k])

        #vertexIds = numpy.array(graph.getAllVertexIds(), numpy.int)
        #vertexIdsDate = numpy.array(self.vertexIds.keys(), numpy.int)

        return iterator