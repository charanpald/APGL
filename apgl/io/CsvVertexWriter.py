"""
A class to write the vertices of a graph out to a file
"""
import csv
import logging

class CsvVertexWriter():
    def __init__(self):
        pass

    def writeToFile(self, fileName, graph):
        logging.info('Writing to file: ' + fileName + ".csv")
        indices = graph.getAllVertexIds()
        writer = csv.writer(open(fileName + ".csv", 'w'), delimiter=',', lineterminator="\n")

        for i in indices:
            writer.writerow(graph.getVertex(i))

        logging.info("Wrote " + str(graph.getNumVertices()) + " vertices.")
            
        


