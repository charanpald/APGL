import unittest
import numpy 
from apgl.util.GraphLayoutEngine import GraphLayoutEngine
import cProfile
import pstats

class  GraphLayoutEngineTestCase(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True)

    def testGraphLayoutEngine(self):
        try:
            import wx
            import networkx
            import matplotlib
            matplotlib.use('WXAgg') # do this before importing pylab
            import matplotlib.pyplot
        except ImportError:
            pass 

        numpy.random.seed(21)

        #graph = networkx.erdos_renyi_graph(50, 0.1, seed=21)
        graph = networkx.Graph()

        #for i in range(0, 10):
        #    graph.add_node(i)


        
        graph.add_edge(0,1)
        graph.add_edge(0,2)
        graph.add_edge(0,3)
        graph.add_edge(0,4)
        graph.add_edge(0,5)
        graph.add_edge(5,6)
        graph.add_edge(0,7)
        graph.add_edge(0,8)
        

        numVertices = graph.order()
        ages = numpy.array(list(range(0, numVertices)))/float(numVertices)
        #ages = None

        layoutEngine = GraphLayoutEngine()
        nodePositions, allPositions, iterations = layoutEngine.layout(graph, ages)
        #nodePositions = networkx.spring_layout(graph)

        fig = matplotlib.pyplot.figure(figsize=(8,8))

        def updateGraph(event):
            if updateGraph.i >= iterations:
                return False

            print((updateGraph.i))

            matplotlib.pyplot.clf()
            #print(positionsDict)
            networkx.draw_networkx(graph, pos=allPositions[updateGraph.i])
            #matplotlib.pyplot.axis([-10, 10, -100, 100])

            fig.canvas.draw()
            updateGraph.i = updateGraph.i + 1

        updateGraph.i = 0

        id = wx.NewId()
        actor = fig.canvas.manager.frame
        timer = wx.Timer(actor, id=id)
        timer.Start(50)
        wx.EVT_TIMER(actor, id, updateGraph)

        matplotlib.pyplot.show()        
        
    def testProfileLayout(self):
        try:
            import networkx
        except ImportError:
            pass 

        numpy.random.seed(21)
        graph = networkx.erdos_renyi_graph(50, 0.1, seed=21)

        layoutEngine = GraphLayoutEngine()
        profileFileName = "profile.cprof"

        cProfile.runctx('layoutEngine.layout(graph)', globals(), locals(), profileFileName)
        stats = pstats.Stats(profileFileName)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)
        #stats.sort_stats('cumulative').print_callers(30)


if __name__ == '__main__':
    unittest.main()

