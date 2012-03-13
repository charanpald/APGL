
   def testProfileWriteToFile(self):
        numVertices = 100
        numFeatures = 10
        vList = VertexList(numVertices, numFeatures)
        sGraph = SparseGraph(vList, True)
        pw = PajekWriter()

        p = 0.1 #The re-wiring probability
        k = 15 #default number of neighbours for each vertex
        edgeWeight = 1 #The weight on each edge (currently has no meaning)

        #Create the graph edges according to the small world model
        graphGen = SmallWorldGenerator(sGraph)
        sGraph = graphGen.generateGraph(p, k, edgeWeight)

        directory = "output/test/"
        fileName = directory + "tempGraph.net"
        profileFileName = "profile.cprof"

        cProfile.runctx('pw.writeToFile(fileName, sGraph)', globals(), locals(), profileFileName)
        stats = pstats.Stats(profileFileName)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)

        os.remove(fileName + ".net")
        os.remove(profileFileName)