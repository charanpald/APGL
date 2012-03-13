
"""
Profile the advanceGraph method of the EgoSimulator 
"""

numVertices = 1000
p1 = 0.1
vList = self.egoGenerator.generateIndicatorVertices(numVertices, self.means, self.vars, p1)
sGraph = SparseGraph(vList)

p2 = 0.1
k = 5
edgeWeight = 1

#Create the graph edges according to the small world model
graphGen = SmallWorldGenerator(sGraph)
sGraph = graphGen.generateGraph(p2, k, edgeWeight)

egoSimulator = EgoSimulator(sGraph, self.nb)

profileFileName = "profile.cprof"
cProfile.runctx('egoSimulator.advanceGraph()', globals(), locals(), profileFileName)
stats = pstats.Stats(profileFileName)
stats.strip_dirs().sort_stats("cumulative").print_stats(20)
