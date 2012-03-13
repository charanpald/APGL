

def testGenerateIndicatorVertices2Profile(self):
    profileFileName = "profile.cprof"

    egoGenerator = EgoGenerator()

    numVertices = 1000

    eCsvReader = EgoCsvReader()
    egoFileName = "data/EgoData.csv"

    self.egoQuestionIds = eCsvReader.getEgoQuestionIds()
    (X, _) = eCsvReader.readFile(egoFileName, self.egoQuestionIds)

    (mu, sigmaSq) = Util.computeMeanVar(X)

    p = 0.1

    cProfile.runctx('egoGenerator.generateIndicatorVertices2(numVertices, mu, sigmaSq, p, X.min(0), X.max(0))', globals(), locals(), profileFileName)
    stats = pstats.Stats(profileFileName)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)