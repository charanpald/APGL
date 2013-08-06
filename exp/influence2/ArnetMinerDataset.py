import os 
import numpy 
import logging 
import difflib 
import re 
import sklearn.feature_extraction.text 
from apgl.util.Util import Util 
from apgl.util.PathDefaults import PathDefaults 

class ArnetMinerDataset(object): 
    """
    We process the ArnetMinerDataset into two graphs - a coauthor and an 
    abstract similarity. The output is two graphs - collaboration and 
    abstract similarity. 
    """    
    def __init__(self, field):
        numpy.random.seed(21)
        dataDir = PathDefaults.getDataDir() + "dblpCitation/" 
        self.dataFilename = dataDir + "DBLP-citation-Feb21.txt" 
        #self.dataFilename = dataDir + "DBLP-citation-small.txt" 
        
        resultsDir = PathDefaults.getDataDir() + "reputation/" + field + "/"
        self.expertsFileName = resultsDir + "experts.txt"
        
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.trainExpertMatchesFilename = resultsDir + "experts_train_matches.csv"
        self.testExpertMatchesFilename = resultsDir + "experts_test_matches.csv"
        self.coauthorsFilenameL1 = resultsDir + "coauthorsL1.csv"
        self.coauthorsFilenameL2 = resultsDir + "coauthorsL2.csv"
        self.coauthorSimilarityFilename = resultsDir + "coauthorSimilarity"        
        
        self.stepSize = 100000    
        self.numLines = 15192085
        self.matchCutoff = 0.95
        self.p = 0.5
        #self.p = 1        
        
        self.matchExperts()
        self.splitExperts()   
        self.writeCoauthors()
        self.writeSimilarityGraph()
        
    def matchExperts(self): 
        expertsFile = open(self.expertsFileName)
        expertsSet = expertsFile.readlines()
        expertsSet = set([x.strip() for x in expertsSet])
        
        if not os.path.exists(self.expertMatchesFilename): 
            inFile = open(self.dataFilename)    
            expertMatches = set([])
            i = 0 
            
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                if i % self.stepSize == 0: 
                    logging.debug(expertMatches)
                    
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = authors[0].split(",")                        
                    for author in authors: 
                        possibleMatches = difflib.get_close_matches(author, expertsSet, cutoff=self.matchCutoff)
                        if len(possibleMatches) != 0: 
                            expertMatches.add(author)
                            if author == possibleMatches[0]: 
                                expertsSet.remove(possibleMatches[0])
                            
                            if len(expertsSet) == 0: 
                                logging.debug("Found all experts, breaking")
                                break  
                i += 1
            
            expertMatches = sorted(list(expertMatches))
            expertMatchesFile = open(self.expertMatchesFilename, "w")
            
            for expert in expertMatches: 
                expertMatchesFile.write(expert + "\n")
            expertMatchesFile.close()
            
            logging.debug("All done")
        else: 
            logging.debug("File already generated: " + self.expertMatchesFilename)
        
    def splitExperts(self): 
        if not (os.path.exists(self.trainExpertMatchesFilename) and os.path.exists(self.testExpertMatchesFilename)): 
            file = open(self.expertMatchesFilename) 
            logging.debug("Splitting list of experts given in file " +  self.expertMatchesFilename)
            names = file.readlines()
            names = numpy.array([x.strip() for x in names])
            file.close() 
            
            inds = numpy.random.permutation((len(names)))
            trainSize = int(self.p * len(names))
            
            trainNames = names[inds[0:trainSize]]
            testNames = names[inds[trainSize:]]
            
            logging.debug("Train names: " + str(len(trainNames)) + " test names: " + str(len(testNames))) 
            
            trainNames = numpy.array([name + "\n" for name in trainNames])  
            testNames = numpy.array([name + "\n" for name in testNames])
                
            trainFile = open(self.trainExpertMatchesFilename, "w") 
            trainFile.writelines(trainNames) 
            trainFile.close() 
            
            testFile = open(self.testExpertMatchesFilename, "w") 
            testFile.writelines(testNames) 
            testFile.close() 
            
            logging.debug("Wrote train and test names to " +  self.trainExpertMatchesFilename + " and " + self.testExpertMatchesFilename) 
        else: 
            logging.debug("Generated files exist: " + self.trainExpertMatchesFilename + " " + self.testExpertMatchesFilename)
            
    def writeCoauthors(self): 
        if not os.path.exists(self.coauthorsFilenameL1) and not not os.path.exists(self.coauthorsFilenameL2): 
            logging.debug("Finding coauthors of path length <= 2 from experts")
            matchedExpertsFile = open(self.trainExpertMatchesFilename)
            matchedExperts = matchedExpertsFile.readlines()
            matchedExperts = set([x.strip() for x in matchedExperts])
            
            inFile = open(self.dataFilename)  
            expertCoauthors1 = set([])
            
            for i, line in enumerate(inFile):
                Util.printIteration(i, self.stepSize, self.numLines)
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = [x.strip() for x in authors[0].split(",")]     
                    for author in authors: 
                        if author in matchedExperts: 
                            expertCoauthors1 = expertCoauthors1.union(set(authors))
            
            logging.debug("Found " + str(len(expertCoauthors1)) + " coauthors at level 1")
            inFile.close()
            
            coauthorsFile = open(self.coauthorsFilenameL1, "w")
            expertCoauthors = sorted([coauthor + "\n" for coauthor in expertCoauthors1]) 
            coauthorsFile.writelines(expertCoauthors)
            coauthorsFile.close()
            logging.debug("Wrote coauthors to file " + str(self.coauthorsFilenameL1))
            
            #Now find their coauthors
            inFile = open(self.dataFilename)  
            expertCoauthors2 = set([])
            
            for i, line in enumerate(inFile):
                Util.printIteration(i, self.stepSize, self.numLines)
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = [x.strip() for x in authors[0].split(",")]     
                    for author in authors: 
                        if author in expertCoauthors1: 
                            expertCoauthors2 = expertCoauthors2.union(set(authors))
            
            logging.debug("Found " + str(len(expertCoauthors2)) + " coauthors at level 2")
            inFile.close()            
            
            #Now just write out the coauthors 
            coauthorsFile = open(self.coauthorsFilenameL2, "w")
            expertCoauthors = sorted([coauthor + "\n" for coauthor in expertCoauthors2]) 
            coauthorsFile.writelines(expertCoauthors)
            coauthorsFile.close()
            logging.debug("Wrote coauthors to file " + str(self.coauthorsFilenameL2))
        else: 
            logging.debug("Files already generated: " + self.coauthorsFilenameL1 + " " + self.coauthorsFilenameL2)  
    
    def writeSimilarityGraph(self): 
        """
        Run through the abstracts and compare them from each author. 
        """        
        coauthorsFile = open(self.coauthorsFilename)
        coauthors = coauthorsFile.readlines()
        coauthorsList = [x.strip() for x in coauthors]
        coauthors = set(coauthorsList)
        
        titleAbstracts = []
        for i in range(len(coauthors)): 
            titleAbstracts.append("")   
            
        coauthorMatrix = numpy.ones((len(coauthors), len(coauthors)), numpy.int16)
        
        if not os.path.exists(self.coauthorSimilarityFilename + ".npy"): 
            inFile = open(self.dataFilename)
            i = 0     
                        
            lastAbstract = ""
            lastTitle = ""    
            lastAuthors = []                    
                        
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                    
                #Match the fields in the file 
                emptyLine = line == "\n"
                title = re.findall("#\*(.*)", line)
                currentAuthors = re.findall("#@(.*)", line)  
                abstract = re.findall("#!(.*)", line)
                
                if emptyLine:
                    for author in lastAuthors:
                        ind = coauthorsList.index(author)
                        titleAbstracts[ind] += lastTitle + " " + lastAbstract 
                            
                        for author2 in lastAuthors: 
                            ind2 = coauthorsList.index(author2)
                            coauthorMatrix[ind, ind2] += 1    
                            
                    lastAbstract = ""
                    lastTitle = ""
                    lastAuthors = []
 
                if len(title) != 0 and len(title[0]) != 0: 
                    lastTitle = title[0]               
                
                if len(abstract) != 0 and len(abstract[0]) != 0: 
                    lastAbstract = abstract[0]
                           
                if len(currentAuthors) != 0: 
                    currentAuthors = currentAuthors[0].split(",")  
                    currentAuthors = set([x.strip() for x in currentAuthors])
                    currentAuthors = coauthors.intersection(currentAuthors)   

                    lastAuthors = currentAuthors                     

                i += 1
                
            j = 0                
                
                
            min_df = 2
            ngram_max= 2
            binary = True
            sublinear = True
            norm = "l2"
            numFeatures = 1500
            #Try all param values
            #Try different similarity e.g. distance (this is the case when 2-norm = 1)
            #How to combine coauthor and similarity graphs 
            #l1 norm is useless 
            
            #titleAbstracts = titleAbstracts[0:10000]
            print(len(titleAbstracts))
            vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(min_df=min_df, ngram_range=(1,ngram_max), binary=binary, sublinear_tf=sublinear, norm=norm, max_df=0.95, max_features=numFeatures)
            X = vectoriser.fit_transform(titleAbstracts)
            logging.debug("Generated vectoriser")
            #K = numpy.array((X.dot(X.T)).todense())
            #numpy.save(self.coauthorSimilarityFilename + str(j), K) 
            #logging.debug("Wrote coauthors to file " + str(self.coauthorSimilarityFilename) + str(j) + ".npy")
            #j += 1
            
            clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=100, batch_size=1000, n_init=10)
            clusterer.fit(X)
            
            #Now find all examples in same cluster as experts 
            
            
        else: 
            logging.debug("File already generated: " + self.coauthorSimilarityFilename + ".npy")  
            