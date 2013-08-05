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
        
        resultsDir += "arnetminer/"
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.trainExpertMatchesFilename = resultsDir + "experts_train_matches.csv"
        self.testExpertMatchesFilename = resultsDir + "experts_test_matches.csv"
        self.coauthorsFilename = resultsDir + "coauthors.csv"
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
        if not os.path.exists(self.coauthorsFilename): 
            matchedExpertsFile = open(self.trainExpertMatchesFilename)
            matchedExperts = matchedExpertsFile.readlines()
            matchedExperts = set([x.strip() for x in matchedExperts])
            
            inFile = open(self.dataFilename)
            i = 0     
            
            expertCoauthors = set([])
            
            for line in inFile:
                Util.printIteration(i, self.stepSize, self.numLines)
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = authors[0].split(",")  
                    authors = [x.strip() for x in authors]     
                    for author in authors: 
                        if author in matchedExperts: 
                            expertCoauthors = expertCoauthors.union(set(authors))
                            
                i += 1
            
            logging.debug("Found " + str(len(expertCoauthors)) + " coauthors")
            
            #Now just write out the coauthors 
            coauthorsFile = open(self.coauthorsFilename, "w")
            expertCoauthors = sorted(list(expertCoauthors))
            
            for coauthor in expertCoauthors: 
                coauthorsFile.write(coauthor + "\n")
            coauthorsFile.close()
            
            logging.debug("Wrote coauthors to file " + str(self.coauthorsFilename))
        else: 
            logging.debug("File already generated: " + self.coauthorsFilename)  
    
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
            
        coauthorMatrix = numpy.ones((len(coauthors), len(coauthors)))
        
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
                
            print(coauthorsList)
            vectoriser = sklearn.feature_extraction.text.TfidfVectorizer(min_df=2, ngram_range=(1,2), binary=True, sublinear_tf=True, norm="l2")
            X = vectoriser.fit_transform(titleAbstracts)
            K = numpy.array((X.dot(X.T)).todense())
            
            
            #Save matrix 
            numpy.save(self.coauthorSimilarityFilename, K) 
            logging.debug("Wrote coauthors to file " + str(self.coauthorSimilarityFilename) + ".npy")
        else: 
            logging.debug("File already generated: " + self.coauthorSimilarityFilename + ".npy")  
            