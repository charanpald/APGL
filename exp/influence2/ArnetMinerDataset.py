import os 
import numpy 
import logging 
import difflib 
import re 
import sklearn.feature_extraction.text as text 
import gc
import scipy.sparse
import scipy.io
import string 
import array
import pickle 
import Stemmer
from apgl.util.Util import Util 
from apgl.util.PathDefaults import PathDefaults 
from exp.sandbox.RandomisedSVD import RandomisedSVD
from exp.util.IdIndexer import IdIndexer


#Tokenise the documents                 
class PorterTokeniser(object):
    def __init__(self):
        self.stemmer = Stemmer.Stemmer('english')
        self.minWordLength = 2
     
    def __call__(self, doc):
        doc = doc.lower()
        tokens =  [self.stemmer.stemWord(t) for t in doc.split()]  
        return [token for token in tokens if len(token) >= self.minWordLength]

class ArnetMinerDataset(object): 
    """
    We process the ArnetMinerDataset into two graphs - a coauthor and an 
    abstract similarity. The output is two graphs - collaboration and 
    abstract similarity. 
    """    
    def __init__(self, field):
        numpy.random.seed(21)
        dataDir = PathDefaults.getDataDir() + "dblpCitation/" 
        
        self.field = field 
        #self.dataFilename = dataDir + "DBLP-citation-Feb21.txt" 
        self.dataFilename = dataDir + "DBLP-citation-1000000.txt" 
        
        baseDir = PathDefaults.getDataDir() + "reputation/"
        resultsDir = baseDir + field + "/"
        
        self.expertsFileName = resultsDir + "experts.txt"
        self.docTermMatrixSVDFilename = baseDir + "termDocMatrixSVD.npz"
        self.authorListFilename = baseDir + "authorList.pkl"
        self.vectoriserFilename = baseDir + "vectoriser.pkl"        
        
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.trainExpertMatchesFilename = resultsDir + "experts_train_matches.csv"
        self.testExpertMatchesFilename = resultsDir + "experts_test_matches.csv"
        self.coauthorsFilename = resultsDir + "coauthors.csv"
        self.similarAuthorsFilename = resultsDir + "similarAuthors.npy"   
        
        self.relevantExpertsFilename = resultsDir + "relevantExperts.pkl"          

        self.stepSize = 100000    
        self.numLines = 15192085
        self.matchCutoff = 0.95
        self.p = 0.5
        #self.p = 1      
        
        self.similarityCutoff = 0.4 
        self.k = 100
        self.numFeatures = None
        
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
        """
        Using the relevant authors we find all coauthors. 
        """
        if not os.path.exists(self.coauthorsFilename): 
            logging.debug("Finding coauthors of relevant experts")
            
            relevantExpertsFile = open(self.relevantExpertsFilename, "w") 
            relevantExperts = pickle.load(relevantExpertsFile)
            relevantExpertsFile.close()
            
            dataFile = open(self.dataFilename)  
            authorIndexer = IdIndexer()
            author1Inds = array.array("i")
            author2Inds = array.array("i")
            expertCoauthors1 = set([])
            
            for i, line in enumerate(dataFile):
                Util.printIteration(i, self.stepSize, self.numLines)
                authors = re.findall("#@(.*)", line)  
                                
                if len(authors) != 0: 
                    authors = [x.strip() for x in authors[0].split(",")]     
                    for author in authors: 
                        if author in relevantExperts: 
                            expertCoauthors1 = expertCoauthors1.union(set(authors))
            
            logging.debug("Found " + str(len(expertCoauthors1)) + " coauthors at level 1")
            dataFile.close()
                                   
            #Now just write out the coauthors 
            coauthorsFile = open(self.coauthorsFilename, "w")
            expertCoauthors = sorted([coauthor + "\n" for coauthor in expertCoauthors1]) 
            coauthorsFile.writelines(expertCoauthors)
            coauthorsFile.close()
            logging.debug("Wrote coauthors to file " + str(self.coauthorsFilename))
        else: 
            logging.debug("Files already generated: " + self.coauthorsFilename)  
    
    def vectoriseDocuments(self):
        """
        We want to go through the dataset and vectorise all the title+abstracts.
        """
        if not os.path.exists(self.docTermMatrixSVDFilename) or not os.path.exists(self.authorListFilename):
            #We load all title+abstracts 
            inFile = open(self.dataFilename)  
            authorList = []
            documentList = []
                        
            lastAbstract = ""
            lastTitle = ""    
            lastAuthors = []                    
                        
            for i, line in enumerate(inFile):
                Util.printIteration(i, self.stepSize, self.numLines)
                    
                #Match the fields in the file 
                emptyLine = line == "\n"
                title = re.findall("#\*(.*)", line)
                currentAuthors = re.findall("#@(.*)", line)  
                abstract = re.findall("#!(.*)", line)
                
                if emptyLine:
                    document = lastTitle + " " + lastAbstract 
                    documentList.append(document.translate(string.maketrans("",""), string.punctuation)) 
                    authorList.append(lastAuthors)
    
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
                    lastAuthors = currentAuthors                     
    
            inFile.close() 
            logging.debug("Finished reading file")    
            
            authorListFile = open(self.authorListFilename, "w")
            pickle.dump(authorList, authorListFile) 
            authorListFile.close()
            del authorListFile
            logging.debug("Wrote to file " + self.authorListFilename)            
            
            vectoriser = text.TfidfVectorizer(min_df=2, ngram_range=(1,2), binary=False, sublinear_tf=True, norm="l2", max_df=0.95, stop_words="english", tokenizer=PorterTokeniser(), max_features=self.numFeatures)
            X = vectoriser.fit_transform(documentList)
            logging.debug("Finished vectorising documents")
            
            print(X[:, vectoriser.get_feature_names().index("boost")])
                
            #Save vectoriser - note that we can't pickle the tokeniser so it needs to be reset when loaded 
            vectoriser.tokenizer = None 
            vectoriserFile = open(self.vectoriserFilename, "w")
            pickle.dump(vectoriser, vectoriserFile)
            vectoriserFile.close()
            logging.debug("Wrote vectoriser to file " + self.vectoriserFilename)    
            del vectoriser  
            gc.collect()
                
            #Take the SVD of X (maybe better to use PROPACK here depending on size of X)
            logging.debug("Computing the SVD of the document-term matrix")
            X = X.tocsc()
            U, s, V = RandomisedSVD.svd(X, self.k)
            
            numpy.savez(self.docTermMatrixSVDFilename, U, s, V)
            logging.debug("Wrote to file " + self.docTermMatrixSVDFilename)
        else: 
            logging.debug("Files already generated: " + self.docTermMatrixSVDFilename + " " + self.authorListFilename)   
            
    def findSimilarDocuments(self): 
        """
        Find all documents within the same field. 
        """
        if not os.path.exists(self.relevantExpertsFilename): 
            #First load all the components 
            vectoriserFile = open(self.vectoriserFilename)
            vectoriser = pickle.load(vectoriserFile)
            vectoriser.tokenizer = PorterTokeniser() 
            vectoriserFile.close() 
            
            authorListFile = open(self.authorListFilename)
            authorList = pickle.load(authorListFile)
            authorListFile.close()     
            
            data = numpy.load(self.docTermMatrixSVDFilename)    
            U, s, V = data["arr_0"], data["arr_1"], data["arr_2"] 
            
            #Normalised rows of U 
            normU = numpy.sqrt((U**2).sum(1))
            invNormU = 1/(normU + numpy.array(normU==0, numpy.int))
            U = (U.T*invNormU).T
        
            #newX = vectoriser.transform(["java"])
            newX = vectoriser.transform([self.field])
            if newX.nnz == 0: 
                raise ValueError("Query term not found") 
            
            newU = newX.dot(V*(1/s)).T
            newU = newU/numpy.linalg.norm(newU)
            similarities = U.dot(newU).ravel()
            
            relevantDocs = numpy.arange(similarities.shape[0])[similarities >= self.similarityCutoff]  
                    
            #Now find all authors corresponding to the documents 
            experts = [] 
            for docInd in relevantDocs: 
                experts.extend(authorList[docInd])
                
            experts = set(experts)
            logging.debug("Authors: " + str(experts))
            logging.debug("Number of authors : " + str(len(experts)))
            
            relevantExpertsFile = open(self.relevantExpertsFilename, "w") 
            pickle.dump(experts, relevantExpertsFile)
            relevantExpertsFile.close()
            logging.debug("Saved experts in file " + self.relevantExpertsFilename)
        else: 
            logging.debug("File already generated " + self.relevantExpertsFilename)
        
        