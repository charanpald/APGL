import os 
import numpy 
import logging 
import difflib 
import re 
import sklearn.feature_extraction.text as text 
import gc
import scipy.sparse
import igraph
import string 
import array
import pickle 
import itertools 
import Stemmer
from apgl.util.Util import Util 
from apgl.util.PathDefaults import PathDefaults 
from exp.sandbox.RandomisedSVD import RandomisedSVD
from exp.util.IdIndexer import IdIndexer
from collections import Counter, OrderedDict 


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
    def __init__(self, field, k=50):
        numpy.random.seed(21)
        self.dataDir = PathDefaults.getDataDir() + "dblpCitation/" 
        
        self.field = field 
        self.dataFilename = self.dataDir + "DBLP-citation-Feb21.txt" 
        #self.dataFilename = self.dataDir + "DBLP-citation-7000000.txt" 
        #self.dataFilename = self.dataDir + "DBLP-citation-100000.txt"
        
        baseDir = PathDefaults.getDataDir() + "reputation/"
        resultsDir = baseDir + field.replace(' ', '') + "/"
                
        self.docTermMatrixSVDFilename = baseDir + "termDocMatrixSVD.npz"
        self.authorListFilename = baseDir + "authorList.pkl"
        self.vectoriserFilename = baseDir + "vectoriser.pkl"        
        
        self.expertsFileName = resultsDir + "experts.txt"
        self.expertMatchesFilename = resultsDir + "experts_matches.csv"
        self.coauthorsFilename = resultsDir + "coauthors.pkl"
        self.relevantExpertsFilename = resultsDir + "relevantExperts.pkl"          

        self.stepSize = 1000000    
        self.numLines = 15192085
        self.matchCutoff = 0.90   
        
        #Params for finding relevant authors
        self.similarityCutoff = 0.4
        self.maxRelevantAuthors = 500

        #Params for vectoriser 
        self.numFeatures = None
        self.binary = True 
        self.sublinearTf = False
        self.minDf = 3 
        
        #params for RSVD        
        self.k = k
        self.q = 3
        self.p = 20 
        
        self.overwriteRelevantExperts = False
        self.overwriteCoauthors = False
        self.overwriteSVD = False
        
    def matchExperts(self): 
        """
        Match experts in the set of relevant experts. It returns expertMatches 
        which is the name of the expert in the database, and expertSet which is 
        the complete set of experts. 
        """
        expertsFile = open(self.expertsFileName)
        expertsSet = expertsFile.readlines()
        expertsSet = set([x.strip() for x in expertsSet])
        expertsFile.close()
        
        relevantExperts = set(Util.loadPickle(self.relevantExpertsFilename)) 
        expertMatches = set([])
        
        for relevantExpert in relevantExperts: 
            possibleMatches = difflib.get_close_matches(relevantExpert, expertsSet, cutoff=self.matchCutoff)
            if len(possibleMatches) != 0: 
                expertMatches.add(relevantExpert)

        for author in expertsSet.difference(expertMatches):
            possibleMatches = difflib.get_close_matches(author, relevantExperts, cutoff=0.6)
            if len(possibleMatches) != 0: 
                logging.debug("Possible matches for " + author + " : " + str(possibleMatches))
                          
        
        expertMatches = sorted(list(expertMatches))
        logging.debug("Total number of matches " + str(len(expertMatches)) + " of " + str(len(expertsSet)))        
        
        return expertMatches, expertsSet 

    def coauthorsGraphFromAuthors(self, relevantExperts): 
        """
        Take a set of relevant authors and return the graph. 
        """
        dataFile = open(self.dataFilename)  
        authorIndexer = IdIndexer()
        author1Inds = array.array("i")
        author2Inds = array.array("i")
        
        for relevantExpert in relevantExperts: 
            authorIndexer.append(relevantExpert)
        
        for i, line in enumerate(dataFile):
            Util.printIteration(i, self.stepSize, self.numLines)
            authors = re.findall("#@(.*)", line)  
                            
            if len(authors) != 0: 
                authors = set([x.strip() for x in authors[0].split(",")]) 
                if len(authors.intersection(relevantExperts)) != 0: 
                    iterator = itertools.combinations(authors, 2)
                
                    for author1, author2 in iterator: 
                        author1Ind = authorIndexer.append(author1) 
                        author2Ind = authorIndexer.append(author2)
                        
                        author1Inds.append(author1Ind)
                        author2Inds.append(author2Ind)
        
        logging.debug("Found " + str(len(authorIndexer.getIdDict())) + " coauthors")
                               
        #Coauthor graph is undirected 
        author1Inds = numpy.array(author1Inds, numpy.int)
        author2Inds = numpy.array(author2Inds, numpy.int)
        edges = numpy.c_[author1Inds, author2Inds]            
        
        graph = igraph.Graph()
        graph.add_vertices(len(authorIndexer.getIdDict()))
        graph.add_edges(edges)
        graph.es["weight"] = numpy.ones(graph.ecount())
        graph.simplify(combine_edges=sum)   
        graph.es["invWeight"] = 1.0/numpy.array(graph.es["weight"]) 
        
        return graph, authorIndexer
            
    def coauthorsGraph(self): 
        """
        Using the relevant authors we find all coauthors. 
        """ 
        relevantExperts = Util.loadPickle(self.relevantExpertsFilename)     
        
        if not os.path.exists(self.coauthorsFilename) or self.overwriteCoauthors: 
            logging.debug("Finding coauthors of relevant experts")
            graph, authorIndexer = self.coauthorsGraphFromAuthors(set(relevantExperts))
            #print(graph.ecount(), edges.shape)
            #print(graph.count_multiple())
            #print(graph.es["weight"])
            logging.debug(graph.summary())
            
            coauthorsFile = open(self.coauthorsFilename, "w")
            pickle.dump([graph, authorIndexer], coauthorsFile)
            coauthorsFile.close()
            
            logging.debug("Coauthors graph saved as " + self.coauthorsFilename)
        else: 
            logging.debug("Files already generated: " + self.coauthorsFilename)  

        coauthorsFile = open(self.coauthorsFilename)
        graph, authorIndexer = pickle.load(coauthorsFile)
        coauthorsFile.close()
            
        return graph, authorIndexer, relevantExperts 
    
    def vectoriseDocuments(self):
        """
        We want to go through the dataset and vectorise all the title+abstracts.
        """
        if not os.path.exists(self.docTermMatrixSVDFilename) or not os.path.exists(self.authorListFilename) or self.overwriteSVD:
            logging.debug("Vectorising documents")            
            
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
            
            Util.savePickle(authorList, self.authorListFilename)
            del authorList
            logging.debug("Wrote to file " + self.authorListFilename)            
            
            #vectoriser = text.HashingVectorizer(ngram_range=(1,2), binary=self.binary, norm="l2", stop_words="english", tokenizer=PorterTokeniser(), dtype=numpy.float)
            vectoriser = text.TfidfVectorizer(min_df=self.minDf, ngram_range=(1,2), binary=self.binary, sublinear_tf=self.sublinearTf, norm="l2", max_df=0.95, stop_words="english", tokenizer=PorterTokeniser(), max_features=self.numFeatures, dtype=numpy.float)
            X = vectoriser.fit_transform(documentList)
            logging.debug("Finished vectorising documents")
                
            #Save vectoriser - note that we can't pickle the tokeniser so it needs to be reset when loaded 
            vectoriser.tokenizer = None 
            Util.savePickle(vectoriser, self.vectoriserFilename)
            logging.debug("Wrote vectoriser to file " + self.vectoriserFilename)    
            del documentList
            del vectoriser  
            gc.collect()
                
            #Take the SVD of X (maybe better to use PROPACK here depending on size of X)
            logging.debug("Computing the SVD of the document-term matrix of shape " + str(X.shape) + " with " + str(X.nnz) + " non zeros")
            X = X.tocsc()
            U, s, V = RandomisedSVD.svd(X, self.k, q=self.q, p=self.p)
            del X 
            gc.collect()
            
            numpy.savez(self.docTermMatrixSVDFilename, U, s, V)
            logging.debug("Wrote to file " + self.docTermMatrixSVDFilename)
        else: 
            logging.debug("Files already generated: " + self.docTermMatrixSVDFilename + " " + self.authorListFilename)   
    
    def loadVectoriser(self): 
        self.vectoriser = Util.loadPickle(self.vectoriserFilename)
        self.vectoriser.tokenizer = PorterTokeniser() 
        
        self.authorList = Util.loadPickle(self.authorListFilename)     
        
        data = numpy.load(self.docTermMatrixSVDFilename)    
        self.U, self.s, self.V = data["arr_0"], data["arr_1"], data["arr_2"] 
        
        logging.debug("Loaded vectoriser, author list and SVD")
        
    def unloadVectoriser(self): 
        del self.vectoriser
        del self.authorList
        del self.U 
        del self.s 
        del self.V
        
        logging.debug("Unloaded vectoriser, author list and SVD")
  
    def expertsFromDocSimilarities(self, similarities): 
        """
        Given a set of similarities work out which documents are relevent 
        and then return a list of ranked authors using these scores. 
        """
        relevantDocs = numpy.arange(similarities.shape[0])[similarities >= self.similarityCutoff]
        
        #Now find all authors corresponding to the documents 
        expertDict = {} 
        expertsSet = set([])
        for docInd in relevantDocs: 
            for author in self.authorList[docInd]: 
                if author not in expertsSet: 
                    expertsSet.add(author)
                    expertDict[author] = similarities[docInd]
                else: 
                    expertDict[author] += similarities[docInd]
        
        expertDict = OrderedDict(sorted(expertDict.items(), key=lambda t: t[1], reverse=True))
        experts = expertDict.keys()[0:self.maxRelevantAuthors]
        
        return experts 
      
    def findSimilarDocuments(self): 
        """
        Find all documents within the same field. Makes a call to loadVectoriser 
        first. 
        """
        if not os.path.exists(self.relevantExpertsFilename) or self.overwriteRelevantExperts: 
            if not hasattr(self, 'vectoriser'): 
                self.loadVectoriser()

            #Normalised rows of U 
            normU = numpy.sqrt((self.U**2).sum(1))
            invNormU = 1/(normU + numpy.array(normU==0, numpy.int))
            U = (self.U.T*invNormU).T
        
            newX = self.vectoriser.transform([self.field])
            if newX.nnz == 0: 
                raise ValueError("Query term not found") 
            
            newU = newX.dot(self.V*(1/self.s)).T
            newU = newU/numpy.linalg.norm(newU)
            similarities = U.dot(newU).ravel()
            
            experts = self.expertsFromDocSimilarities(similarities)
            logging.debug("Number of relevant authors : " + str(len(experts)))
            
            Util.savePickle(experts, self.relevantExpertsFilename)
            logging.debug("Saved experts in file " + self.relevantExpertsFilename)
        else: 
            logging.debug("File already generated " + self.relevantExpertsFilename)
        
        