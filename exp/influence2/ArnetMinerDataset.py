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
import gensim.matutils
import sys 
import scipy.io
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
import gensim.similarities
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
        #self.dataFilename = self.dataDir + "DBLP-citation-Feb21.txt" 
        self.dataFilename = self.dataDir + "DBLP-citation-7000000.txt" 
        #self.dataFilename = self.dataDir + "DBLP-citation-100000.txt"
        
        baseDir = PathDefaults.getDataDir() + "reputation/"
        resultsDir = baseDir + field.replace(' ', '') + "/"
                
        self.docTermMatrixSVDFilename = baseDir + "termDocMatrixSVD.npz"
        self.docTermMatrixFilename = baseDir + "termDocMatrix"
        self.authorListFilename = baseDir + "authorList.pkl"
        self.vectoriserFilename = baseDir + "vectoriser.pkl"        
        self.ldaModelFilename = baseDir + "ldaModel.pkl"
        self.lsiModelFilename = baseDir + "lsiModel.pkl"
        
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
        self.minDf = 2 
        self.ngram = 2
        
        #params for RSVD        
        self.k = k
        self.q = 3
        self.p = 30 
        
        self.overwrite = False
        self.overwriteVectoriser = False
        self.overwriteModel = False
        
        self.chunksize = 5000
        self.tfidf = True 
        
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
        
        if not os.path.exists(self.coauthorsFilename) or self.overwrite: 
            logging.debug("Finding coauthors of relevant experts")
            graph, authorIndexer = self.coauthorsGraphFromAuthors(set(relevantExperts))
            logging.debug(graph.summary())
            Util.savePickle([graph, authorIndexer], self.coauthorsFilename, debug=True)
        else: 
            logging.debug("Files already generated: " + self.coauthorsFilename)  

        graph, authorIndexer = Util.loadPickle(self.coauthorsFilename)
        return graph, authorIndexer, relevantExperts 

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

    def vectoriseDocuments(self):
        """
        We want to go through the dataset and vectorise all the title+abstracts.
        The results are saved in TDIDF format in a matrix X. 
        """
        if not os.path.exists(self.docTermMatrixFilename + ".mtx") or not os.path.exists(self.authorListFilename) or not os.path.exists(self.vectoriserFilename) or self.overwriteVectoriser:
            logging.debug("Vectorising documents")            
            
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
            
            Util.savePickle(authorList, self.authorListFilename, debug=True)
            del authorList
            
            #vectoriser = text.HashingVectorizer(ngram_range=(1,2), binary=self.binary, norm="l2", stop_words="english", tokenizer=PorterTokeniser(), dtype=numpy.float)
            
            if self.tfidf:             
                vectoriser = text.TfidfVectorizer(min_df=self.minDf, ngram_range=(1,self.ngram), binary=self.binary, sublinear_tf=self.sublinearTf, norm="l2", max_df=0.95, stop_words="english", tokenizer=PorterTokeniser(), max_features=self.numFeatures, dtype=numpy.float)
            else: 
                vectoriser = text.CountVectorizer(min_df=self.minDf, ngram_range=(1,self.ngram), binary=self.binary, max_df=0.95, stop_words="english", max_features=self.numFeatures, dtype=numpy.float, tokenizer=PorterTokeniser())            
            
            X = vectoriser.fit_transform(documentList)
            del documentList
            scipy.io.mmwrite(self.docTermMatrixFilename, X)
            logging.debug("Wrote X to file " + self.docTermMatrixFilename + ".mtx")
            del X 
                
            #Save vectoriser - note that we can't pickle the tokeniser so it needs to be reset when loaded 
            vectoriser.tokenizer = None 
            Util.savePickle(vectoriser, self.vectoriserFilename, debug=True) 
            del vectoriser  
            gc.collect()
        else: 
            logging.debug("Files already generated: " + self.docTermMatrixSVDFilename + " " + self.authorListFilename)   
   
    def loadVectoriser(self): 
        self.vectoriser = Util.loadPickle(self.vectoriserFilename)
        self.vectoriser.tokenizer = PorterTokeniser() 
        self.authorList = Util.loadPickle(self.authorListFilename)     
        logging.debug("Loaded vectoriser and author list")
           
    def computeLSI(self): 
        if not os.path.exists(self.docTermMatrixSVDFilename) or self.overwriteModel:
            self.vectoriseDocuments()
            #Take the SVD of X (maybe better to use PROPACK here depending on size of X)
            X = scipy.io.mmread(self.docTermMatrixFilename)
            X = X.tocsc()
            logging.debug("Computing the SVD of the document-term matrix of shape " + str(X.shape) + " with " + str(X.nnz) + " non zeros")
            U, s, V = RandomisedSVD.svd(X, self.k, q=self.q, p=self.p)
            del X 
            gc.collect()
            
            numpy.savez(self.docTermMatrixSVDFilename, U, s, V)
            logging.debug("Wrote to file " + self.docTermMatrixSVDFilename)
        else: 
            logging.debug("File already generated: " + self.docTermMatrixSVDFilename)
        
    def loadLSI(self): 
        data = numpy.load(self.docTermMatrixSVDFilename)    
        self.U, self.s, self.V = data["arr_0"], data["arr_1"], data["arr_2"]
        logging.debug("Loaded SVD")
      
    def findSimilarDocumentsLSI(self): 
        """
        Find all documents within the same field using Latent Semantic Indexing
        and the cosine similarity metric. 
        """
        if not os.path.exists(self.relevantExpertsFilename) or self.overwrite: 
            self.computeLSI()
            self.loadVectoriser()
            self.loadLSI()

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
            
            Util.savePickle(experts, self.relevantExpertsFilename, debug=True)
            logging.debug("Saved experts in file " + self.relevantExpertsFilename)
        else: 
            logging.debug("File already generated " + self.relevantExpertsFilename)


    def computeLDA(self):
        if not os.path.exists(self.ldaModelFilename) or self.overwriteModel:
            self.vectoriseDocuments()
            self.loadVectoriser()
            X = scipy.io.mmread(self.docTermMatrixFilename)
            #corpus = gensim.matutils.MmReader(self.docTermMatrixFilename + ".mtx", True)
            corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
            del X 
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))   
            
            logging.getLogger('gensim').setLevel(logging.INFO)
            lda = LdaModel(corpus, num_topics=self.k, id2word=id2WordDict, chunksize=self.chunksize, distributed=False) 
            index = gensim.similarities.docsim.SparseMatrixSimilarity(lda[corpus], num_features=len(self.vectoriser.get_feature_names()))             
            
            Util.savePickle([lda, index], self.ldaModelFilename, debug=True)
            gc.collect()
        else: 
            logging.debug("File already exists: " + self.ldaModelFilename)
   
    def findSimilarDocumentsLDA(self): 
        """ 
        We use LDA in this case 
        """
        if not os.path.exists(self.relevantExpertsFilename) or self.overwrite: 
            self.computeLDA()
            self.loadVectoriser()
                                
            lda, index = Util.loadPickle(self.ldaModelFilename)
            
            newX = self.vectoriser.transform([self.field])
            #newX = self.vectoriser.transform(["database"])
            newX = [(i, newX[0, i])for i in newX.nonzero()[1]]
            result = lda[newX]             
            similarities = index[result]
            experts = self.expertsFromDocSimilarities(similarities)
            
            logging.debug("Number of relevant authors : " + str(len(experts)))
            Util.savePickle(experts, self.relevantExpertsFilename, debug=True)
        else: 
            logging.debug("File already generated " + self.relevantExpertsFilename)

    def computeLSI2(self):
        """
        Compute using the LSI version in gensim 
        """
        if not os.path.exists(self.ldaModelFilename) or self.overwriteModel:
            self.vectoriseDocuments()
            self.loadVectoriser()
            X = scipy.io.mmread(self.docTermMatrixFilename)
            #corpus = gensim.matutils.MmReader(self.docTermMatrixFilename + ".mtx", True)
            corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
            del X 
            id2WordDict = dict(zip(range(len(self.vectoriser.get_feature_names())), self.vectoriser.get_feature_names()))   
            
            logging.getLogger('gensim').setLevel(logging.INFO)
            lsi = LsiModel(corpus, num_topics=self.k, id2word=id2WordDict, chunksize=self.chunksize, distributed=False) 
            index = gensim.similarities.docsim.SparseMatrixSimilarity(lsi[corpus], num_features=len(self.vectoriser.get_feature_names()))             
            
            Util.savePickle([lsi, index], self.lsiModelFilename, debug=True)
            gc.collect()
        else: 
            logging.debug("File already exists: " + self.ldaModelFilename)   


    def findSimilarDocumentsLSI2(self): 
        """ 
        We use LSI from gensim in this case 
        """
        if not os.path.exists(self.relevantExpertsFilename) or self.overwrite: 
            self.computeLSI2()
            self.loadVectoriser()
                                
            lsi, index = Util.loadPickle(self.lsiModelFilename)
            
            newX = self.vectoriser.transform([self.field])
            #newX = self.vectoriser.transform(["database"])
            newX = [(i, newX[0, i])for i in newX.nonzero()[1]]
            result = lsi[newX]             
            similarities = index[result]
            experts = self.expertsFromDocSimilarities(similarities)
            
            logging.debug("Number of relevant authors : " + str(len(experts)))
            Util.savePickle(experts, self.relevantExpertsFilename, debug=True)
        else: 
            logging.debug("File already generated " + self.relevantExpertsFilename)            