import gensim
import gensim.matutils
import logging 
import sys 
import numpy 
import sklearn.feature_extraction.text as text 
import scipy.sparse 
from exp.util.PorterTokeniser import PorterTokeniser
from gensim.models.ldamodel import LdaModel
from exp.util.SparseUtils import SparseUtils
from apgl.data.Standardiser import Standardiser

"""
Try to get the right params for Gensim 
"""

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

documentList = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS. user interface management system",
              "System and human system engineering testing of EPS.",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

vectoriser = text.TfidfVectorizer(min_df=2, ngram_range=(1,2), binary=True, sublinear_tf=False, norm="l2", max_df=0.95, stop_words="english", tokenizer=PorterTokeniser())
X = vectoriser.fit_transform(documentList)

print(vectoriser.get_feature_names()) 

corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

id2WordDict = dict(zip(range(len(vectoriser.get_feature_names())), vectoriser.get_feature_names()))   

k = 10
logging.getLogger('gensim').setLevel(logging.INFO)
lda = LdaModel(corpus, num_topics=k, id2word=id2WordDict, chunksize=1000, distributed=False) 
index = gensim.similarities.docsim.SparseMatrixSimilarity(lda[corpus], num_features=k)          

newX = vectoriser.transform(["graph"])
newX = [(i, newX[0, i])for i in newX.nonzero()[1]]
result = lda[newX]             
similarities = index[result]
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
print(similarities)

#Compute Helliger distance 
result = [i[1] for i in result]
newX = scipy.sparse.csc_matrix(result)
distances = SparseUtils.hellingerDistances(index.index, newX)
print(1 - distances)

#Try cosine metric 
X = Standardiser().normaliseArray(numpy.array(index.index.todense()).T).T
newX = numpy.array(newX.todense())
similarities = X.dot(newX.T).flatten()
print(similarities)