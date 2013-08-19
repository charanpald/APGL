import Stemmer
import string

#Tokenise the documents                 
class PorterTokeniser(object):
    def __init__(self):
        self.stemmer = Stemmer.Stemmer('english')
        self.minWordLength = 2
     
    def __call__(self, doc):
        doc = doc.lower().encode('utf-8').strip()
        doc = doc.translate(string.maketrans("",""), string.punctuation).decode("utf-8")
        tokens =  [self.stemmer.stemWord(t) for t in doc.split()]  
        return [token for token in tokens if len(token) >= self.minWordLength]