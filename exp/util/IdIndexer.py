import array 
import numpy 

class IdIndexer(object): 
    def __init__(self, arrayType="i"): 
        self.inds = array.array(arrayType)
        
        self.idSet = set([])
        self.idDict = {}
        self.p = 0 
        
    def append(self, id): 
        if id not in self.idSet: 
            self.idSet.add(id)
            self.idDict[id] = self.p
            ind = self.p 
            self.p += 1 
        else: 
            ind = self.idDict[id]   
            
        self.inds.append(ind)
        return ind 

    def translate(self, id): 
        """
        Take the ID and translate it into a index without adding to the array. 
        """
        if id not in self.idSet: 
            self.idSet.add(id)
            self.idDict[id] = self.p
            ind = self.p 
            self.p += 1 
        else: 
            ind = self.idDict[id]   
            
        return ind         

    def getArray(self): 
        return numpy.array(self.inds)
        
    def getIdDict(self): 
        return self.idDict