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
        Take the ID or list of IDs and translate it into a index without adding 
        to the array. 
        """
        try: 
            iter(id)
            itemList = []
            for item in id: 
                itemList.append(self.idDict[item]) 
            return itemList 
        except TypeError: 
            return self.idDict[id]
            
    def reverseTranslate(self, ind): 
        """
        Take an index and convert back into the ID
        """
        try: 
            iter(ind)
            itemList = []
            for item in ind: 
                itemList.append(self.idDict.keys()[self.idDict.values()[item]]) 
            return itemList 
        except TypeError: 
            return self.idDict.keys()[self.idDict.values()[ind]]    
        
    def getArray(self): 
        return numpy.array(self.inds)
        
    def getIdDict(self): 
        return self.idDict