import os
import time
from apgl.util.PathDefaults import PathDefaults 



while True: 
    filename = PathDefaults.getOutputDir() + "deleteMe"
    file = open(filename, "w")
    
    file.write("This is a test")
    file.close() 
    print("Wrote to file " + filename)
    
    time.sleep(10)
        
    
    for param in os.environ.keys():
        print "%20s %s" % (param,os.environ[param])   
    

