
import numpy 
import scipy.sparse.linalg 

class EfficientNystrom(object): 
    def __init__(self): 
        """
        Implements the eigenvector computations of "Time and Space Efficient Spectral
        Clustering via Column Sampling" 
        """        
        pass 
    
    @staticmethod 
    def eigWeight(W, m, k): 
        """
        Find the k largest eigenvectors and eigenvalues of M = D^-1/2 W D^-1/2 using a weight 
        matrix. This is the same as I - L where L is the normalised Laplacian. 
        
        :param W: A sparse weight matrix 
        
        :param m: The number of columns to sample 
        
        :param k: The number of eigenvectors/eigenvalues to find. 
        """
        if isinstance(m, int):
            inds = numpy.sort(numpy.random.permutation(W.shape[0])[0:m])
        else:
            inds = m      
        
        W11 = W[:, inds][inds, :]
        dStar = numpy.array(W11.sum(0)).ravel()
        dStar[dStar!=0] = dStar[dStar!=0]**-0.5
        DStar = scipy.sparse.spdiags(dStar, 0, dStar.shape[0], dStar.shape[0], format='csr')
        
        MStar = DStar.dot(W11).dot(DStar)
        
        lmbda, V = scipy.sparse.linalg.eigsh(MStar, min(k, MStar.shape[0]-1), which="LM", ncv = min(10*k, MStar.shape[0]))
        print(lmbda)
        
        InvLmbda = scipy.sparse.spdiags(lmbda**-1, 0, k, k, format='csr') 
        V = scipy.sparse.csr_matrix(V)
        B = DStar.dot(V).dot(InvLmbda)
        
        Q = W[:, inds].dot(B)
        print(Q.sum(0).transpose().shape)
        dHat = numpy.array((Q.dot(InvLmbda).dot(Q.sum(0).transpose()))).ravel()
        dHat2 = numpy.array(Q.dot(InvLmbda).dot(Q.T).todense())
   
        print(dHat) 
        print(dHat2, dHat2.shape)  
             
        DHat = scipy.sparse.spdiags(dHat**-0.5, 0, dHat.shape[0], dHat.shape[0], format='csr')
                
             
                
        U = DHat.dot(Q)
        
        return lmbda, U 
        
        