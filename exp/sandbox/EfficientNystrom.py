
import numpy 
import scipy.sparse.linalg 
from apgl.util import Parameter 

class EfficientNystrom(object): 
    def __init__(self): 
        """
        Implements the eigenvector computations of "Time and Space Efficient Spectral
        Clustering via Column Sampling" 
        """        
        pass 
    
    @staticmethod 
    def eigWeight(W, m, k, orthogonalise=True): 
        """
        Find the k largest eigenvectors and eigenvalues of M = D^-1/2 W D^-1/2 using a weight 
        matrix. This is the same as I - L where L is the normalised Laplacian. 
        
        :param W: A sparse weight matrix 
        
        :param m: The number of columns to sample 
        
        :param k: The number of eigenvectors/eigenvalues to find. 
        
        :param orthogonalise: Whether the orthogonalise the final eigenvectors 
        """
        Parameter.checkInt(k, 1, W.shape[0])   
        #This constraint is due to MStar being rank m and only being able to find m-1 eigenvalues 
        m = min(W.shape[0], m)
        Parameter.checkInt(m, k+1, W.shape[0])
        
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
        
        Lmbda = scipy.sparse.spdiags(lmbda, 0, k, k, format='csr')
        InvLmbda = scipy.sparse.spdiags(lmbda**-1, 0, k, k, format='csr')
        V = scipy.sparse.csr_matrix(V)
        B = DStar.dot(V).dot(InvLmbda)
        
        Q = W[:, inds].dot(B)
        dHat = numpy.array((Q.dot(Lmbda).dot(Q.sum(0).transpose()))).ravel()
        #Note that W[:, inds] may have all zero rows (even when full W doesn't) and hence
        #Q can have zero columns meaning dHat can have zero elements and DHat is no longer valid. 
        #There is no answer to this in the paper 
        
        DHat = scipy.sparse.spdiags(dHat**-0.5, 0, dHat.shape[0], dHat.shape[0], format='csr')
                
        U = DHat.dot(Q)
        U = numpy.asarray(U.todense())
        
        if not orthogonalise: 
            return lmbda, U 
        else: 
            return EfficientNystrom.orthogonalise(lmbda, U) 
    
    @staticmethod 
    def orthogonalise(lmbda, U): 
        """
        Take a set of approximate eigenvalues and non-orthogonal eigenvectors
        given by lmbda, U and make U orthogonal according to Algorithm 3 
        in the paper. 
        """
        P = U.T.dot(U)
        
        sigma, V = numpy.linalg.eigh(P)
        sigma12 = sigma**0.5 
        V2 = V*sigma12
        B = (V2.T*lmbda).dot(V2)
        
        lmbdaTilde, VTilde = numpy.linalg.eigh(B)
        
        inds = numpy.flipud(numpy.argsort(lmbdaTilde)) 
        lmbdaTilde = lmbdaTilde[inds]
        VTilde = VTilde[:, inds]
        
        sigmam12 = sigma**-0.5 
        UTilde = U.dot(V*sigmam12).dot(VTilde)
        
        return lmbdaTilde, UTilde
        