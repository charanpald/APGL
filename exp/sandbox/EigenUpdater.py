import numpy
import scipy
import scipy.linalg
import logging
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.util.ProfileUtils import ProfileUtils

class EigenUpdater(object):
    """
    A class to peform certain types of eigen-decomposition updates.
    """
    def __init__(self):
        pass

    @staticmethod
    def eigenConcat(omega, Q, AB, BB, k):
        """
        Find the eigen update of a matrix [A, B]'[A B] where  A'A = V diag(s) V*
        and AB = A*B, BB = B*B. Q is the set of eigenvectors of A*A and s is the
        vector of eigenvalues. 
        """
        logging.debug("< eigenConcat >")
        Parameter.checkInt(k, 0, omega.shape[0])
        if not numpy.isrealobj(omega) or not numpy.isrealobj(Q):
            raise ValueError("Eigenvalues and eigenvectors must be real")
        if not numpy.isrealobj(AB) or not numpy.isrealobj(BB):
            raise ValueError("AB and BB must be real")
        if omega.ndim != 1:
            raise ValueError("omega must be 1-d array")
        if omega.shape[0] != Q.shape[1]:
            raise ValueError("Must have same number of eigenvalues and eigenvectors")
        if Q.shape[0] != AB.shape[0]:
            raise ValueError("Q must have the same number of rows as AB")
        if AB.shape[1] != BB.shape[0] or  BB.shape[0]!=BB.shape[1]:
            raise ValueError("AB must have the same number of cols/rows as BB")

        #Check Q is orthogonal
        if __debug__:
            Parameter.checkOrthogonal(Q, tol=EigenUpdater.tol, softCheck=True, arrayInfo = "input Q in eigenConcat()")

        m = Q.shape[0]
        p = BB.shape[0]

        inds = numpy.flipud(numpy.argsort(numpy.abs(omega)))
        Q = Q[:, inds[0:k]]
        omega = omega[inds[0:k]]
        Omega = numpy.diag(omega)

        QAB = Q.conj().T.dot(AB)

        F = numpy.c_[numpy.r_[Omega, QAB.conj().T], numpy.r_[QAB, BB]]
        D = numpy.c_[numpy.r_[Q, numpy.zeros((p, Q.shape[1]))], numpy.r_[numpy.zeros((m, p)), numpy.eye(p)]]

        pi, H = scipy.linalg.eigh(F)

        inds = numpy.flipud(numpy.argsort(numpy.abs(pi)))
        inds = inds[numpy.abs(pi)>EigenUpdater.tol]

        H = H[:, inds[0:k]]
        pi = pi[inds[0:k]]

        V = numpy.dot(D, H)

        logging.debug("</ eigenConcat >")
        return pi, V

    @staticmethod
    def lazyEigenConcatAsUpdate(omega, Q, AB, BB, k, debug= False):
        """
        Find the eigen update of a matrix [A, B]'[A B] where
        A'A = Q diag(omega) Q* and AB = A*B, BB = B*B. Q is the set of
        eigenvectors of A*A and omega is the vector of eigenvalues.
        
        Simply expand Q, and update the eigen decomposition using EigenAdd2.
        Computation could be upgraded a bit because of the particular update
        type (Y1Bar = Y1 = [0,I]',  Y2Bar = [(I-QQ')A'B, 0]').
        """
        logging.debug("< lazyEigenConcatAsUpdate >")
        Parameter.checkClass(omega, numpy.ndarray)
        Parameter.checkClass(Q, numpy.ndarray)
        Parameter.checkClass(AB, numpy.ndarray)
        Parameter.checkClass(BB, numpy.ndarray)
        Parameter.checkInt(k, 0, AB.shape[0] + BB.shape[0])
        if not numpy.isrealobj(omega) or not numpy.isrealobj(Q):
            logging.info("Eigenvalues or eigenvectors are not real")
        if not numpy.isrealobj(AB) or not numpy.isrealobj(BB):
            logging.info("AB or BB are not real")
        if omega.ndim != 1:
            raise ValueError("omega must be 1-d array")
        if omega.shape[0] != Q.shape[1]:
            raise ValueError("Must have same number of eigenvalues and eigenvectors")
        if Q.shape[0] != AB.shape[0]:
            raise ValueError("Q must have the same number of rows as AB")
        if AB.shape[1] != BB.shape[0] or  BB.shape[0]!=BB.shape[1]:
            raise ValueError("AB must have the same number of cols/rows as BB")

        if __debug__:
            if not Parameter.checkOrthogonal(Q, tol=EigenUpdater.tol, softCheck=True, investigate=True, arrayInfo="input Q in lazyEigenConcatAsUpdate()"):
                print("omega:\n", omega)


        m = Q.shape[0]
        p = BB.shape[0]
        
        Q = numpy.r_[Q, numpy.zeros((p, Q.shape[1]))]
        Y1 = numpy.r_[numpy.zeros((m,p)), numpy.eye(p)]
        Y2 = numpy.r_[AB, 0.5*BB]
        pi, V = EigenUpdater.eigenAdd2(omega, Q, Y1, Y2, k, debug=debug)
        logging.debug("</ lazyEigenConcatAsUpdate >")
        return pi, V

    @staticmethod
    def eigenAdd(omega, Q, Y, k):
        """
        Perform an eigen update of the form A*A + Y*Y in which Y is a low-rank matrix
        and A^*A = Q Omega Q*. We use the rank-k approximation of A:  Q_k Omega_k Q_k^*
        and then approximate [A^*A_k Y^*Y]_k.
        """
        logging.debug("< eigenAdd >")
        Parameter.checkInt(k, 0, omega.shape[0])
        #if not numpy.isrealobj(omega) or not numpy.isrealobj(Q):
        #    raise ValueError("Eigenvalues and eigenvectors must be real")
        if omega.ndim != 1:
            raise ValueError("omega must be 1-d array")
        if omega.shape[0] != Q.shape[1]:
            raise ValueError("Must have same number of eigenvalues and eigenvectors")

        if __debug__:
            Parameter.checkOrthogonal(Q, tol=EigenUpdater.tol, softCheck=True, arrayInfo="input Q in eigenAdd()")

        #Taking the abs of the eigenvalues is correct
        inds = numpy.flipud(numpy.argsort(numpy.abs(omega)))

        omega, Q = Util.indEig(omega, Q, inds[numpy.abs(omega)>EigenUpdater.tol])
        Omega = numpy.diag(omega)

        YY = Y.conj().T.dot(Y)
        QQ = Q.dot(Q.conj().T)
        Ybar = Y - Y.dot(QQ)

        Pbar, sigmaBar, Qbar = numpy.linalg.svd(Ybar, full_matrices=False)
        inds = numpy.flipud(numpy.argsort(numpy.abs(sigmaBar)))
        inds = inds[numpy.abs(sigmaBar)>EigenUpdater.tol]
        Pbar, sigmaBar, Qbar = Util.indSvd(Pbar, sigmaBar, Qbar, inds)
        
        SigmaBar = numpy.diag(sigmaBar)
        Qbar = Ybar.T.dot(Pbar)
        Qbar = Qbar.dot(numpy.diag(numpy.diag(Qbar.T.dot(Qbar))**-0.5))

        r = sigmaBar.shape[0]

        YQ = Y.dot(Q)
        Zeros = numpy.zeros((r, omega.shape[0]))
        D = numpy.c_[Q, Qbar]

        YYQQ = YY.dot(QQ)
        Z = D.conj().T.dot(YYQQ + YYQQ.conj().T).dot(D)
        F = numpy.c_[numpy.r_[Omega - YQ.conj().T.dot(YQ), Zeros], numpy.r_[Zeros.T, SigmaBar.conj().dot(SigmaBar)]]
        F = F + Z 

        pi, H = scipy.linalg.eigh(F)
        inds = numpy.flipud(numpy.argsort(numpy.abs(pi)))

        H = H[:, inds[0:k]]
        pi = pi[inds[0:k]]

        V = D.dot(H)
        logging.debug("</ eigenAdd >")
        return pi, V

    @staticmethod 
    def eigenAdd2(omega, Q, Y1, Y2, k, debug= False):
        """
        Compute an approximation of the eigendecomposition A^*A + Y1Y2^* +Y2Y1^*
        in which Y1, Y2 are low rank, Y1^*Y2=0 and A^*A = Q Omega Q*. We use the
        rank-k approximation of A^*A: Q_k Omega_k Q_k^* and then approximate
        [A^*A_k + Y1Y2^* + Y2Y1^*]_k.
        """
        logging.debug("< eigenAdd2 >")
        Parameter.checkInt(k, 0, float('inf'))
        Parameter.checkClass(omega, numpy.ndarray)
        Parameter.checkClass(Q, numpy.ndarray)
        Parameter.checkClass(Y1, numpy.ndarray)
        Parameter.checkClass(Y2, numpy.ndarray)
        if not numpy.isrealobj(omega) or not numpy.isrealobj(Q):
            logging.warn("Eigenvalues or eigenvectors are not real")
        if not numpy.isrealobj(Y1) or not numpy.isrealobj(Y2):
            logging.warn("Y1 or Y2 are not real")
        if omega.ndim != 1:
            raise ValueError("omega must be 1-d array")
        if omega.shape[0] != Q.shape[1]:
            raise ValueError("Must have same number of eigenvalues and eigenvectors")
        if Q.shape[0] != Y1.shape[0]:
            raise ValueError("Q must have the same number of rows as Y1 rows")
        if Q.shape[0] != Y2.shape[0]:
            raise ValueError("Q must have the same number of rows as Y2 rows")
        if Y1.shape[1] != Y2.shape[1]:
            raise ValueError("Y1 must have the same number of columns as Y2 columns")

        if __debug__:
            Parameter.checkArray(omega, softCheck=True, arrayInfo="omega as input in eigenAdd2()")
            Parameter.checkArray(Q, softCheck=True, arrayInfo="Q as input in eigenAdd2()")
            Parameter.checkOrthogonal(Q, tol=EigenUpdater.tol, softCheck=True, arrayInfo="Q as input in eigenAdd2()")
            Parameter.checkArray(Y1, softCheck=True, arrayInfo="Y1 as input in eigenAdd2()")
            Parameter.checkArray(Y2, softCheck=True, arrayInfo="Y2 as input in eigenAdd2()")
            


        #Get first k eigenvectors/values of A^*A
        omega, Q = Util.indEig(omega, Q, numpy.flipud(numpy.argsort(omega))[0:k])

        QY1 = Q.conj().T.dot(Y1)
        Y1bar = Y1 - Q.dot(QY1)

        P1bar, sigma1Bar, Q1bar = Util.safeSvd(Y1bar)
        inds = numpy.arange(sigma1Bar.shape[0])[numpy.abs(sigma1Bar)>EigenUpdater.tol]
        P1bar, sigma1Bar, Q1bar = Util.indSvd(P1bar, sigma1Bar, Q1bar, inds)
        # checks on SVD decomposition of Y1bar
        if __debug__:
            Parameter.checkArray(QY1, softCheck=True, arrayInfo="QY1 in eigenAdd2()")
            Parameter.checkArray(Y1bar, softCheck=True, arrayInfo="Y1bar in eigenAdd2()")
            Parameter.checkArray(P1bar, softCheck=True, arrayInfo="P1bar in eigenAdd2()")
            if not Parameter.checkOrthogonal(P1bar, tol=EigenUpdater.tol, softCheck=True, arrayInfo="P1bar in eigenAdd2()", investigate=True):
                print ("corresponding sigma: ", sigma1Bar)
            Parameter.checkArray(sigma1Bar, softCheck=True, arrayInfo="sigma1Bar in eigenAdd2()")
            Parameter.checkArray(Q1bar, softCheck=True, arrayInfo="Q1bar in eigenAdd2()")
            if not Parameter.checkOrthogonal(Q1bar, tol=EigenUpdater.tol, softCheck=True, arrayInfo="Q1bar in eigenAdd2()"):
                print ("corresponding sigma: ", sigma1Bar)

        del Y1bar

        P1barY2 = P1bar.conj().T.dot(Y2)
        QY2 = Q.conj().T.dot(Y2)
        Y2bar = Y2 - Q.dot(QY2) - P1bar.dot(P1barY2)
        
        P2bar, sigma2Bar, Q2bar = Util.safeSvd(Y2bar)
        inds = numpy.arange(sigma2Bar.shape[0])[numpy.abs(sigma2Bar)>EigenUpdater.tol]
        P2bar, sigma2Bar, Q2bar = Util.indSvd(P2bar, sigma2Bar, Q2bar, inds)
        # checks on SVD decomposition of Y1bar
        if __debug__:
            Parameter.checkArray(P1barY2, softCheck=True, arrayInfo="P1barY2 in eigenAdd2()")
            Parameter.checkArray(QY2, softCheck=True, arrayInfo="QY2 in eigenAdd2()")
            Parameter.checkArray(Y2bar, softCheck=True, arrayInfo="Y2bar in eigenAdd2()")
            Parameter.checkArray(P2bar, softCheck=True, arrayInfo="P2bar in eigenAdd2()")
            Parameter.checkOrthogonal(P2bar, tol=EigenUpdater.tol, softCheck=True, arrayInfo="P2bar in eigenAdd2()")
            Parameter.checkArray(sigma2Bar, softCheck=True, arrayInfo="sigma2Bar in eigenAdd2()")
            Parameter.checkArray(Q2bar, softCheck=True, arrayInfo="Q2bar in eigenAdd2()")
            Parameter.checkOrthogonal(Q2bar, tol=EigenUpdater.tol, softCheck=True, arrayInfo="Q2bar in eigenAdd2()")

        del Y2bar 

        r = omega.shape[0]
        p = Y1.shape[1]
        p1 = sigma1Bar.shape[0]
        p2 = sigma2Bar.shape[0]

        D = numpy.c_[Q, P1bar, P2bar]
        del P1bar
        del P2bar 
        # rem: A*s = A.dot(diag(s)) ; A*s[:,new] = diag(s).dot(A)
        DStarY1 = numpy.r_[QY1, sigma1Bar[:,numpy.newaxis] * Q1bar.conj().T, numpy.zeros((p2, p))]
        DStarY2 = numpy.r_[QY2, P1barY2, sigma2Bar[:,numpy.newaxis] * Q2bar.conj().T]
        DStarY1Y2StarD = DStarY1.dot(DStarY2.conj().T)

        del DStarY1
        del DStarY2
        
        r = omega.shape[0]
        F = numpy.zeros((r+p1+p2, r+p1+p2))
        F[range(r),range(r)] = omega
        F = F + DStarY1Y2StarD + DStarY1Y2StarD.conj().T

        #A check to make sure DFD^T is AA_k + Y1Y2 + Y2Y1
        #assert numpy.linalg.norm(D.dot(F).dot(D.T) - Q.dot(numpy.diag(omega).dot(Q.T)) - Y1.dot(Y2.T) - Y2.dot(Y1.T)) < 10**-6
        
        # checks on F
        if __debug__:
            #Parameter.checkArray(DStarY1, softCheck=True, arrayInfo="DStarY1 in eigenAdd2()")
            #Parameter.checkArray(DStarY2, softCheck=True, arrayInfo="DStarY2 in eigenAdd2()")
            Parameter.checkArray(DStarY1Y2StarD, softCheck=True, arrayInfo="DStarY1Y2StarD in eigenAdd2()")
            Parameter.checkArray(F, softCheck=True, arrayInfo="F in eigenAdd2()")
            Parameter.checkSymmetric(F, tol=EigenUpdater.tol, softCheck=True, arrayInfo="F in eigenAdd2()")

        pi, H = scipy.linalg.eigh(F)
        # remove too small eigenvalues
        pi, H = Util.indEig(pi, H, numpy.arange(pi.shape[0])[numpy.abs(pi)>EigenUpdater.tol])
        # keep greatest eigenvalues
        pi, H = Util.indEig(pi, H, numpy.flipud(numpy.argsort(pi))[:min(k,pi.shape[0])])


        V = D.dot(H)

        if __debug__:
            if not Parameter.checkOrthogonal(D, tol=EigenUpdater.tol, softCheck=True, investigate=True, arrayInfo="D in eigenAdd2()"):
                print("pi:\n", pi)
            if not Parameter.checkOrthogonal(H, tol=EigenUpdater.tol, softCheck=True, investigate=True, arrayInfo="H in eigenAdd2()"):
                print("pi:\n", pi)

        if ProfileUtils.memory() > 10**9:
            ProfileUtils.memDisplay(locals())
            
        logging.debug("</ eigenAdd2 >")
        if debug:
            return pi, V, D, DStarY1Y2StarD + DStarY1Y2StarD.conj().T
        else:
            return pi, V
        
    @staticmethod 
    def eigenRemove(omega, Q, n, k, debug=False):
        """
        Remove a set of rows and columns from a matrix whose eigen-decomposition
        is Q diag(omega) Q^T. Keep the first n rows/cols i.e. the rows/cols starting
        from n to the end are removed and k is the number of eigenvectors/values
        to return for the new matrix. We could generalise this to delete a given
        list of rows/cols.
        """
        logging.debug("< eigenRemove >")
        Parameter.checkClass(omega, numpy.ndarray)
        Parameter.checkClass(Q, numpy.ndarray)
        Parameter.checkInt(k, 0, float('inf'))
        Parameter.checkInt(n, 0, Q.shape[0])
        if omega.ndim != 1:
            raise ValueError("omega must be 1-d array")
        if omega.shape[0] != Q.shape[1]:
            raise ValueError("Must have same number of eigenvalues and eigenvectors")

        if __debug__:
            Parameter.checkOrthogonal(Q, tol=EigenUpdater.tol, softCheck=True, arrayInfo="input Q in eigenRemove()")

        inds = numpy.flipud(numpy.argsort(numpy.abs(omega)))
        inds = inds[omega[inds]>EigenUpdater.tol]
        
        omega, Q = Util.indEig(omega, Q, inds[0:k])
        AB = (Q[0:n, :]*omega).dot(Q[n:, :].T)
        BB = (Q[n:, :]*omega).dot(Q[n:, :].T)

        p = BB.shape[0]
        Y1 = numpy.r_[numpy.zeros((n, p)), numpy.eye(p)]
        Y2 = -numpy.r_[AB, 0.5*BB]
        pi, V = EigenUpdater.eigenAdd2(omega, Q, Y1, Y2, k)

        #check last rows are zero
        if numpy.linalg.norm(V[n:, :]) >= EigenUpdater.tol:
            logging.warn("numpy.linalg.norm(V[n:, :])= %s" % str(numpy.linalg.norm(V[n:, :])))

        logging.debug("</ eigenRemove >")
        if not debug:
            return pi, V[0:n, :]
        else:
            C = (Q*omega).dot(Q.T)
            K = C + Y1.dot(Y2.T) + Y2.dot(Y1.T)
            assert numpy.linalg.norm(BB- C[n:, n:]) <= EigenUpdater.tol
            assert numpy.linalg.norm(AB - C[0:n, n:]) <= EigenUpdater.tol, "%s \n %s" % (AB, C[0:n, n:])
            return pi, V[0:n, :], K, Y1, Y2, omega

    #If a value is less than this we consider it to be zero 
    tol = 10**-8
