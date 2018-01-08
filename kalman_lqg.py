#[K,L,Cost,Xa,XSim,CostSim] = 
#   kalman_lqg( A,B,C,C0, H,D,D0, E0, Q,R, X1,S1  [NSim,Init,Niter] )
#
# Compute optimal controller and estimator for generalized LQG
#
# u(t)    = -L(t) x(t)
# x(t+1)  = A x(t) + B (I + Sum(C(i) rnd_1)) u(t) + C0 rnd_n
# y(t)    = H x(t) + Sum(D(i) rnd_1) x(t) + D0 rnd_n
# xhat(t+1) = A xhat(t) + B u(t) + K(t) (y(t) - H xhat(t)) + E0 rnd_n
# x(1)    ~ mean X1, covariance S1
#
# cost(t) = u(t)' R u(t) + x(t)' Q(t) x(t)
#
# NSim    number of simulated trajectories (default 0)  (optional)
# Init    0 - open loop; 1 (default) - LQG; 2 - random  (optional)
# Niter   iterations; 0 (default) - until convergence   (optional)
#
# K       Filter gains
# L       Control gains
# Cost    Expected cost (per iteration)
# Xa      Expected trajectory
# XSim    Simulated trajectories
# CostSim Empirical cost
#
# This is an implementation of the algorithm described in:
#  Todorov, E. (2005) Stochastic optimal control and estimation
#  methods adapted to the noise characteristics of the
#  sensorimotor system. Neural Computation 17(5): 1084-1108
# The paper is available online at www.cogsci.ucsd.edu/~todorov

# Copyright (C) Emanuel Todorov, 2004-2006

from __future__ import print_function
from numpy import array, ones, zeros, diag, trace, log10, sqrt, diff, stack
from numpy import outer
from scipy.linalg import pinv, svd, norm
from numpy.matlib import repmat
from numpy.random import randn
import matlab


def is_scalar(variable):
    """ Treat variable as a scalar if it is a float or an int. """
    return isinstance(variable, float) or isinstance(variable, int)


def dist(positions):
    """ Emulate the single argument version of MATLAB's dist function which
    returns a matrix of the distances between a set of locations. """
    return array([abs(array(positions) - i) for i in positions])


def size(A, dimension):
    """ Emulate MATLAB's size function. """
    if is_scalar(A):
        size = 1
    elif len(A.shape) < dimension:
        size = 1
    else:
        size = A.shape[dimension - 1]
    return size


def kalman_lqg(system, NSim=0,Init=1,Niter=0 ):
    #######################################################################
    # initialization
    #######################################################################

    # adapt code for time varying case
    As = system['A']
    Bs = system['B']
    Cs = system['C']
    C0s = system['C0']
    Hs = system['H']
    Ds = system['D']
    D0s = system['D0']
    E0s = system['E0']
    Q = system['Q']
    Rs = system['R']
    X1 = system['X1']
    S1 = system['S1']
    assert len(Q.shape) == 3, \
            "Q must contain a state cost matrix for each time step"

    # determine sizes
    szX = size(As, 1)
    szU = size(Bs, 2)
    szY = size(Hs, 1)
    szC = size(Cs, 3)
#print(Cs.shape)
    szC0 = size(C0s, 2)
    szD = size(Ds, 3)
    szD0 = size(D0s, 2)
    szE0 = size(E0s, 2)
    N = size(Q, 3)

    # if C or D are scalar, replicate them into vectors
    if size(Cs,1) == 1 and szU > 1:
        Cs = Cs*ones([szU,1,1])
    if is_scalar(Ds):
        if Ds == 0:
            Ds = zeros([szY,szX,1])
        else:
            Ds = Ds*ones([szX,1,1])
            assert szX == szY, 'D can only be a scalar when szX = szY'
    
    # if C0,D0,E0 are scalar, set them to 0 matrices and adjust size
    if is_scalar(C0s) and C0s == 0:
        C0s = zeros([szX, 1])
    if is_scalar(D0s) and D0s == 0:
        D0s = zeros([szY, 1])
    if is_scalar(E0s) and E0s == 0:
        E0s = zeros([szX, 1])
    
    
    if len(As.shape) == 2: 
        # if A is time invariant make copies for each time step but the last
        As = stack([As for k in range(N-1)], -1)
    if len(Bs.shape) == 2: 
        # if B is time invariant make copies for each time step but the last
        Bs = stack([Bs for k in range(N-1)], -1)
    if len(Rs.shape) == 2: 
        # if R is time invariant make copies for each time step but the last
        Rs = stack([Rs for k in range(N-1)], -1)
    if len(C0s.shape) == 2: 
        # if C0 is time invariant make copies for each time step but the last
        C0s = stack([C0s for k in range(N-1)], -1)
    if len(Cs.shape) == 3: 
        # if C is time invariant make copies for each time step but the last
        Cs = stack([Cs for k in range(N-1)], -1)
    if len(E0s.shape) == 2: 
        # if E0 is time invariant make copies for each time step but the last
        E0s = stack([E0s for k in range(N-1)], -1)
    if len(Hs.shape) == 2: 
        # if H is time invariant make copies for each time step
        Hs = stack([Hs for k in range(N)], -1)
    if len(D0s.shape) == 2: 
        # if D0 is time invariant make copies for each time step
        D0s = stack([D0s for k in range(N)], -1)
    if len(Ds.shape) == 3: 
        # if D is time invariant make copies for each time step
        Ds = stack([Ds for k in range(N)], -1)
    
    # numerical parameters
    MaxIter = 500
    Eps = 1e-15
    
    # initialize policy and filter
    K = zeros([szX, szY, N-1])
    L = zeros([szU, szX, N-1])
    
    #######################################################################
    # run iterative algorithm - until convergence or MaxIter
    
    Cost = zeros([MaxIter])
    for iteration in range(MaxIter):
       
        # initialize covariances
        SiE = S1
        #SiX = X1.dot(X1.T)
        SiX = outer(X1, X1)
        SiXE = zeros([szX, szX])
        
        # forward pass - recompute Kalman filter   
        for k in range(N-1):
            # adapt this loop for time-varying systems
            A = As[:,:,k]
            B = Bs[:,:,k]
            C0 = C0s[:,:,k]
            C = Cs[:,:,:,k]
            H = Hs[:,:,k]
            D0 = D0s[:,:,k]
            D = Ds[:,:,:,k]
            E0 = E0s[:,:,k]
           
            # compute Kalman gain
            temp = SiE + SiX + SiXE + SiXE.T
            if size(D,2) == 1:
                DSiD = diag(diag(temp)*D**2)
            else:
                DSiD = zeros([szY, szY])
                for i in range(szD):
                    DSiD = DSiD + D[:,:,i].dot(temp).dot(D[:,:,i].T)
  
            temp_K = A.dot(SiE).dot(H.T)
            K[:,:,k] = temp_K.dot(pinv(H.dot(SiE).dot(H.T)+D0.dot(D0.T)+DSiD))
            
            # compute new SiE
            #print(C0.shape)
            newE = (E0.dot(E0.T) + C0.dot(C0.T) +
                    (A-K[:,:,k].dot(H)).dot(SiE).dot(A.T))
            LSiL = L[:,:,k].dot(SiX).dot(L[:,:,k].T)
            if size(C, 2) == 1:
                newE = newE + B.dot(diag(diag(LSiL)*C**2)).dot(B.T)
            else:
                for i in range(szC):
                    newE = (newE +
                            B.dot(C[:,:,i]).dot(LSiL).dot(C[:,:,i].T).dot(B.T))
            
            # update SiX, SiE, SiXE
            SiX = (E0.dot(E0.T) + K[:,:,k].dot(H).dot(SiE).dot(A.T)
                   + (A-B.dot(L[:,:,k])).dot(SiX).dot((A-B.dot(L[:,:,k])).T) +
                   (A-B.dot(L[:,:,k])).dot(SiXE).dot(H.T).dot(K[:,:,k].T) +
                   K[:,:,k].dot(H).dot(SiXE.T).dot((A-B.dot(L[:,:,k])).T))
            SiE = newE
            SiXE = ((A-B.dot(L[:,:,k])).dot(SiXE).dot((A-K[:,:,k].dot(H)).T) -
                    E0.dot(E0.T))
        
        
        # first pass initialization
        if iteration == 0:
            if Init == 0:
               # open loop
               K = zeros([szX,szY,N-1])
            elif Init == 2:
               # random
               K = randn([szX,szY,N-1])
        
        
        # initialize optimal cost-to-go function
        Sx = Q[:,:,N-1]
        Se = zeros([szX, szX])
        Cost[iteration] = 0
        
        # backward pass - recompute control policy
        for k in range(N-2,-1,-1):
            # adapt this loop for time-varying systems
            A = As[:,:,k]
            B = Bs[:,:,k]
            C0 = C0s[:,:,k]
            C = Cs[:,:,:,k]
            H = Hs[:,:,k]
            D0 = D0s[:,:,k]
            D = Ds[:,:,:,k]
            E0 = E0s[:,:,k]
            R = Rs[:,:,k]

            # update Cost
            Cost[iteration] = \
                   (Cost[iteration] + trace(Sx.dot(C0).dot(C0.T)) +
                    trace(Se.dot(K[:,:,k].dot(D0).dot(D0.T).dot(K[:,:,k].T)
                                 + E0.dot(E0.T) + C0.dot(C0.T))))

           
            # Controller
            temp = R + B.T.dot(Sx).dot(B)
            BSxeB = B.T.dot(Sx+Se).dot(B)
            if size(C, 2) == 1:
                temp = temp + diag(diag(BSxeB)*C**2)
            else:
                for i in range(size(C, 3)):
                    temp = temp + C[:,:,i].T.dot(BSxeB).dot(C[:,:,i])
            L[:,:,k] = pinv(temp).dot(B.T).dot(Sx).dot(A)
 
            # compute new Se
            newE = (A.T.dot(Sx).dot(B).dot(L[:,:,k]) +
                    (A-K[:,:,k].dot(H)).T.dot(Se).dot(A-K[:,:,k].dot(H)))
           
            # update Sx and Se
            Sx = Q[:,:,k] + A.T.dot(Sx).dot(A-B.dot(L[:,:,k]))
            KSeK = K[:,:,k].T.dot(Se).dot(K[:,:,k])
            if size(D, 2) == 1:
                Sx = Sx + diag(diag(KSeK)*D**2)
            else:
                for i in range(szD):
                    Sx = Sx + D[:,:,i].T.dot(KSeK).dot(D[:,:,i])
            Se = newE
        
        # adjust cost
        Cost[iteration] = (Cost[iteration] + X1.T.dot(Sx).dot(X1) +
                           trace((Se+Sx).dot(S1)))
        
        # progress bar
        #if (iteration + 1) % 10 == 0:
            #print('.', end="")
        
        # check convergence of Cost
        if ((Niter > 0 and iteration >= Niter) or
            (Niter == 0 and iteration > 0 and
             abs(Cost[iteration-1] - Cost[iteration]) < Eps) or
            (Niter == 0 and iteration > 20 and
             all(sum(diff(dist(range(iteration-10,iteration+1)), axis=0) > 0) >
             3))):
            break
     
    # print result
    #print()
    #print("iterations %d" % (iteration+1))
    #print("%.15f" % Cost[iteration-1])
    #print("%.15f" % Cost[iteration])
    #if Cost[iteration-1] != Cost[iteration]:
        #print(' Log10DeltaCost = %.2f\n' % \
                    #log10(abs(Cost[iteration-1]-Cost[iteration])))
    #else:
        #print(' DeltaCost = 0\n')
     
     
     
    #######################################################################
    # compute average trajectory
    
    Xa = zeros([szX, N])
    Xa[:,0] = X1
    
    for k in range(N-1):
        u = -L[:,:,k].dot(Xa[:,k])
        Xa[:,k+1] = A.dot(Xa[:,k]) + B.dot(u)
    
    
    
    #######################################################################
    # simulate noisy trajectories
    
    if NSim > 0:
       
        # square root of S1
        [u,s,v] = svd(S1)
        sqrtS = u.dot(diag(sqrt(diag(s)))).dot(v.T)
       
        # initialize
        XSim = zeros([szX,NSim,N])
        Xhat = zeros([szX,NSim,N])
        X1_column_vector = array([X1]).T
        Xhat[:,:,0] = repmat(X1_column_vector, 1, NSim)
        XSim[:,:,0] = (repmat(X1_column_vector, 1, NSim)
                       + sqrtS.dot(randn(szX,NSim)))
       
        CostSim = 0
       
        # loop over N
        for k in range(N-1):
          
            # update control and cost
            U = -L[:,:,k].dot(Xhat[:,:,k])
            CostSim = CostSim + sum(sum(U*(R.dot(U))))
            CostSim = CostSim + sum(sum(XSim[:,:,k]*(Q[:,:,k].dot(XSim[:,:,k]))))
          
            # compute noisy control
            Un = U
            if size(C, 2) == 1:
                Un = Un + U*randn(szU,NSim)*repmat(C, 1, NSim)
            else:
                for i in range(szC):
                    Un = Un + (C[:,:,i].dot(U))*repmat(randn(1,NSim), szU, 1)
          
            # compute noisy observation
            y = H.dot(XSim[:,:,k]) + D0.dot(randn(szD0,NSim))
            if size(D, 2) == 1:
                y = y + XSim[:,:,k]*randn(szY,NSim)*repmat(D, 1, NSim)
            else:
                for i in range(szD):
                    y = y + (D[:,:,i].dot(XSim[:,:,k]))*repmat(randn(1,NSim),szY,1)
          
            XSim[:,:,k+1] = (A.dot(XSim[:,:,k]) + B.dot(Un) +
                             C0.dot(randn(szC0,NSim)))
            Xhat[:,:,k+1] = (A.dot(Xhat[:,:,k]) + B.dot(U) +
                             K[:,:,k].dot(y-H.dot(Xhat[:,:,k])) +
                             E0.dot(randn(szE0,NSim)))
       
        # final cost update
        CostSim = CostSim + sum(sum(XSim[:,:,N-1]*(Q[:,:,N-1].dot(XSim[:,:,N-1]))))
        CostSim = CostSim / NSim
       
    else:
        XSim = []
        CostSim = []
    
    return [K, L, Cost, Xa, XSim, CostSim, iteration]


def matlab_kalman_lqg(eng, system):
    """ This function makes it easy to Call Todorov's MATLAB code from Python.
    The eng argument is a MATLAB execution engine. This is the usage model:

        import matlab.engine
        eng = matlab.engine.start_matlab()
        K, L, Cost, Xa, XSim, CostSim, iterations = \
                matlab_kalman_lqg(eng, system)
        eng.quit()
    """
    # convert numpy arrays to matlab arrays
    A = matlab.double(system['A'][:,:].tolist())
    B = matlab.double(system['B'][:,:].tolist())
    C = matlab.double(system['C'][:,:].tolist())
    C0 = matlab.double(system['C0'][:,:].tolist())
    H = matlab.double(system['H'][:,:].tolist())
    D = matlab.double(system['D'][:,:].tolist())
    D0 = matlab.double(system['D0'][:,:].tolist())
    E0 = matlab.double(system['E0'][:,:].tolist())
    Q = matlab.double(system['Q'].tolist())
    R = matlab.double(system['R'][:,:].tolist())
    # make sure X1 is a column vector
    if len(system['X1'].shape) == 1:
        X1 = matlab.double(array([system['X1']]).T.tolist())
    else:
        X1 = matlab.double(system['X1'].tolist())
    S1 = matlab.double(system['S1'].tolist())

    matlab_returns = eng.kalman_lqg(A, B,  C, C0,
                                    H, D, D0, E0,
                                    Q, R, X1, S1, nargout=7)
    # convert matlab arrays to numpy arrays and return
    return [array(r) for r in matlab_returns]


