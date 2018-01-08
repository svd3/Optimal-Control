# I'm trying to code up the equations from Todorov's 2007 iLQG paper so I can call them from iLQG.py instead of calling kalman_lqg.py.  I'm hoping this will enable the iLQG algo to find the optimal nominal state and control trajectories and do so with fewer iterations.

from __future__ import print_function
from numpy import array, ones, zeros, stack
from scipy.linalg import pinv, svd, norm, inv
from numpy.matlib import repmat
from numpy.random import randn
import matlab


def is_scalar(variable):
    """ Treat variable as a scalar if it is a float or an int. """
    return isinstance(variable, float) or isinstance(variable, int)


def size(A, dimension):
    """ Emulate MATLAB's size function. """
    if is_scalar(A):
        size = 1
    elif len(A.shape) < dimension:
        size = 1
    else:
        size = A.shape[dimension - 1]
    return size


def unpack_Qa(Qas, k, szX, szU):
    qs = Qas[:,:,k][-1,-1]
    q = 2*Qas[:,:,k][0:szX,-1]
    Q = Qas[:,:,k][0:szX,0:szX]
    if k == Qas.shape[2] - 1:
        # last time step, no control costs
        r = zeros(szU)
        R = zeros([szU,szU])
    else:
        # state augmentation puts the control costs in the state cost matrix
        # for THE NEXT TIME STEP
        r = 2*Qas[:,:,k+1][szX:szX+szU,-1]
        R = Qas[:,:,k+1][szX:szX+szU,szX:szX+szU]
    Qa = Qas[:,:,k]
    Ra = zeros([szU,szU])
    return qs, q, Q, r, R, Qa, Ra


def inner_loop(system, NSim=0,Init=1,Niter=0 ):
    #######################################################################
    # initialization
    #######################################################################

    # adapt code for time varying case
    As = system['A']
    Bs = system['B']
    Qas = system['Q']
    Rs = system['R']
    X1 = system['X1']
    S1 = system['S1']
    assert len(Qas.shape) == 3, \
            "Q must contain a state cost matrix for each time step"

    # determine sizes
    szU = size(Bs, 2)
    szX = size(As, 1) - szU - 1
    N = size(Qas, 3)

    if len(As.shape) == 2: 
        # if A is time invariant make copies for each time step but the last
        As = stack([As for k in range(N-1)], -1)
    if len(Bs.shape) == 2: 
        # if B is time invariant make copies for each time step but the last
        Bs = stack([Bs for k in range(N-1)], -1)
    if len(Rs.shape) == 2: 
        # if R is time invariant make copies for each time step but the last
        Rs = stack([Rs for k in range(N-1)], -1)
    
    # allocate memory for La (augmented L)
    L_return = zeros([szU, szX+szU+1, N-1])

    # initialize Ss
    qs, q, Q, r, R, Qa, Ra = unpack_Qa(Qas, N-1, szX, szU)
    Sxa = Qa
    Sxha = zeros(Qa.shape)
    Sxxha = zeros(Qa.shape)
    Sx = Q
    Sxh = zeros(Q.shape)
    Sxxh = zeros(Q.shape)
    sx = q
    sxh = zeros(q.shape)
    s = qs
    # backward pass - recompute control policy
    for k in range(N-2,-1,-1):
        # adapt this loop for time-varying systems
        A = As[0:szX,0:szX,k]
        B = Bs[0:szX,0:szU,k]
        Aa = As[:,:,k]
        Ba = Bs[:,:,k]
        qs, q, Q, r, R, Qa, Ra = unpack_Qa(Qas, k, szX, szU)

        # calculate g, G, and H
        """ Augmented """
        Ha = Ra + Ba.T.dot(Sxa + Sxha + 2*Sxxha).dot(Ba)
        Gxa = Ba.T.dot(Sxa + Sxxha).dot(Aa)
        Gxha = Ba.T.dot(Sxha + Sxxha).dot(Aa)
        Ga = Gxa + Gxha
        """ Regular """
        H = R + B.T.dot(Sx + Sxh + 2*Sxxh).dot(B)
        g = r + B.T.dot(sx + sxh)
        Gx = B.T.dot(Sx + Sxxh).dot(A)
        Gxh = B.T.dot(Sxh + Sxxh).dot(A)
        G = Gx + Gxh

        # calculate l and L
        """ Augmented """
        La = -inv(Ha).dot(Ga)
        """ Regular """
        l = -inv(H).dot(g)
        L = -inv(H).dot(G)

        # repackage l and L into an augmented L to match kalman_lqg interface
        L_return[:,0:szX,k] = -L
        L_return[:,-1,k] = -l
        #L_return[:,:,k] = -La

        # update Ss
        """ Augmented """
        Sxa = Qa + Aa.T.dot(Sxa).dot(Aa)
        Sxha = (Aa.T.dot(Sxha).dot(Aa) + La.T.dot(Ha).dot(La)
               + La.T.dot(Gxha) + Gxha.T.dot(La))
        Sxxha = Aa.T.dot(Sxxha).dot(Aa) + Gxa.T.dot(La)

        """ Regular """
        Sx = Q + A.T.dot(Sx).dot(A)
        Sxh = (A.T.dot(Sxh).dot(A) + L.T.dot(H).dot(L)
               + L.T.dot(Gxh) + Gxh.T.dot(L))
        Sxxh = A.T.dot(Sxxh).dot(A) + Gx.T.dot(L)
        sx = q + A.T.dot(sx) + Gx.T.dot(l)
        sxh = A.T.dot(sxh) + L.T.dot(H).dot(l) + L.T.dot(g) + Gxh.T.dot(l)
        s = qs + s + l.T.dot(g) + 0.5*l.T.dot(H).dot(l)

    # no need to calculate these
    K = []
    Cost = []
    Xa = []
    XSim = []
    CostSim = []
    iteration = 1
    
    return [K, L_return, Cost, Xa, XSim, CostSim, iteration]


