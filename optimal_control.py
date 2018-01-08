""" Shared optimal control functions. """

from numpy import array, zeros, stack
from numpy.random import randn


def is_scalar(variable):
    """ Treat variable as a scalar if it is a float or an int. """
    return isinstance(variable, float) or isinstance(variable, int)


def noise(C0=0, Cx=0, x=0, Cu=0, u=0, NSim=1):
    """ Produce state and control dependent noise. """
    if is_scalar(C0):
        independent_noise = 0
    else:
        assert len(C0.shape) == 2, "C0 must have 2 dimensions"
        independent_noise = C0.dot(randn(C0.shape[1],NSim))
    if is_scalar(Cx):
        state_dep_noise = 0
    else:
        assert len(Cx.shape) == 3, "Cx must have 3 dimensions"
        nCx = Cx.shape[2]
        state_dep_noise = sum([Cx[:,:,i].dot(x)*randn(NSim)
                               for i in range(nCx)])

    if is_scalar(Cu):
        control_dep_noise = 0
    else:
        assert len(Cu.shape) == 3, "Cu must have 3 dimensions"
        nCu = Cu.shape[2]
        control_dep_noise = sum([Cu[:,:,i].dot(u)*randn(NSim)
                                 for i in range(nCu)])
    noise = independent_noise + state_dep_noise + control_dep_noise
    #if not is_scalar(noise):
        # convert column vector to array slice
        #noise = noise[:,0]
    return noise


def compute_cost(system, K, L, NSim=1, deterministic=False):
    """
    Use the cost function x.T*Q*x + u.T*R*u to compute the total cost for NSim
    state trajectories starting from x0 and obeying these equations:
    x(k+1) = A(k)*x(k) + B(k)*L(k)*x_hat(k) + process_noise()
    y(k) = H(k)*x(k) + observation_noise()
    x_hat(k+1) = A(k)*x_hat(k) + B(k)*L(k)*x_hat(k) + K(k)*(y(k) - H*x_hat(k))
    """
    x0 = system['X1']
    A = system['A']
    B = system['B']
    C0 = system['C0']
    C = system['C']
    H = system['H']
    D0 = system['D0']
    D = system['D']
    Q = system['Q']
    R = system['R']
    E0 = system['E0']
    assert len(K.shape) == 3, \
            "K must contain a matrix for each time step but the last"
    assert len(L.shape) == 3, \
            "L must contain a matrix for each time step but the last"
    Nu = L.shape[2]
    Nx = Nu + 1
    if len(A.shape) == 2: 
        # if A is time invariant make copies for each time step
        A = stack([A for k in range(Nu)], -1)
    assert A.shape[2] == Nu, \
            "A must contain a matrix for each time step"
    if len(B.shape) == 2: 
        # if B is time invariant make copies for each time step
        B = stack([B for k in range(Nu)], -1)
    assert B.shape[2] == Nu, \
            "B must contain a matrix for each time step"
    if len(H.shape) == 2: 
        # if H is time invariant make copies for each time step
        H = stack([H for k in range(Nx)], -1)
    assert H.shape[2] == Nx, \
            "H must contain a matrix for each time step"
    assert len(Q.shape) == 3 and Q.shape[2] == Nx, \
            "Q must contain a matrix for each time step"
    if len(R.shape) == 2: 
        # if R is time invariant make copies for each time step
        R = stack([R for k in range(Nu)], -1)
    assert R.shape[2] == Nu, \
            "R must contain a matrix for each time step but the last"
    cost = zeros(NSim)
    x = array([x0[:,0] for i in range(NSim)]).T
    x_hat = x
    if deterministic:
        # no noise
        observation_noise = lambda k: noise()
        control_dep_noise = lambda k: noise()
        process_noise = lambda k: noise()
    else:
        observation_noise = lambda k: noise(x=x, Cx=D[:,:,:,k], C0=D0[:,:,k],
                                            NSim=NSim)
        control_dep_noise = lambda k: noise(u=u, Cu=C[:,:,:,k], NSim=NSim)
        process_noise = lambda k: noise(C0=C0[:,:,k], NSim=NSim)
    for k in range(Nu):
        cost += array([x[:,i].dot(Q[:,:,k]).dot(x[:,i]) for i in range(NSim)])
        u = -L[:,:,k].dot(x_hat)
        u = u + control_dep_noise(k)
        cost += array([u[:,i].dot(R[:,:,k]).dot(u[:,i]) for i in range(NSim)])
        y = H[:,:,k].dot(x) + observation_noise(k)
        x = A[:,:,k].dot(x) + B[:,:,k].dot(u) + process_noise(k)
        x_hat = (A[:,:,k].dot(x_hat) + B[:,:,k].dot(u)
                 + K[:,:,k].dot(y - H[:,:,k].dot(x_hat))
                 + E0[:,:,k].dot(randn(E0[:,:,k].shape[1],NSim)))

    cost += array([x[:,i].dot(Q[:,:,Nx-1]).dot(x[:,i]) for i in range(NSim)])
    return cost


