""" I just created this small script as a reality check to confirm that the
state augmentation strategy I'm using is valid. """

from __future__ import print_function
import sys, re
from numpy import array, identity, zeros, stack
from numpy.random import randn
from numpy.linalg import cholesky
from bokeh.plotting import figure, output_file, gridplot, show
from kalman_lqg import kalman_lqg
from iLQG import iterative_lqg, compute_trajectories, optimize_nominal_trajectories

# define constants
nx = 2
nu = 2
nw = 2
ny = 2
nv = 2
A = array([[0.9, 0.0], [0.0, 0.1]])
B = array([[1.0, 0.0], [0.0, 1.0]])
C0 = cholesky(array([[0.11, 0],[0, 0.11]]))
H = array([[1.0, 0.0], [0.0, 1.0]])
D0 = cholesky(array([[0.1, 0],[0, 0.1]]))
Q = array([[0.4, 0.0],[0.0, 0.0]])
R = array([[9.9, 0.0],[0.0, 0.0]])
Qf = array([[4.1, 0.0],[0.0, 0.0]])
Qf = array([[0.4, 0.0],[0.0, 0.0]])


# find the optimal solution for this LQG system using kalman_lqg.py
N = 10 # number of time steps + 1
x0 = array([100,100])
system = {}
system['X1'] = x0
system['S1'] = identity(nx)
system['A'] = A
system['B'] = B
system['C0'] = C0
system['C'] = zeros([nu, nu, 1])
system['H'] = H
system['D0'] = D0
system['D'] = zeros([ny, nx, 1])
system['E0'] = zeros([nx, 1])
system['Q'] = stack([Q if k < N-1 else Qf for k in range(N)], -1)
system['R'] = R
 
def augment_system(system):
    """ Modify the system matrices to accommodate the augmented state vector:

    xa = (x u 1).T

    This requires augmentation of the matrices A, B, C0, H, D, and Q. The
    augmentation of C0, H, and D is trivial as it simply involves adding zeros.
    The augmentation of A contains two 1s for the constant terms added to the
    state vector.  The augmentation of B contains an identity submatrix which
    enables the addition of the control inputs to the state vector. The
    augmentation of Q is the most interesting.

    Qa = [[ Q   0  q/2]
          [ 0...R  r/2]
          [q/2 r/2  0]]
    
    Since the control costs are now state costs the control costs passed to
    kalman_lqg are zero, i.e. Ra = 0.
    """
    x_star = array([13.0, 0.0])
    u_star = array([3.1, 0.0])
    # save the matrices needed for modification
    X1 = system['X1']
    S1 = system['S1']
    A = system['A']
    B = system['B']
    C0 = system['C0']
    H = system['H']
    D = system['D']
    Q = system['Q']
    R = system['R']
    dt = 1  # until there's a reason to use something else
    nx = A.shape[0]
    nxa = A.shape[0] + B.shape[1] + 1 # for state augmentation
    nu = B.shape[1]
    szC0 = C0.shape[1]
    ny = H.shape[0]
    szD0 = D0.shape[1]
    N = Q.shape[2]

    # build the vector for the initial augmented state
    x0a = X1.tolist()
    u0a = [0 if i != nu else 1 for i in range(nu+1)]
    system['X1'] = array(x0a + u0a).T  # column vector
    S1 = identity(nxa)
    S1[-1,-1] = 0
    system['S1'] = S1
    system['A'] = zeros([nxa, nxa, N-1])
    system['B'] = zeros([nxa, nu, N-1])
    system['C0'] = zeros([nxa, szC0, N-1])
    system['C'] = zeros([nu, nu, szC0, N-1])
    system['H'] = zeros([ny, nxa, N-1])
    system['D0'] = zeros([ny, szD0, N-1])
    system['D'] = zeros([ny, nxa, szD0, N-1])
    system['Q'] = zeros([nxa, nxa, N])
    system['R'] = zeros([nu, nu, N-1])
    for k in range(N-1):
        # augment matrices to accommodate linear state and control costs
        Aa = zeros([nxa, nxa])
        Aa[0:nx,0:nx] = A
        Aa[-1,-1] = 1
        system['A'][:,:,k] = Aa
        Ba = zeros([nxa, nu])
        Ba[0:nu,0:nu] = B
        Ba[nx:nx+nu,0:nu] = identity(nu)
        system['B'][:,:,k] = Ba
        C0a = zeros([nxa, szC0])
        C0a[0:nx,0:szC0] = C0
        system['C0'][:,:,k] = C0a
        # meethinks no augmentation necessary for C now
        #for j in range(C.shape[2]):
        #    system['C'][:,:,j,k] = add_row(add_col(C[:,:,j]))
        Ha = zeros([ny, nxa])
        Ha[0:ny,0:nx] = H
        system['H'][:,:,k] = Ha
        for j in range(D.shape[2]):
            Da = zeros([ny, nxa])
            Da[0:ny,0:nx] = D[:,:,j]
            system['D'][:,:,j,k] = Da
        Qa = zeros([nxa, nxa])
        Qa[0:nx,0:nx] = Q[:,:,k]
        q = -2*x_star.dot(Q[:,:,k])
        Qa[0:nx,nx+nu] = q/2
        Qa[nx+nu,0:nx] = q/2
        Qa[nx:nx+nu,nx:nx+nu] = R
        r = -2*u_star.dot(R)
        Qa[nx:nx+nu,nx+nu] = r/2
        Qa[nx+nu,nx:nx+nu] = r/2
        Qa[-1,-1] = x_star.dot(Q[:,:,k]).dot(x_star)
        Qa[-1,-1] += u_star.dot(R).dot(u_star)
        system['Q'][:,:,k] = Qa
        Ra = zeros([nu, nu])
        #Ra = R
        system['R'][:,:,k] = Ra

    # build Qa for the last time point's state cost 
    #Q = Q
    q = zeros(nx)
    #R = R
    Qa = zeros([nxa, nxa])
    Qa[0:nx,0:nx] = Q[:,:,-1]
    q = -2*x_star.dot(Q[:,:,k])
    Qa[0:nx,nx+nu] = q/2
    Qa[nx+nu,0:nx] = q/2
    Qa[nx+1:nx+nu+1,nx+1:nx+nu+1] = R
    r = -2*u_star.dot(R)
    Qa[nx:nx+nu,nx+nu] = r/2
    Qa[nx+nu,nx:nx+nu] = r/2
    Qa[-1,-1] = x_star.dot(Q[:,:,k]).dot(x_star)
    Qa[-1,-1] += u_star.dot(R).dot(u_star)
    system['Q'][:,:,N-1] = Qa
    # iLQG does not accommodate noise added to the state estimate
    system['E0'] = zeros([1, 1, N])
    return system


def compute_LQG_trajectories(x0, l, L):
    """ Compute the deterministic state and control trajectories using the
    control law u(k) = -L(k)x(k). """
    dt = 1  # until there's a reason to use something else
    nu = L.shape[0]
    N = L.shape[2] + 1
    u = zeros([nu, N-1])
    nx = x0.shape[0]
    x = zeros([nx, N])
    x[:,0] = x0
    for k in range(N-1):
        u[:,k] = -(l[:,k] + L[:,:,k].dot(x[:,k]))
        x[:,k+1] = A.dot(x[:,k]) + B.dot(u[:,k])
    return x, u


def parse_L(L, nx):
    N = L.shape[2] + 1
    nu = L.shape[0]
    l = zeros([nu, N-1])
    Lx = zeros([nu, nx, N-1])
    for k in range(N-1):
        l[:,k] = L[:,-1,k]
        Lx[:,:,k] = L[:,0:nx,k]
    return l, Lx


def perturb(matrix_trajectory, scale):
    """ Perturb the given matrix trajectory with Gaussian noise scaled by
    scale.  """
    perturbed_trajectory = matrix_trajectory.flatten()
    for i in range(len(perturbed_trajectory)):
        perturbed_trajectory[i] += scale*randn()
    return perturbed_trajectory.reshape(matrix_trajectory.shape)


def compute_cost(system, L):
    """
    Use the cost function x.T*Q*x + u.T*R*u to compute the total cost for a 
    deterministic state trajectory starting from x0 and obeying these equations:
    x(k+1) = A(k)*x(k) + B(k)*L(k)*x(k)
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
    assert len(Q.shape) == 3 and Q.shape[2] == Nx, \
            "Q must contain a matrix for each time step"
    if len(R.shape) == 2: 
        # if R is time invariant make copies for each time step
        R = stack([R for k in range(Nu)], -1)
    assert R.shape[2] == Nu, \
            "R must contain a matrix for each time step but the last"
    cost = 0
    x = x0
    for k in range(Nu):
        cost += x.dot(Q[:,:,k]).dot(x)
        u = -L[:,:,k].dot(x)
        cost += u.dot(R[:,:,k]).dot(u)
        x = A[:,:,k].dot(x) + B[:,:,k].dot(u)

    cost += x.dot(Q[:,:,Nx-1]).dot(x)
    return cost


# the augmented system should produce the optimal solution
augmented_system = augment_system(system)
K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(augmented_system)
l, Lx = parse_L(L, nx)
x_alqg, u_alqg = compute_LQG_trajectories(x0, l, Lx)

# setup output file to plot figures
script_name = re.split(r"/",sys.argv[0])[-1]
output_file_name = script_name.replace(".py", ".html")
output_file(output_file_name, title="")

# plot the state trajectories
kx = range(N)
p1 = figure(title="State Trajectory", x_axis_label='time', y_axis_label='')
p1.line(kx, x_alqg[0,:], line_width=2, line_color="blue")

# plot the control trajectories
ku = range(N-1)
p2 = figure(title="Control Trajectory", x_axis_label='time', y_axis_label='')
p2.line(ku, u_alqg[0,:], line_width=2, line_color="blue")

p = gridplot([[p1, p2]])
show(p)

optimal_cost = compute_cost(system, L)
print("Optimal cost:", optimal_cost)

perturbation_size = 0.0001
for perturbations in range(100):
    # perturb the state trajectory and make sure the expected cost is higher
    suboptimal_L = perturb(L, perturbation_size)
    suboptimal_cost = compute_cost(system, suboptimal_L)
    print("Suboptimal cost - optimal cost: %.15f" %
          (suboptimal_cost - optimal_cost))
    if suboptimal_cost < optimal_cost:
        print("The state trajectory is not optimal!")
    assert suboptimal_cost >= optimal_cost
