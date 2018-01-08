import sys, re
from numpy import array, identity, zeros, stack
from numpy.random import randn
from numpy.linalg import cholesky
from bokeh.plotting import figure, output_file, gridplot, show
from kalman_lqg import kalman_lqg
from kalman_lqg_simplified import kalman_lqg as kalman_lqg2
from iLQG_inner_loop import inner_loop
from iLQG import iterative_lqg, compute_trajectories, optimize_nominal_trajectories

# define constants
nx = 2
nu = 2
nw = 2
ny = 2
nv = 2
x0 = array([100.0,100.0])
A = array([[1.0, 2.0], [3.0, 4.0]])
B = 1.0/A
C0 = cholesky(array([[0.11, 0],[0, 0.11]]))
C0 = zeros([nx, nw])
#H = array([[5.0, 3.0], [2.0, 1.0]])
H = array([[1.0, 0.0], [0.0, 1.0]])
D0 = cholesky(array([[0.1, 0.0],[0.0, 0.1]]))
D0 = zeros([ny, nv])
Q = array([[4.0, 1.0],[1.0, 3.0]])
R = array([[0.3, 0.1],[0.1, 0.4]])
Qf = array([[4.1, 1.0],[1.0, 3.2]])


def compute_LQG_trajectories(x0, L):
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
        u[:,k] = -L[:,:,k].dot(x[:,k])
        x[:,k+1] = A.dot(x[:,k]) + B.dot(u[:,k])
    return x, u


def compute_LQG_state_trajectory(x0, u):
    """ Compute the deterministic state trajectory using the given sequence of
    control inputs. """
    dt = 1  # until there's a reason to use something else
    nu = u.shape[0]
    N = u.shape[1] + 1
    nx = x0.shape[0]
    x = zeros([nx, N])
    x[:,0] = x0
    for k in range(N-1):
        x[:,k+1] = A.dot(x[:,k]) + B.dot(u[:,k])
    return x


def const_gain_trajectories(x0, L, Nu):
    """ Compute the deterministic state and control trajectories using the
    control law u(k) = -L(k)x(k). """
    dt = 1  # until there's a reason to use something else
    nu = L.shape[0]
    u = zeros([nu, Nu])
    nx = x0.shape[0]
    x = zeros([nx, Nu + 1])
    x[:,0] = x0
    for k in range(Nu):
        u[:,k] = -L[:,:,0].dot(x[:,k])
        x[:,k+1] = A.dot(x[:,k]) + B.dot(u[:,k])
    return x, u


def compute_cost(system, x, u):
    """
    Use the cost function x.T*Q*x + u.T*R*u to compute the total cost for the
    deterministic part of the given system.
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
    Nu = u.shape[1]
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
    if len(Q.shape) == 2: 
        # if Q is time invariant make copies for each time step
        Q = stack([Q for k in range(Nx)], -1)
    assert len(Q.shape) == 3 and Q.shape[2] == Nx, \
            "Q must contain a matrix for each time step"
    if len(R.shape) == 2: 
        # if R is time invariant make copies for each time step
        R = stack([R for k in range(Nu)], -1)
    assert R.shape[2] == Nu, \
            "R must contain a matrix for each time step but the last"
    cost = 0
    for k in range(Nu):
        cost += x[:,k].dot(Q[:,:,k]).dot(x[:,k])
        cost += u[:,k].dot(R[:,:,k]).dot(u[:,k])

    cost += x[:,Nx-1].dot(Q[:,:,Nx-1]).dot(x[:,Nx-1])
    return cost


def augment_system(system):
    """ Augment a standard LQG system so it can be solved by inner_loop """
    N = system['Q'].shape[2]
    nx = system['A'].shape[0]
    nu = system['B'].shape[1]
    nxa = nx + nu + 1 # for state augmentation

    #szC0 = F(x[:,0], u[:,0]).shape[1]
    #ny = g(x[:,0], u[:,0]).shape[0]
    #szD0 = G(x[:,0], u[:,0]).shape[1]
    augmented_system = {}

    # build the vector for the initial augmented state
    x0 = zeros(nxa)
    x0[0:nx] = system['X1']
    x0[nx:nx+nu] = zeros(nu)
    x0[-1] = 1.0
    augmented_system['X1'] = x0
    S1 = identity(nxa)
    S1[-1,-1] = 0.0
    augmented_system['S1'] = S1
    augmented_system['A'] = zeros([nxa, nxa, N-1])
    Aa = zeros([nxa, nxa])
    Aa[0:nx,0:nx] = system['A']
    Aa[-1,-1] = 1.0
    augmented_system['A'] = Aa
    Ba = zeros([nxa, nu])
    Ba[0:nu,0:nu] = B
    Ba[nx:nx+nu,0:nu] = identity(nu)
    augmented_system['B'] = Ba
    #augmented_system['A'] = zeros([nxa, nxa, N-1])
    #augmented_system['B'] = zeros([nxa, nu, N-1])
    #system['C0'] = zeros([nxa, szC0, N-1])
    #system['C'] = zeros([nu, nu, szC0, N-1])
    #system['H'] = zeros([ny, nxa, N-1])
    #system['D0'] = zeros([ny, szD0, N-1])
    #system['D'] = zeros([ny, nxa, szD0, N-1])
    #system['R'] = zeros([nu, nu, N-1])
    augmented_system['R'] = zeros([nu, nu])
    augmented_system['Q'] = zeros([nxa, nxa, N])
    for k in range(N):
        r = zeros(nu)
        if k == 0:
            # Due to state augmentation, the cost for control at k=0 will be
            # paid when k=1 so r[0] and R[0] are all zeros.
            R = zeros([nu, nu])
        else:
            R = system['R']
        qs = 0
        q = zeros(nx)
        Q = system['Q'][:,:,k]
        Qa = zeros([nxa, nxa])
        Qa[0:nx,0:nx] = Q
        Qa[0:nx,nx+nu] = q/2
        Qa[nx+nu,0:nx] = q/2
        Qa[nx:nx+nu,nx:nx+nu] = R
        Qa[nx:nx+nu,nx+nu] = r/2
        Qa[nx+nu,nx:nx+nu] = r/2
        Qa[-1,-1] = qs
        augmented_system['Q'][:,:,k] = Qa

    return augmented_system


# find the optimal solution for this LQG system using kalman_lqg.py
N = 6 # number of time steps + 1
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
 
# setup output file to plot figures
script_name = re.split(r"/",sys.argv[0])[-1]
output_file_name = script_name.replace(".py", ".html")
output_file(output_file_name, title="")

K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg2(system)
x_lqg, u_lqg = compute_LQG_trajectories(x0, L)

# iLQG inner loop should produce the same solution
augmented_system = augment_system(system)
K, La, Cost, Xa, XSim, CostSim, iterations = inner_loop(augmented_system)
L = La[0:nu,0:nx]
x_ilqg, u_ilqg = compute_LQG_trajectories(x0, L)

print "kalman_lqg cost:"
print compute_cost(system, x_lqg, u_lqg)

print "iLQG cost:"
print compute_cost(system, x_ilqg, u_ilqg)

# setup output file to plot figures
script_name = re.split(r"/",sys.argv[0])[-1]
output_file_name = script_name.replace(".py", ".html")
output_file(output_file_name, title="")

# plot the state trajectories
def vector_coordinates(x):
    return [array([0, x[0]]), array([0, x[1]])]


from numpy.linalg import eig, norm
from bokeh.models import Range1d
eig_values, eig_vectors = eig(A)

kx = range(N)
p1 = figure(title="State Trajectories", x_axis_label='time', y_axis_label='')
p1.line(kx, x_lqg[0,:], line_width=2, line_color="blue", legend="LQG x[0]")
p1.line(kx, x_lqg[1,:], line_width=2, line_color="green", legend="LQG x[1]")
p1.line(kx, x_ilqg[0,:], line_width=2, line_color="blue", line_dash='dashed',
        legend="iLQG x[0]")
p1.line(kx, x_ilqg[1,:], line_width=2, line_color="green", line_dash='dashed',
        legend="iLQG x[1]")
p1.legend.location = "bottom_right"

# plot the control trajectories
ku = range(N-1)
p2 = figure(title="Control Trajectories", x_axis_label='time', y_axis_label='')
p2.line(ku, u_lqg[0,:], line_width=2, line_color="blue", legend="LQG u[0]")
p2.line(ku, u_lqg[1,:], line_width=2, line_color="green", legend="LQG u[1]")
p2.line(ku, u_ilqg[0,:], line_width=2, line_color="blue", line_dash='dashed',
        legend="iLQG u[0]")
p2.line(ku, u_ilqg[1,:], line_width=2, line_color="green", line_dash='dashed',
        legend="iLQG u[1]")
p2.legend.location = "bottom_right"

# plot the state trajectories
p3 = figure(title="State Trajectories", x_axis_label='state0',
            y_axis_label='state1')
p3.line(x_lqg[0,:], x_lqg[1,:], line_width=2, line_color="blue", legend="LQG")
p3.line(x_ilqg[0,:], x_ilqg[1,:], line_width=2, line_color="blue", line_dash='dashed',
        legend="iLQG")
p3.x_range = Range1d(-500,100)
p3.y_range = Range1d(-100,500)
p3.legend.location = "bottom_right"

# plot the control trajectories
p4 = figure(title="Control Trajectories", x_axis_label='control0',
            y_axis_label='control1')
p4.line(u_lqg[0,:], u_lqg[1,:], line_width=2, line_color="blue", legend="LQG")
p4.line(u_ilqg[0,:], u_ilqg[1,:], line_width=2, line_color="blue", line_dash='dashed',
        legend="iLQG")
p4.legend.location = "bottom_right"

p = gridplot([[p1, p2], [p3, p4]])
show(p)
