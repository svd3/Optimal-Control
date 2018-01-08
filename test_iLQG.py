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
x0 = array([100.0,100.0])
A = array([[1.0, 2.0], [3.0, 4.0]])
B = 1.0/A
C0 = cholesky(array([[0.11, 0],[0, 0.11]]))
C0 = zeros([nx, nw])
H = array([[5.0, 3.0], [2.0, 1.0]])
D0 = cholesky(array([[0.1, 0.0],[0.0, 0.1]]))
D0 = zeros([ny, nv])
R = array([[0.3, 0.1],[0.1, 0.4]])
Q = array([[4.0, 1.0],[1.0, 3.0]])
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


def f(x, u):
    """ If x(k+1) = Ax(k) + Bu(k) then f(x,u) = (A-I)x(k) + Bu(k) """
    I = identity(nx)
    return (A-I).dot(x) + B.dot(u) 

def dfdx(x):
    I = identity(nx)
    return A-I

def dfdu(u):
    return B

def F(x, u):
    return C0

def g(x, u):
    return H.dot(x)

def G(x, u):
    return D0

def h(x):
    return x.dot(Qf).dot(x)

def l(x, u):
    return x.dot(Q).dot(x) + u.dot(R).dot(u)
    
def dldx(x):
    return 2*Q.dot(x)
    
def d2ldx2(x):
    return 2*Q
    
def dldu(u):
    return 2*R.dot(u)
    
def d2ldu2(u):
    return 2*R
    

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


# find the optimal solution for this LQG system using kalman_lqg.py
N = 16 # number of time steps + 1
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
 
#import matlab
#import matlab.engine
#from test_kalman_lqg import matlab_kalman_lqg
#eng = matlab.engine.start_matlab()
#matlab_returns = matlab_kalman_lqg(eng, system)
# parse MATLAB return values according to the function definition
#function [K,L,Cost,Xa,XSim,CostSim,iter] = ...
#K = array(matlab_returns[0])
#L = array(matlab_returns[1])
#Cost = array(matlab_returns[2])
#Xa = array(matlab_returns[3])
#XSim = array(matlab_returns[4])
#CostSim = array(matlab_returns[5])
# I added this return value to make sure the algorithm converged
#iterations = array(matlab_returns[6])
#eng.quit()

# setup output file to plot figures
script_name = re.split(r"/",sys.argv[0])[-1]
output_file_name = script_name.replace(".py", ".html")
output_file(output_file_name, title="")

K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
x_lqg, u_lqg = compute_trajectories(f, x0, zeros([nu,N-1]), -L)

# iLQG should produce the same solution
x_ilqg, u_ilqg, Lx, lx, K = iterative_lqg(f, F, g, G, h, l, x0, N)

# optimization strategy for finding the nominal trajectories
#x_op, u_op = optimize_nominal_trajectories(f, h, l, x0, N)

print "kalman_lqg cost:"
print compute_cost(system, x_lqg, u_lqg)

print "iLQG cost:"
print compute_cost(system, x_ilqg, u_ilqg)

# setup output file to plot figures
script_name = re.split(r"/",sys.argv[0])[-1]
output_file_name = script_name.replace(".py", ".html")
output_file(output_file_name, title="")

# plot the state trajectories
kx = range(N)
p1 = figure(title="State Trajectories", x_axis_label='time', y_axis_label='')
p1.line(kx, x_lqg[0,:], line_width=2, line_color="blue", legend="LQG x[0]")
p1.line(kx, x_lqg[1,:], line_width=2, line_color="green", legend="LQG x[1]")
p1.line(kx, x_ilqg[0,:], line_width=2, line_color="blue", line_dash='dashed',
        legend="iLQG x[0]")
p1.line(kx, x_ilqg[1,:], line_width=2, line_color="green", line_dash='dashed',
        legend="iLQG x[1]")
p1.legend.location = "bottom_left"

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
