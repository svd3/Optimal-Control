import sys, re
import numpy as np
import math
from numdifftools import Jacobian, Hessian
from numpy.random import randn
from numpy.linalg import cholesky, eig, pinv, norm
#from bokeh.plotting import figure, output_file, gridplot, show
from matplotlib import pyplot as plt
from iLQG import iterative_lqg, linearize_and_quadratize
from kalman_lqg import kalman_lqg

# define constants
# dimensions
nx = 2 #A.shape[0]
nu = 2 #B.shape[1]
nw = 2 #C0.shape[1]
ny = 2 #H.shape[0]
nv = 2 #D0.shape[1]


x0 = np.array([2.0, 0.0])
#x0 = np.array([2.0])
#A = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0,0.0,2.0]])
#B = 1.0/A
#C0 = cholesky(np.array([[0.11, 0],[0, 0.11]]))
C0 = np.zeros([nx, nw])
#H = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
#H = np.identity(3)
#D0 = cholesky(np.array([[0.1, 0.0],[0.0, 0.1]]))
D0 = np.zeros([ny, nv])

R = np.array([[10.0, 0.0],[0.0, 2.0]])
#R = np.array([10.0])
#Q = np.array([1.0])
Q = np.identity(2)
Qf = Q
#Qf = np.array([[5.0, 0.0, 0.0],[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])


# define the system in iLQG form
def f(x, u):
    """ f defines the deterministic part of the system dynamics. """
    # If x(k+1) = Ax(k) + Bu(k) then f(x,u) = (A-I)x(k) + Bu(k)
    #fx = np.array([-x[1]*u[1], x[0]*u[1]-u[0]])
    fx  = -u
    #return (A-I).dot(x) + B.dot(u)
    return fx

def F(x, u):
    """ F defines the stochastic part of the system dynamics. """
    return C0

def g(x, u):
    """ g defines the deterministic part of the system observables. """
    #return H.dot(x)
    return u;

def G(x, u):
    """ G defines the stochastic part of the system observables. """
    return D0

def l(x, u, k):
    """ l defines the system costs prior to the final state. """
    return x.dot(Q).dot(x) + u.dot(R).dot(u)
    
def h(x):
    """ h defines the system costs in the final state. """
    return x.dot(Qf).dot(x)

def generate_ilqg_trajectory(f, F, g, G, h, l, x0, N, nu, xf=None):
    """ Generate an N point state trajectory and its corresponding N-1 point
    control trajectory for the opimal control system described by these
    equations:
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    where x is a vector describing the state of the system, y is a vector
    containing measurable properties of the system and J is the cost to go.
    """
    dt = 0.1 # until there's a reason to use something else
    nx = len(x0)
    u_n = np.zeros([nu,N-1]) # nominal control
    u_p = np.zeros([nu,N-1]) #  actual
    x_n = np.zeros([nx,N]) # nominal
    x_p = np.zeros([nx,N]) # actual
    x_hat = np.zeros([nx,N]) # estimate of x (deviation) from observable
    x_n[:,0] = x0
    x_p[:,0] = x0
    x_hat[:,0] = x_p[:,0] - x_n[:,0]
    ny = len(g(x_n[:,0], u_n[:,0])) #no. of observables
    y_n = np.zeros([ny,N])
    y_p = np.zeros([ny,N])
    Lx = np.zeros([nu,nx,N-1]) # control law (l + Lx)
    lx = np.zeros([nu,N-1]) #
    K = np.zeros([nx,ny,N-1]) # Kalman gain
    nw = F(x_n[:,0], u_n[:,0]).shape[1]
    nv = G(x_n[:,0], u_n[:,0]).shape[1]
    for k in range(N-1):
        print "%3d" % k,
        #print "till now:"
        """print(x_p[0,:])
        print(x_n[0,:])
        print(x_p[1,:])
        print(x_n[1,:])"""
        
        if k == 0 or norm(x_p[:,k] - x_n[:,k]) > 0.05*norm(x_n[:,k]):
            # If k=0 then we have not yet found an approximately optimal
            # control law. If the actual trajectory, x_p, has deviated from the
            # nominal trajectory, x_n, by more than 10% then update the control
            # law.
            print "Deviated"
            solution = iterative_lqg(f, F, g, G, h, l, x_p[:,k], N-k, nu, xf)
            #solution = iterative_lqg(f, F, g, G, h, l, x_hat[:,k], N-k, nu, xf)
            #print(x_n.shape)
            x_n[:,k:N] = solution[0]
            u_n[:,k:N-1] = solution[1]
            print "iterative update (step taken)"
            #print(x_n[0,:])
            #print(x_n[1,:])
            Lx[:,:,k:N-1] = solution[2]
            lx[:,k:N-1] = solution[3]
            K[:,:,k:N-1] = solution[4]
            system = linearize_and_quadratize(f, F, g, G, h, l, x_n, u_n)
            # calculate the nominal observations
            y_n[:,k:N-1] = np.array([g(x_n[:,j], u_n[:,j])
                                     for j in range(k,N-1)]).T
        # calculate the control input
        u_p[:,k] = (Lx[:,:,k].dot(x_hat[:,k]) + lx[:,k] + u_n[:,k])
        #u_p[:,k] = u_n[:,k]
        # calculate the next state
        x_p[:,k+1] = (x_p[:,k] + f(x_p[:,k], u_p[:,k])*dt
                     + F(x_p[:,k], u_p[:,k]).dot(randn(nw)))
        #x_p[:,k+1] = x_n[:,k+1]
        #print(x_p[:,k])
        #print(x_p[:,k+1])
        # calculate the noisy observation
        if k == 0:
            y_p[:,k] = y_n[:,k]
        y_p[:,k+1] = (y_p[:,k] + g(x_p[:,k], u_p[:,k])*dt
                      + G(x_p[:,k], u_p[:,k]).dot(randn(nv)))
        # calculate the state estimate
        y = y_p[:,k] - y_n[:,k]
        A = system['A'][0:nx,0:nx,k]
        B = system['B'][0:nx,:,k]
        H = system['H'][:,0:nx,k]
        u = u_p[:,k] - u_n[:,k]
        x_hat[:,k+1] = (A.dot(x_hat[:,k]) + B.dot(u)
                        + K[:,:,k].dot(y - H.dot(x_hat[:,k])))

#print(system['A'].shape)
    return x_p, u_p


N = 20 # number of time steps + 1
M = 1 # number of trajectories

# Generate some iLQG trajectories
x_p = np.zeros([nx,M,N])
u_p = np.zeros([nu,M,N-1])
x_n = np.zeros([nx,M,N])
u_n = np.zeros([nu,M,N-1])
Lx = np.zeros([nu,nx,M,N-1])
lx = np.zeros([nu,M,N-1])
for m in range(M):
    x_p[:,m,:], u_p[:,m,:] = generate_ilqg_trajectory(f,F,g,G,h,l,x0,N,nu)

# plot the state trajectories
kx = range(N)
p1 = plt.figure()
plt.title("State Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
    #plt.plot(kx, x_lqg[0,:], linewidth=2, color="blue",
    #   linestyle='solid', label="x_lqg[0]")
    #plt.plot(kx, x_lqg[1,:], linewidth=2, color="green",
    #   linestyle='solid', label="x_lqg[1]")
    #plt.plot(kx, x_hat_lqg[0,:], linewidth=2, color="blue",
    #   linestyle='dotted', label="x_hat[0]")
    #plt.plot(kx, x_hat_lqg[1,:], linewidth=2, color="green",
    #   linestyle='dotted', label="x_hat[1]")
    #plt.plot(kx, XSim[0,0,:], linewidth=2, color="blue",
    #   linestyle='dashed', label="XSim[0]")
    #plt.plot(kx, XSim[1,0,:], linewidth=2, color="green",
#   linestyle='dashed', label="XSim[1]")
plt.plot(kx, x_p[0,0,:], linewidth=2, color="green",
         linestyle='solid', label="x")
#plt.plot(kx, x_p[1,0,:], linewidth=2, color="blue",
#    linestyle='solid', label="y")
plt.show()

"""p2 = plt.figure()
plt.title("Angle Trajectories")
axes = plt.gca()
plt.plot(kx, x_p[1,0,:]*180/math.pi, linewidth=2, color="red",
         linestyle='solid', label="o")"""
#for m in range(M):
#    #plt.plot(kx, x_n[0,m,:], linewidth=2, color="blue", label="x_n[0]")
#    #plt.plot(kx, x_n[1,m,:], linewidth=2, color="green", label="x_n[1]")
#    plt.plot(kx, x_p[0,m,:], linewidth=2, color="blue", linestyle='dashed',
#             label="x_p[0]")
#    plt.plot(kx, x_p[1,m,:], linewidth=2, color="green", linestyle='dashed',
#             label="x_p[1]")
#plt.legend(loc="lower right")

# plot the control trajectories
ku = range(N-1)
p3 = plt.figure()
plt.title("Control Trajectories")
axes = plt.gca()
axes.set_xlabel('time')
"""plt.plot(ku, u_lqg[0,:], linewidth=2, color="blue", linestyle='solid',
         label="u_lqg[0]")
plt.plot(ku, u_lqg[1,:], linewidth=2, color="green", linestyle='solid',
         label="u_lqg[1]") """

plt.plot(ku, u_p[0,0,:], linewidth=2, color="green", linestyle='solid',
         label="v")
    #plt.plot(ku, u_p[1,0,:], linewidth=2, color="blue", linestyle='solid',
#    label="ang. w")
plt.show()
#for m in range(M):
#    #plt.plot(ku, u_n[0,m,:], linewidth=2, color="blue", label="u_n[0]")
#    #plt.plot(ku, u_n[1,m,:], linewidth=2, color="green", label="u_n[1]")
#    plt.plot(ku, u_p[0,m,:], linewidth=2, color="blue", linestyle='dashed',
#             label="u_p[0]")
#    plt.plot(ku, u_p[1,m,:], linewidth=2, color="green", linestyle='dashed',
#             label="u_p[1]")
#plt.legend(loc="lower right")
#plt.show(block=False)

## plot the state trajectories
#p3 = figure(title="State Trajectories", x_axis_label='state0',
#            y_axis_label='state1')
#p3.line(x_n[0,:], x_n[1,:], line_width=2, line_color="blue", legend="x_n")
#for m in range(M):
#    p3.line(x_p[0,m,:], x_p[1,m,:], line_width=2, line_color="blue",
#            line_dash='dashed', legend="x_p")
#p3.legend.location = "bottom_right"
#
## plot the control trajectories
#p4 = figure(title="Control Trajectories", x_axis_label='control0',
#            y_axis_label='control1')
#p4.line(u_n[0,:], u_n[1,:], line_width=2, line_color="blue", legend="u_n")
#for m in range(M):
#    p4.line(u_p[0,m,:], u_p[1,m,:], line_width=2, line_color="blue",
#            line_dash='dashed', legend="u_p")
#p4.legend.location = "bottom_right"
#
#p = gridplot([[p1, p2], [p3, p4]])


"""# Find the analytic solution to this LQG system.
    system = {}
    system['X1'] = x0
    system['S1'] = np.identity(nx)
    system['A'] = A
    system['B'] = B
    system['C0'] = C0
    system['C'] = np.zeros([nu, nu, 1])
    system['H'] = H
    system['D0'] = D0
    system['D'] = np.zeros([ny, nx, 1])
    system['E0'] = np.zeros([nx, 1])
    system['Q'] = np.stack([Q if k < N-1 else Qf for k in range(N)], -1)
    system['R'] = R
    
    K_lqg, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system, NSim=1)
    #import matlab
    #import matlab.engine
    #from kalman_lqg import matlab_kalman_lqg
    #eng = matlab.engine.start_matlab()
    #K_lqg, L, Cost, Xa, XSim, CostSim, iterations = matlab_kalman_lqg(eng, system)
    #eng.quit()
    
    u_lqg = np.zeros([nu,N-1])
    x_lqg = np.zeros([nx,N])
    y_lqg = np.zeros([ny,N])
    x_hat_lqg = np.zeros([nx,N])
    x_lqg[:,0] = x0
    y_lqg[:,0] = H.dot(x_lqg[:,0]) + D0.dot(randn(nv))
    x_hat_lqg[:,0] = x0
    for k in range(N-1):
    u_lqg[:,k] = -L[:,:,k].dot(x_hat_lqg[:,k])
    x_lqg[:,k+1] = A.dot(x_lqg[:,k]) + B.dot(u_lqg[:,k]) + C0.dot(randn(nw))
    y_lqg[:,k] = H.dot(x_lqg[:,k]) + D0.dot(randn(nv))
    x_hat_lqg[:,k+1] = (A.dot(x_hat_lqg[:,k]) + B.dot(u_lqg[:,k])
    + K_lqg[:,:,k].dot(y_lqg[:,k] - H.dot(x_hat_lqg[:,k]))) """
