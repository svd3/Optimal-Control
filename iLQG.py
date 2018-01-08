"""
This is an implementation of the optimal control algorithm from a 2007 paper by
Li and Todorov titled "Iterative linearization methods for approximately
optimal control and estimation of non-linear stochastic system." Unlike that
paper, this implementation uses Todorov's 2005 code (kalman_lqg) as the inner
loop in this iterative algorithm. In general the system to be controlled will
have non-linear dynamics, non-quadratic costs, and control and state dependent
noise. An initial state trajectory is chosen and the control trajectory
required to produce that state trajectory is computed.  At each time step, the
system dynamics will be linearized, the costs quadratized, the process noise
linearized with respect to the control signal and the observation noise
linearized with respect to the state so that kalman_lqg can be used to find an
approximately optimal control law. This approximate control law is used to
update the state and control trajectories used for the linearization and
quadritization process and this two-step process is repeated until the solution
reaches a steady state.

"""

# import rlcompleter, readline
##readline.parse_and_bind('tab: complete')

from numpy import array, zeros, ones, identity, swapaxes, einsum, diag
from numpy import sqrt
from numpy.linalg import pinv
from numpy.random import randn
from numdifftools import Jacobian, Hessian
from optimal_control import noise
from kalman_lqg import kalman_lqg
# debug stuff
import sys, re
from bokeh.plotting import figure, output_file, gridplot, show
from iLQG_inner_loop import inner_loop
from matplotlib import pyplot as plt

dt = 0.1

def compute_control_trajectory(f, x, nu):
    """ Compute the control trajectory from the state trajectory and
    the function that describes the system dynamics. """
    #assert len(x.shape) == 2, "x must have 2 dimensions"
    # Allocate memory for the control trajectory.
    N = x.shape[1]
    u = zeros([nu, N-1])
    # Calculate the the control input estimate, u_hat, for the first time step.
    u_hat = zeros(nu)
    #dt = 1.0  # until there's a reason to use something else
    dx = x[:,1] - x[:,0]
    dfdu = Jacobian(lambda u: f(x[:,0], u))
    u_hat = pinv(dfdu(u_hat)).dot(dx/dt - f(x[:,0], u_hat))
    for k in range(N-1):
        dfdu = Jacobian(lambda u: f(x[:,k], u))
        # find the change in u that makes f(x,u)dt close to dx
        dx = x[:,k+1] - x[:,k]
        du = pinv(dfdu(u_hat)).dot(dx/dt - f(x[:,k], u_hat))
        u_hat += du
        u[:,k] += u_hat
    return u


def compute_state_trajectory(f, x0, u):
    """ Compute the state trajectory given the system dynamics, f, initial
    state, x0, and a control trajectory, u. """
    #dt = 1.0  # until there's a reason to use something else
    N = u.shape[1] + 1
    nx = x0.shape[0]
    x = zeros([nx, N])
    x[:,0] = x0
    for k in range(N-1):
        x[:,k+1] = x[:,k] + f(x[:,k], u[:,k])*dt
    return x


def initial_trajectories(f, x0, xf, nu, N):
    """ Compute the initial state and control trajectories to use for the first
    linearization of the system dynamics and quadratization the costs. Try to
    find a control trajectory that makes the state trajectory a straight line
    from x0 to xf."""
    # Compute the straight line trajectory from x0 to xf.
    dx = (xf - x0) / float(N-1)
    x = array([x0 + i*dx for i in range(N)]).T
    # Compute the control trajectory required for this state trajectory.
    nx = x0.shape[0]
    #x = x = zeros([nx, N]) +1
    u = compute_control_trajectory(f, x, nu)
    return x, u


def linearize_and_quadratize(f, F, g, G, h, l, x, u):
    """ Linearize the system dynamics and quadratize the costs around the state
    and control trajectories described by x and u for a system governed by the
    following equations.
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    Where J(x) is the cost to go.
    We are using kalman_lqg as the 'inner loop' of this algorithm and it does
    not explicitly support linear state and control costs so we will augment
    the state vector to include the control inputs and a constant term. The
    augmented state vector, xa, is shown below.

    xa[k] = (x[k] u[k-1] 1).T

    This requires augmentation of the matrices A, B, C0, H, D, and Q. The
    augmentation of C0, H, and D is trivial as it simply involves adding zeros.
    The augmentation of A contains an additional 1 for the constant term added
    to the state vector.  The augmentation of B contains an identity submatrix
    which enables the addition of the control inputs to the state vector. The
    augmentation of Q is the most interesting.

    Qa[k] = [[ Q[k]    0         q[k]/2   ]
             [ 0       R[k-1]    r[k-1]/2 ]
             [ q[k]/2  r[k-1]/2  qs[k]    ]]
    
    Since the control costs are now state costs the control costs passed to
    kalman_lqg are zero, i.e. Ra = 0.
    """
    #dt = 1.0  # until there's a reason to use something else
    nx = x.shape[0]
    nxa = x.shape[0] + u.shape[0] + 1 # for state augmentation
    nu = u.shape[0]
    szC0 = F(x[:,0], u[:,0]).shape[1]
    ny = g(x[:,0], u[:,0]).shape[0]
    szD0 = G(x[:,0], u[:,0]).shape[1]
    N = x.shape[1]
    system = {}

    # build the vector for the initial augmented state
    x0a = [0.0 for i in range(nx)]
    u0a = [0.0 if i != nu else 1.0 for i in range(nu+1)]
    system['X1'] = array(x0a + u0a)
    S1 = identity(nxa)
    S1[-1,-1] = 0.0
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
        dfdx = Jacobian(lambda x: f(x, u[:,k]))
        A = dfdx(x[:,k]) + identity(nx)
        dfdu = Jacobian(lambda u: f(x[:,k], u))
        B = dfdu(u[:,k])
        C0 = sqrt(dt)*F(x[:,k], u[:,k])
        dFdu = Jacobian(lambda u: F(x[:,k], u))
        # dFdu is x by w by u, BC is x by u by w, so swap last 2 dimensions
        # and multiply by pinv(B) to get C
        # !! Dont swap dFdu is x by u by w
        #C = sqrt(dt)*einsum('hi,ijk', pinv(B), swapaxes(dFdu(u[:,k]), -1, -2))
        C = sqrt(dt)*einsum('hi,ijk', pinv(B), dFdu(u[:,k]))
        #print(B.shape)
        #print(dFdu(u[:,k]).shape)
        #print(C.shape)
        dgdx = Jacobian(lambda x: g(x, u[:,k]))
        H = dgdx(x[:,k])
        system['D0'][:,:,k] = G(x[:,k], u[:,k])/sqrt(dt)
        dGdx = Jacobian(lambda x: G(x, u[:,k]))
        # !!!!!!!!!!!!!!! print(dGdx(x[:,k]).shape)
        # dGdx is y by v by x, D is y by x by v, so swap last 2 dimensions  !!
        #D = swapaxes(dGdx(x[:,k]), -1, -2)/sqrt(dt)
        D = dGdx(x[:,k])/dt
        # State cost, constant, linear, quadratic terms
        qs = dt*l(x[:,k], u[:,k],k)
        dldx = Jacobian(lambda x: l(x, u[:,k],k))
        q = dt*dldx(x[:,k])
        d2ldx2 = Hessian(lambda x: l(x, u[:,k],k))
        Q = dt*d2ldx2(x[:,k])
        if k == 0:
            # Due to state augmentation, the cost for control at k=0 will be
            # paid when k=1 so r[0] and R[0] are all zeros.
            r = zeros(nu)
            R = zeros([nu, nu])
        else:
            dldu = Jacobian(lambda u: l(x[:,k-1], u,k))
            d2ldu2 = Hessian(lambda u: l(x[:,k-1], u,k))
            r = dt*dldu(u[:,k-1])
            R = dt*d2ldu2(u[:,k-1])
        # augment matrices to accommodate linear state and control costs
        Aa = zeros([nxa, nxa])
        Aa[0:nx,0:nx] = A
        Aa[-1,-1] = 1.0
        system['A'][:,:,k] = Aa
        Ba = zeros([nxa, nu])
        Ba[0:nx,0:nu] = B
        Ba[nx:nx+nu,0:nu] = identity(nu)
        system['B'][:,:,k] = Ba
        C0a = zeros([nxa, szC0])
        C0a[0:nx,0:szC0] = C0
        system['C0'][:,:,k] = C0a
        Ha = zeros([ny, nxa])
        Ha[0:ny,0:nx] = H
        system['H'][:,:,k] = Ha
        for j in range(D.shape[2]):
            Da = zeros([ny, nxa])
            #print(D.shape)
            #print(ny)
            #print(nxa)
            Da[0:ny,0:nx] = D[:,:,j]
            system['D'][:,:,j,k] = Da
        Qa = zeros([nxa, nxa])
        Qa[0:nx,0:nx] = Q
        Qa[0:nx,nx+nu] = q/2
        Qa[nx+nu,0:nx] = q/2
        Qa[nx:nx+nu,nx:nx+nu] = R
        Qa[nx:nx+nu,nx+nu] = r/2
        Qa[nx+nu,nx:nx+nu] = r/2
        Qa[-1,-1] = qs
        system['Q'][:,:,k] = Qa
        # Control costs are built into the augmented Q matrix, Qa so R=0.
        system['R'][:,:,k] = zeros([nu, nu])

    # last time point, use h(x) for the state cost 
    qs = h(x[:,N-1])
    dhdx = Jacobian(lambda x: h(x))
    q = dhdx(x[:,N-1])
    d2hdx2 = Hessian(lambda x: h(x))
    Q = d2hdx2(x[:,N-1])
    dldu = Jacobian(lambda u: l(x[:,N-2], u,k))
    r = dt*dldu(u[:,N-2])
    d2ldu2 = Hessian(lambda u: l(x[:,N-2], u,k))
    R = dt*d2ldu2(u[:,N-2])
    Qa = zeros([nxa, nxa])
    Qa[0:nx,0:nx] = Q
    Qa[0:nx,nx+nu] = q/2
    Qa[nx+nu,0:nx] = q/2
    Qa[nx:nx+nu,nx:nx+nu] = R
    Qa[nx:nx+nu,nx+nu] = r/2
    Qa[nx+nu,nx:nx+nu] = r/2
    Qa[-1,-1] = qs
    system['Q'][:,:,N-1] = Qa
    # iLQG does not accommodate noise added to the state estimate
    system['E0'] = zeros([1, 1, N])
    return system


def compute_cost(f, h, l, x, u):
    """ Compute the cost of the nominal trajectories. """
    #dt = 1.0  # until there's a reason to use something else
    cost = 0
    N = x.shape[1]
    for k in range(N-1):
        cost += l(x[:,k],u[:,k],k)*dt
        #x[:,k+1] = x[:,k] + f(x[:,k], u[:,k])*dt
    cost += h(x[:,-1])
    return cost


def update_trajectories(f, x_n, u_n, La):
    """ Update the nominal state and control trajectories to use for
    linearizing and quadratizing the system dynamics and costs. """
    #dt = 1.0  # until there's a reason to use something else
    N = La.shape[2] + 1
    #print(La.shape)
    nu = u_n.shape[0]
    u_p = zeros([nu, N-1])
    nx = x_n.shape[0]
    x_p = zeros([nx, N])
    x_p[:,0] = x_n[:,0]
    l = zeros([nu, N-1])
    L = zeros([nu, nx, N-1])
    #print(La.shape)
    Lu = zeros([nu, nu, N-1]) #nu,nx,N-1
    for k in range(N-1):
        x = x_p[:,k] - x_n[:,k]
        # parse La to get L and l
        L[:,:,k] = La[:,0:nx,k]
        l[:,k] = La[:,-1,k]
        #print(La[:,nx:nx+nu,k].shape)
        Lu[:,:,k] = La[:,nx:nx+nu,k]
        #print(La[:,nx:(nx+nu),k])
        if Lu.any() != 0:
            # If Lu != 0 then the control input for the current time step
            # depends on the control input for the previous time step. This
            # is not allowed in the control law, u(k) = l(k) + L(k)x(k)
            print "Lu is not zero!"
            import ipdb; ipdb.set_trace()
        u = -l[:,k] - L[:,:,k].dot(x)
        u_p[:,k] = u_n[:,k] + u
        x_p[:,k+1] = x_p[:,k] + f(x_p[:,k], u_p[:,k])*dt
    return x_p, u_p, L, l


def compute_trajectories(f, x0, l, L):
    # Compute the deterministic state and control trajectories using the
        #control law u(k) = l(k) + L(k)x(k).
    #dt = 1.0  # until there's a reason to use something else
    nu = L.shape[0]
    Nu = L.shape[2]
    u = zeros([nu, Nu])
    nx = x0.shape[0]
    x = zeros([nx, Nu + 1])
    x[:,0] = x0
    for k in range(Nu):
        u[:,k] = l[:,k] + L[:,:,k].dot(x[:,k])
        x[:,k+1] = x[:,k] + f(x[:,k], u[:,k])*dt
    return x, u


def compare_systems(system, previous_system):
    from numpy.linalg import norm
    tolerance = 1e-6
    N = system['R'].shape[2]
    if norm(system['X1'] - previous_system['X1']) > tolerance:
        print "X1 has changed"
    if norm(system['S1'] - previous_system['S1']) > tolerance:
        print "S1 has changed"
    for k in range(N):
        if norm(system['A'][:,:,k] - previous_system['A'][:,:,k]) > tolerance:
            print "A has changed"
            print previous_system['A'][:,:,k]
            print system['A'][:,:,k]
        if norm(system['B'][:,:,k] - previous_system['B'][:,:,k]) > tolerance:
            print "B has changed"
            print previous_system['B'][:,:,k]
            print system['B'][:,:,k]
        if norm(system['C0'][:,:,k] - previous_system['C0'][:,:,k]) > tolerance:
            print "C0 has changed"
            print previous_system['C0'][:,:,k]
            print system['C0'][:,:,k]
        if norm(system['H'][:,:,k] - previous_system['H'][:,:,k]) > tolerance:
            print "H has changed"
            print previous_system['H'][:,:,k]
            print system['H'][:,:,k]
        if norm(system['D0'][:,:,k] - previous_system['D0'][:,:,k]) > tolerance:
            print "D0 has changed"
            print previous_system['D0'][:,:,k]
            print system['D0'][:,:,k]
        if norm(system['Q'][:,:,k] - previous_system['Q'][:,:,k]) > tolerance:
            print "Q has changed at k =", k
            print system['Q'][:,:,k] - previous_system['Q'][:,:,k]
        if norm(system['R'][:,:,k] - previous_system['R'][:,:,k]) > tolerance:
            print "R has changed"
            print previous_system['R'][:,:,k]
            print system['R'][:,:,k]
    

def optimize_nominal_trajectories(f, h, l, x0, N):
    """ I hacked together a VERY simple (aka dumb) optimization strategy to
    find the nominal state and control trajectories that have the lowest cost.
    There is much room for improvement: 
    1. It does not make use of the f, l and h functions to direct the movement
    of the state trajectory.
    2. Currently its criteria for decreasing the size of the trajectory
    changes and for stopping altogether are arbitrary.
    3. This algo recomputes the entire control trajectory every time it moves a
    single point in the state trajectory even though it is only necessary to
    recalculate the control inputs before and after that modified point in
    state space.
    """
    nx = x0.shape[0]
    xf = zeros(nx)
    nu = 2
    x, u = initial_trajectories(f, x0, xf, nu, N)
    cost = compute_cost(f, h, l, x, u)
    print "Optimization initial cost:", cost
    S = 50
    costs = zeros([S,N])
    sigma = 100
    while sigma > 0.01:
        for i in range(S):
            # move each point in the state trajectory except the initial state
            for j in range(1,N):
                # move this point by a random amount
                current_x = array(x[:,j])
                x[:,j] += sigma*randn(nx)
                current_u = u
                u = compute_control_trajectory(f, x, nu)
                costs[i,j] = compute_cost(f, h, l, x, u)
                if costs[i,j] > cost:
                    x[:,j] = current_x
                    u = current_u
                else:
                    print ".",
                    cost = costs[i,j]
        sigma = 0.5*sigma
        print "Sigma:", sigma
    print "Optimization final cost:  ", cost
    return x, u


def iterative_lqg(f, F, g, G, h, l, x0, N, nu, xf=None):
    print(N)
    """ An implementation of Todorov's 2007 iterative LQG algorithm.  The
    system is described by these equations:
        dx = f(x,u)dt + F(x,u)dw(t)
        dy = g(x,u)dt + G(x,u)dv(t)
        J(x) = E(h(x(T)) + integral over t from 0 to T of l(t,x,u))
    Where T is N-1 times dt and J(x) is the cost to go.
    -> x0 is the initial state
    -> N is the number of points on the state trajectory
    -> nu is the number of elements in the control vector u
    -> xf is the approximate final state
    This algorithm returns state and control trajectories for a single run
    along with the state estimates and the filter and feedback matrices used to
    compute the state estimates and feedback controls.
    Because the system is, in general, non-linear the principal of certainty
    equivalence does not apply and K and L will change from run to run.
    """

    # start MATLAB engine, if using matlab_kalman_lqg
    #import matlab.engine
    #from test_kalman_lqg import matlab_kalman_lqg
    #eng = matlab.engine.start_matlab()

    nx = len(x0)
    if xf == None:
        xf = zeros(nx)
    # Generate the initial state and control trajectories.
    x_n, u_n = initial_trajectories(f, x0, xf, nu, N)
    # Compute the cost of the initial trajectories.
    cost = compute_cost(f, h, l, x_n, u_n)
    print "iLQG initial trajectory cost:", cost
    # Linearize and quadratize around (x=x_n, u=u_n).
    system = linearize_and_quadratize(f, F, g, G, h, l, x_n, u_n)

    # save the initial system for debugging
    initial_system = system

    K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
    #print "Calling inner loop"
    #K, L, Cost, Xa, XSim, CostSim, iterations = inner_loop(system)
    # try MATLAB code on this system to make sure it returns the same solution
    #K, L, Cost, Xa, XSim, CostSim, iterations = matlab_kalman_lqg(eng, system)
    #print(L.shape)
    print(L[:,:,1])
    has_not_converged = True
    iteration = 1
    while has_not_converged:

        if False:
            """# plot some figures
            figures = []
            script_name = re.split(r"/",sys.argv[0])[-1]
            output_file_name = script_name.replace(".py", ".html")
            output_file(output_file_name, title="")
            
            # plot the state and control trajectories
            ku = range(N-1)
            kx = range(N)
            ps = figure(title="State trajectory", x_axis_label='time',
                        y_axis_label='')
            #p.line(x_n[0,:], x_n[1,:], line_width=2, line_color="blue")
            ps.line(kx, x_n[0,:], line_width=2, line_color="blue")
            ps.line(kx, x_n[1,:], line_width=2, line_color="green")
            
            # plot the control trajectory
            pc = figure(title="Control trajectory", x_axis_label='time',
                        y_axis_label='')
            #pc.line(u_n[0,:], u_n[1,:], line_width=2, line_color="blue")
            pc.line(ku, u_n[0,:], line_width=2, line_color="blue")
            pc.line(ku, u_n[1,:], line_width=2, line_color="green")
            #figures.append([ps, pc])
            
            p = gridplot(figures)
            show(p)"""
            kx = range(N)
            p1 = plt.figure()
            plt.title("State Trajectories")
            axes = plt.gca()
            axes.set_xlabel('time')
            plt.plot(kx, x_n[0,:], linewidth=2, color="green",
                     linestyle='solid', label="x")
            plt.plot(kx, x_n[1,:], linewidth=2, color="blue",
                    linestyle='solid', label="y")
            plt.show()

        # Use the control policy from kalman_lqg to update the nominal state
        # and control trajectories.
        #print(L.shape)
        x_n, u_n, L_n, l_n = update_trajectories(f, x_n, u_n, L)
        previous_cost = cost
        cost = compute_cost(f, h, l, x_n, u_n)
        print "iLQG iteration %d trajectory cost: %f" % (iteration, cost)
        #print(x_n[0,:])
        #print(x_n[1,:])
        #if abs(cost / previous_cost - 1) < 1e-6:
        #if abs(cost - previous_cost) < 1:
        if (cost - previous_cost) > 0 or abs(cost - previous_cost) < 1:
            # convergence criteria has been met, yay!
            has_not_converged = False
        else:
            # Re-linearize the system dynamics and re-quadratize the system
            # costs along the new nominal trajectories.
            #print(x_n[0,:])
            #print(x_n[1,:])
            
            system = linearize_and_quadratize(f, F, g, G, h, l, x_n, u_n)
            # Update the feedback control law.
            #K, L, Cost, Xa, XSim, CostSim, iterations = inner_loop(system)
            K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
            #print(K.shape)
            #K, L, Cost, Xa, XSim, CostSim, iterations = \
            #        matlab_kalman_lqg(eng, system)
        iteration = iteration + 1

    # compare final system to initial system for debugging
    #final_system = system
    #print "Comparing final and initial systems"
    #compare_systems(final_system, initial_system)

    # exit MATLAB engine, if using matlab_kalman_lqg
    #eng.quit()
    #print(K.shape)
    return x_n, u_n, L_n, l_n, K[0:nx,:]


