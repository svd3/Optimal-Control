#!/usr/local/bin/python

from __future__ import print_function
import matlab.engine
from numpy import array, zeros, ones, identity, diag, real, stack, sqrt
from numpy import concatenate
from statistics import mean, stdev
from numpy.linalg import inv, eig, matrix_power, matrix_rank
from scipy.linalg import svd, norm, pinv, expm
import scipy.stats
from numpy.random import randn, rand, randint
from bokeh.plotting import figure, output_file, gridplot, vplot, show
import ipdb
import pickle
import time
from optimal_control import noise, compute_cost
from kalman_lqg import kalman_lqg, matlab_kalman_lqg


"""
This file contains the unit tests for kalman_lqg.py which is a port of
kalman_lqg.m obtained from Emo Todorov's website at the University of
Washington.
"""


def is_scalar(variable):
    """ Treat variable as a scalar if it is a float or an int. """
    return isinstance(variable, float) or isinstance(variable, int)


def equal(A, B):
    """ Determine if two matrices are equal. """
    return norm(A - B) < 1e-12


def save_system(system):
    """ Save the system dictionary passed to kalman_lqg so I can debug
    cases that fail. """
    with open("system.pkl", 'w') as file_handle:
        pickle.dump(system, file_handle)


def load_system():
    """ Load the system dictionary for failure cases from a file they can be
    debugged. """
    with open("system.pkl", 'r') as file_handle:
        system = pickle.load(file_handle)
    return system


def random_symmetric_matrix(n):
    """ return a random n by n symmetric matrix """
    M = randn(n, n)
    return 0.5*(M + M.T)


def random_symmetric_positive_definite_matrix(n):
    """ return a random n by n positive definite matrix """
    eigenvalues = rand(n) + 0.1
    R = randn(n, n)
    # a symmetric matrix has orthogonal eigenvectors, the matrix exponential of
    # an antisymmetric matrix is an orthogonal matrix
    V = real(expm(R - R.T))
    positive_definite_matrix = V.dot(diag(eigenvalues)).dot(V.T)
    evals, evecs = eig(positive_definite_matrix)
    return positive_definite_matrix


def random_symmetric_positive_definite_matrices(n, N):
    """ Generate N random m by n non-singular matrices. """
    k = 0
    matrix = random_symmetric_positive_definite_matrix(n)
    matrices = [matrix]
    while k < N:
        while equal(matrix, matrices[-1]) or min(eig(matrix)[0]) < 0.1:
            # make sure matrix entries only change by a few percent each
            # time step and that the matrix is still positive definite
            evals, evecs = eig(matrix)
            matrix = matrices[-1] * (1 + 1e-2*random_symmetric_matrix(n))
        matrices.append(matrix)
        k = k + 1
    if N == 1:
        matrices = matrices[0]
    else:
        matrices = stack(matrices, -1)
    return matrices


def controllability_matrix(A,B):
    """ Calculate the controllability matrix.  """
    na,ma = A.shape
    assert na == ma, "A must be a square matrix"
    nb,mb = B.shape
    assert nb == na, "B must have the same number of rows as A"
    C = zeros([na, nb*mb])
    for i in range(na):
        C[:,i*mb:(i+1)*mb] = matrix_power(A,i).dot(B)
    return C


def observability_matrix(A,H):
    """ Calculate the observability matrix.  """
    na,ma = A.shape
    assert na == ma, "A must be a square matrix"
    nh,mh = H.shape
    assert mh == ma, "H must have the same number of columns as A"
    obs = zeros([nh*mh, na])
    for i in range(na):
        obs[i*nh:(i+1)*nh,:] = H.dot(matrix_power(A,i))
    return obs


def random_nonsingular_matrices(m, n, N=1):
    """ Generate N random m by n non-singular matrices. """
    k = 0
    matrix = randn(m, n)
    while min(svd(matrix)[1]) < 0.1:
        matrix = randn(m, n)
    matrices = [matrix]
    while k < N:
        while equal(matrix, matrices[-1]) or min(svd(matrix)[1]) < 0.1:
            # make sure matrix is not close to singular and only change each
            # entry by a few percent between time steps
            matrix = matrices[-1] * (1 + 1e-2*randn(m, n))
        matrices.append(matrix)
        k = k + 1
    if N == 1:
        matrices = matrices[0]
    else:
        matrices = stack(matrices, -1)
    return matrices


def random_nonsingular_matrix(m, n):
    """ Generate a random m by n non-singular matrix. """
    return random_nonsingular_matrices(m, n, 1)


def random_stable_state_transition_matrix(n):
    """ Generate a random CLOSED LOOP state transition matrix that has no
    eigenvalues greater than 1. """
    A = randn(n,n)
    while min(svd(A)[1]) < 0.1:
        A = randn(n,n)
    eigenvalues, eigenvectors = eig(A)
    # if any eigenvalues have positive real parts, multiply them by -1
    signs = 2*((real(eigenvalues) <= 0).astype(int) - 0.5)
    new_eigenvalues = signs * eigenvalues
    # produce a new matrix using the new eigenvalues and old eigenvectors
    new_A = eigenvectors.dot(diag(new_eigenvalues)).dot(inv(eigenvectors))
    # the new matrix may have very small imaginary parts, set them to zero
    new_A = real(new_A)
    F = expm(new_A)
    return F


def random_kalman_lqg_system():
    """ Generate a random system for the optimal control algorithm implemented
    in Todorov's kalman_lqg.m MATLAB code. """
    N = 100          # duration in number of time steps
    
    # number of state variables
    nx = randint(2,5)

    # number of control inputs
    nu = randint(2,5)

    # number of observable outputs
    ny = randint(2,5)

    # number of additive process noise variables
    np = randint(2,5)

    # number of control dependent process noise varaibles
    npc = randint(2,5)

    # number of additive measurement noise variables
    nm = randint(2,5)

    # number of state dependent measurement noise variables
    nms = randint(2,5)

    # number of internal noise variables
    ni = randint(2,5)

    # scale factor for noise matrices
    noise_scale = 1e-1

    """ generate a random linear, time invariant, open loop system
    x(k+1) = A*x(k) + B*u(k) """
    # system dynamics matrix, A
    A = random_nonsingular_matrix(nx, nx)

    # control input matrix, B
    B = random_nonsingular_matrix(nx, nu)

    # control input dependent noise matrices, C
    C = noise_scale*randn(nu, nu, npc)

    # additive process noise matrix, C0
    C0 = noise_scale*randn(nx, np)
    
    # measurement matrix, H 
    H = randn(ny, nx)

    # state dependent measurement noise matrices, D
    D = noise_scale*randn(ny, nx, nms)

    # additive measurement noise matrix, D0
    D0 = noise_scale*randn(ny, nm)
    
    # internal noise that directly affects the state estimate
    # zero in LQG systems
    E0 = noise_scale*randn(nx, ni)
    # pick a random initial state and initial covariance matrix
    X1 = randn(nx, 1)
    S1 = identity(nx) 

    # pick random state and control cost matrices
    Q = random_symmetric_positive_definite_matrix(nx)
    Q = stack([Q for k in range(N)], -1)  # copy Q for each time step
    R = random_symmetric_positive_definite_matrix(nu)
    system = {'A': A, 'B': B, 'C': C, 'C0': C0, 'H': H, 'D': D,
              'D0': D0, 'E0': E0, 'Q': Q, 'R': R, 'X1': X1, 'S1': S1}
    return system


def random_time_varying_kalman_lqg_system():
    """ Generate a random time-varying system for the optimal control algorithm
    implemented in Todorov's kalman_lqg.m MATLAB code. """
    N = 100          # duration in number of time steps
    
    # number of state variables
    nx = randint(2,5)

    # number of control inputs
    nu = randint(2,5)

    # number of observable outputs
    ny = randint(2,5)

    # number of additive process noise variables
    np = randint(2,5)

    # number of control dependent process noise varaibles
    npc = randint(2,5)

    # number of additive measurement noise variables
    nm = randint(2,5)

    # number of state dependent measurement noise variables
    nms = randint(2,5)

    # number of internal noise variables
    ni = randint(2,5)

    # scale factor for noise matrices
    noise_scale = 1e-1

    """ generate a random non-linear open loop system
    x(k+1) = A*x(k) + B*u(k) """
    # system dynamics matrix, A
    A = random_nonsingular_matrices(nx, nx, N-1)

    # control input matrix, B
    B = random_nonsingular_matrices(nx, nu, N-1)

    # control input dependent process noise matrices, C
    C = noise_scale*randn(nu, nu, npc, N-1)

    # additive process noise matrix, C0
    C0 = noise_scale*randn(nx, np, N-1)
    
    # measurement matrix, H 
    H = random_nonsingular_matrices(ny, nx, N)

    # state dependent measurement noise matrices, D
    D = noise_scale*randn(ny, nx, nms, N)

    # additive measurement noise matrix, D0
    D0 = noise_scale*randn(ny, nm, N)
    
    # internal noise that directly affects the state estimate
    # zero in LQG systems
    E0 = noise_scale*randn(nx, ni, N-1)

    # pick a random initial state and initial covariance matrix
    X1 = randn(nx, 1)
    S1 = identity(nx) 

    # pick random state and control cost matrices
    Q = random_symmetric_positive_definite_matrices(nx, N)
    #Q = stack([Q for k in range(N)], -1)  # copy Q for each time step
    R = random_symmetric_positive_definite_matrices(nu, N-1)
    system = {'A': A, 'B': B, 'C': C, 'C0': C0, 'H': H, 'D': D,
              'D0': D0, 'E0': E0, 'Q': Q, 'R': R, 'X1': X1, 'S1': S1}
    return system


def perturb(matrix_trajectory, scale):
    """ Perturb the given matrix trajectory with Gaussian noise scaled by
    scale.  """
    perturbed_trajectory = matrix_trajectory.flatten()
    for i in range(len(perturbed_trajectory)):
        perturbed_trajectory[i] += scale*randn()
    return perturbed_trajectory.reshape(matrix_trajectory.shape)


def generate_kalman_lqg_regression_test_cases():
    """
    5/27/2016 James Bridgewater
    I'm writing this function to generate test cases by picking random linear
    time-invariant systems and quadratic cost functions along with control and
    state dependent noise that fit into the framework from Emo Todorov's
    2005 Neural Computation paper.  I pass these to the MATLAB code he
    published to get the state esimation and feedback control matrices and save
    them along with the system description for use in regression tests that
    I will be using on the python code I'm developing as a behavioral model for
    animals performing foraging tasks.  This code will start out as a straight
    port of Todorov's code and then be extended for use with time-varying and
    non-linear cases.
    """
    number_of_test_cases = 100
    test_cases = []
    eng = matlab.engine.start_matlab()
    test_case_number = 0
    while test_case_number < number_of_test_cases:
        test_case_number = test_case_number + 1
        print("Test case #: ", test_case_number)
        system = random_kalman_lqg_system()
        #function [K,L,Cost,Xa,XSim,CostSim,iter] = ...
        K, L, Cost, Xa, XSim, CostSim, iterations = \
                matlab_kalman_lqg(eng, system)
    
        if iterations < 500:
            """ Keep the results if the algorithm converged before stopping at
            the maximum number of iterations. """
            solution = {'K': K, 'L': L, 'Cost': Cost,
                        'Xa': Xa}
            test_case = {'system': system, 'solution': solution}
            test_cases.append(test_case)
    
    # Save the test cases
    with open("kalman_lqg_test_cases.pkl", 'w') as file_handle:
        pickle.dump(test_cases, file_handle)
    eng.quit()
    

def generate_time_varying_regression_test(DEBUG=False):
    """
    The idea here is to generate a random linear time-varying LQG
    system along with a random initial state and test the feedback and
    filter matrices returned by the iterative algorithm in kalman_lqg.py by
    perturbing them and making sure that the perturbed matrices always produce
    higher control costs and estimation errors than the originals.  This
    provides strong evidence that the algorithm in kalman_lqg.py is returning
    the optimal solution.  Once I believe the code is producing the optimal
    result for a test case it is added to the regression tests.
    """
    number_of_test_cases = 10
    test_cases = []
    test_case_number = 0
    while test_case_number < number_of_test_cases:
        test_case_number = test_case_number + 1
        if DEBUG:
            system = load_system()
            print("System loaded for debugging")
            ipdb.set_trace()
        else:
            system = random_time_varying_kalman_lqg_system()
            print("System generated")
    
        #############################################################
        # Call kalman_lqg
        #############################################################
        
        #K,L,Cost,Xa,XSim,CostSim = \
        return_values = kalman_lqg(system)
        K = array(return_values[0])
        L = array(return_values[1])
        Cost = array(return_values[2])
        Xa = array(return_values[3])
        XSim = array(return_values[4])
        CostSim = array(return_values[5])
        # I added this return value to make sure the algorithm converged
        iterations = array(return_values[6])
        if iterations == 500:
            # didn't converge, hit max iterations try another system
            continue
        else:
            print("System solved")
        
        """
        Calculate NSim from the difference between the means and the larger of the
        two mean estimate uncertainties.
        """
        NSim = 100 # number of simulations per batch
        timed_out = False # define so if timed_out doesn't fail on first loop
        N = 0 # define N so if N == NSim doesn't fail on first loop
        perturbation_size = 0.1
        # see if suboptimal matrices fail
        #K = perturb(K, 2*perturbation_size)
        #L = perturb(L, 2*perturbation_size)
        perturbation = 0
        costs_too_large = False
        while perturbation < 40 and not costs_too_large:
            # perturb the state trajectory and make sure the expected cost is higher
            if timed_out:
                # increase perturbation size
                perturbation_size *= 2
                print("perturbation_size", perturbation_size)
            if N == NSim:
                # decrease perturbation size
                perturbation_size *= 0.5
                print("perturbation_size", perturbation_size)
            suboptimal_K = perturb(K, perturbation_size)
            suboptimal_L = perturb(L, perturbation_size)
            N = 0
            optimal_costs = []
            suboptimal_costs = []
            time0 = time.time()
            timed_out = False
            confidence_level_met = False
            while not confidence_level_met and not timed_out:
                costs = compute_cost(system, K, L, NSim=NSim)
                optimal_costs = concatenate((optimal_costs, costs))
                optimal_cost = mean(optimal_costs)
                optimal_std = stdev(optimal_costs)
                costs = compute_cost(system, suboptimal_K, suboptimal_L,
                                     NSim=NSim)
                suboptimal_costs = concatenate((suboptimal_costs, costs))
                try:
                    suboptimal_cost = mean(suboptimal_costs)
                    suboptimal_std = stdev(suboptimal_costs)
                except (OverflowError, AttributeError):
                    # Catch errors caused by costs becoming very large as
                    # perturbation_size is increased in the hopes of meeting
                    # the confidence criteria.
                    costs_too_large = True
                    # try another system
                    break
                N = N + NSim
                print("N = ", N)
                print("optimal cost: ", optimal_cost,
                      "optimal std: ", optimal_std)
                print("suboptimal cost: ", suboptimal_cost,
                      "suboptimal std: ", suboptimal_std)
                # Construct a random variable that is the difference between the
                # uncertain mean costs and calculate the probability that this
                # random variable is less than zero. This corresponds to the
                # probability that the means are deceiving us as to which
                # trajectory produces the lower expected cost.
                difference_in_means = abs(optimal_cost - suboptimal_cost)
                std_of_diff = sqrt((optimal_std**2 + suboptimal_std**2)/N)
                prob_diff_is_less_than_zero = \
                        scipy.stats.t(N-1).cdf(-difference_in_means/std_of_diff)
                        #scipy.stats.norm(difference_in_means,std_of_diff).cdf(0)
                print(prob_diff_is_less_than_zero)
                if prob_diff_is_less_than_zero < 1e-3:
                    confidence_level_met = True
                if time.time() - time0 > 20: # seconds
                    timed_out = True
            if confidence_level_met:
                perturbation = perturbation + 1
                print("perturbation #%d" % perturbation)
                if suboptimal_cost < optimal_cost:
                    """ Something is broken, save the system so we can debug it. """
                    save_system(system)
                    print("The state trajectory is not optimal!")
                    #print("Perturbation at L[%d,%d,%d]" % (row, col, t))
                    print("Suboptimal cost - optimal cost: %.15f" %
                          (suboptimal_cost - optimal_cost))
                assert suboptimal_cost >= optimal_cost
        # save test cases that pass
        solution = {'K': K, 'L': L, 'Cost': Cost, 'Xa': Xa}
        test_case = {'system': system, 'solution': solution}
        test_cases.append(test_case)
        print("Finished with test case #: ", test_case_number)

    with open("time_varying_test_cases.pkl", 'w') as file_handle:
        pickle.dump(test_cases, file_handle)


def test_kalman_lqg():
    """
    5/27/2016 James Bridgewater
    Creating this function as a regression test to compare results from
    kalman_lqg.py to results from kalman_lqg.m.
    """
    # test paramters
    tolerance = 1e-5  # 10 parts per million

    # Load the saved test cases
    with open("kalman_lqg_test_cases.pkl", 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_counter = 0
    for test_case in test_cases:
        test_counter = test_counter + 1
        print("Test case #: ", test_counter)
        system = test_case['system']
        K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
        solution = test_case['solution']
        mat_K = solution['K'].flatten()
        py_K = K.flatten()
        for i in range(len(mat_K)):
            if abs(mat_K[i] - py_K[i]) > abs(tolerance*mat_K[i]):
                print(mat_K[i])
                print(py_K[i])
            assert abs(mat_K[i] - py_K[i]) < abs(tolerance*mat_K[i]), \
                    "K is not within tolerance"
        mat_L = solution['L'].flatten()
        py_L = L.flatten()
        for i in range(len(mat_L)):
            assert abs(mat_L[i] - py_L[i]) < abs(tolerance*mat_L[i]), \
                    "L is not within tolerance"
        mat_Xa = solution['Xa'].flatten()
        py_Xa = Xa.flatten()
        for i in range(len(mat_Xa)):
            assert abs(mat_Xa[i] - py_Xa[i]) < abs(tolerance*mat_Xa[i]), \
                    "Xa is not within tolerance"
    

def test_time_varying_kalman_lqg():
    """
    6/17/2016 James Bridgewater
    Creating this function as a regression test to compare results from
    kalman_lqg.py to results it produced previously that I believe to be
    optimal because they passed perturbation testing.
    """
    # test paramters
    tolerance = 1e-9  # 1 part per billion

    # Load the saved test cases
    with open("time_varying_test_cases.pkl", 'r') as file_handle:
        test_cases = pickle.load(file_handle)
    test_counter = 0
    for test_case in test_cases:
        test_counter = test_counter + 1
        print("Test case #: ", test_counter)
        system = test_case['system']
        K, L, Cost, Xa, XSim, CostSim, iterations = kalman_lqg(system)
        solution = test_case['solution']
        mat_K = solution['K'].flatten()
        py_K = K.flatten()
        for i in range(len(mat_K)):
            if abs(mat_K[i] - py_K[i]) > abs(tolerance*mat_K[i]):
                print(mat_K[i])
                print(py_K[i])
            assert abs(mat_K[i] - py_K[i]) < abs(tolerance*mat_K[i]), \
                    "K is not within tolerance"
        mat_L = solution['L'].flatten()
        py_L = L.flatten()
        for i in range(len(mat_L)):
            assert abs(mat_L[i] - py_L[i]) < abs(tolerance*mat_L[i]), \
                    "L is not within tolerance"
        mat_Xa = solution['Xa'].flatten()
        py_Xa = Xa.flatten()
        for i in range(len(mat_Xa)):
            assert abs(mat_Xa[i] - py_Xa[i]) < abs(tolerance*mat_Xa[i]), \
                    "Xa is not within tolerance"
    

if __name__ == "__main__":
    #test_kalman_lqg()
    test_time_varying_kalman_lqg()
    #generate_time_varying_regression_test()

