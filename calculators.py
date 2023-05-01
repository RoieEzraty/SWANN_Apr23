# IMPORTS
# general
import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt

# my codes
import extras

"""
Different calculators used for go_chaos_simple

Roie 2021
"""


def build_theta(P, poly_descr):
    """
    built_theta builds large matrix theta whose pseudo inverse will be taken to find coefficients such that
    theta*coefficients = d/dt of dynamics (namely dV_R/dt)

    :param P: class with all parameter and variables needed for data including polynomial terms like Q^2 and V_R*V_in
    :param poly_descr: list of strings - verbal description of the polynomial, namely what are the variables to fit upon
    :return: theta: the matrix of all data - rows = different times, cols = different variables
    """

    theta = np.empty([P.length, len(poly_descr)])
    # theta[:, 0] = (1+P.t)/(1+P.t)  # first col has to be zeros
    for i, val in enumerate(poly_descr):
        if val == 'ones':
            vec = P.ones
        elif val == 'V_in':
            vec = P.V_in
        elif val == 'V_R':
            vec = P.V_R
        elif val == 't':
            vec = P.t
        elif val == 'Q':
            vec = P.Q
        elif val == 'V_in2':
            vec = P.V_in2
        elif val == 'V_in3':
            vec = P.V_in3
        elif val == 'V_R2':
            vec = P.V_R2
        elif val == 'V_R3':
            vec = P.V_R3
        elif val == 'Q2':
            vec = P.Q2
        elif val == 'Q3':
            vec = P.Q3
        elif val == 'V_RQ':
            vec = P.V_R * P.Q
        elif val == 'V_inQ':
            vec = P.V_in * P.Q
        elif val == 'V_RQ2':
            vec = P.V_R * P.Q2
        elif val == 'V_inQ2':
            vec = P.V_in * P.Q2
        elif val == 'V_in2Q':
            vec = P.V_in2 * P.Q
        elif val == 'V_R2Q':
            vec = P.V_R2 * P.Q
        elif val == 'V_RV_in':
            vec = P.V_R * P.V_in
        elif val == 'V_R2V_in':
            vec = P.V_R2 * P.V_in
        elif val == 'V_RV_in2':
            vec = P.V_R * P.V_in2
        else:
            print('build theta probably missing a variable')
            vec = 0

        theta[:, i] = vec

    return theta


def find_coeffs_thru_pinv(theta, V_Rt):
    """
    coefficients of PDE through pseudo inverse of theta
    under coefficients = dV_T/dt * theta+ (theta+ is pseudo inverse of theta)
    :param theta: matrix of all the data variables
    :param V_Rt: derivative of the voltage on the resistor dV_R/dt
    :return: coeffs: the calculated coefficients
    """

    theta_inv = lin.pinv(theta)
    coeffs = np.matmul(theta_inv, V_Rt)
    return coeffs


def find_V_R_theor(coeffs, theta_desc, P, weight, scheme='normal'):
    """
    calculate theoretical (integrated) dynamics using the coefficients found and initial conditions of measured data

    :param coeffs: coefficients found with find_coeffs_thru_pinv
    :param theta_desc: a list of strings describing which variable is in each column of theta
    :param P: class containing all parameters / variables of data
    :param weight: for calculation of Q using calculators.calc_Q_w_weight
    :param scheme: scheme of integration - 'normal' is the simple Euler and 'RK4' is 4th order Runge-Kutta
    :return: V_R_theor: numpy array of integrated (theoretical) dynamics of voltage on resistor using the found
                        coefficints and the measured inital conditions
    """

    if scheme == 'RK4':
        V_R_theor = extras.integratorRK4(P, coeffs, theta_desc)
    elif scheme == 'normal':
        V_R_theor = extras.integrator4(P, coeffs, theta_desc, weight)
    else:
        V_R_theor = np.nan
    return V_R_theor


def calc_norm_coeffs_vec(P, coeffs, coeffs_desc):
    """
    calc_norm_coeffs_vec gets a vector of floats representing the coefficients of the different variables of the PDE,
    and the dynamics of all variables for the given time period. The function output a normalized vector of coefficients
    by multiplying each coefficient by the mean of the absolute value of the dynamics in that time period.
    :param P: class of variable dynamics for the given time period
    :param coeffs: vector of floats
    :param coeffs_desc: list of strings describing which variable is assigned to which coefficient
    :return: norm_coeffs_vec: vector of floats representing the normalized coefficients from coeffs
    """
    norm_vec = np.empty([len(coeffs_desc)])
    # for each coefficient - get the variable dynamics from P and normalize accordingly
    for j, val2 in enumerate(coeffs_desc):
        if val2 == 'ones':
            norm_vec[j] = 1
        else:
            vec = getattr(P, val2)
            norm_vec[j] = np.mean(abs(vec))

    norm_coeffs_vec = coeffs * norm_vec
    # normalize and take abs so later norm_coeffs_vec < cutoff = 0 makes sense
    norm_coeffs_vec = abs(norm_coeffs_vec) / lin.norm(norm_coeffs_vec)
    return norm_coeffs_vec


def calc_Q(V_R, dt):
    """
    calc_Q calculates the accumulating charge using Q(index=i) = Q(index=i-1) + V_R(index=i) * dt(index=i)
    :param V_R: vector of floats representing voltage on resistor
    :param dt: vector of floats representing time differences
    :return: Q: vector of floats representing charge Q in time
    """

    Q = np.empty([len(V_R)])
    for i, val in enumerate(V_R):
        if i == 0:
            Q[i] = 0
        else:
            Q[i] = Q[i-1] + V_R[i]*dt[i]
    return Q


def calc_Q_w_weight(V_R, dt, weight):
    """
    calc_Q_w_weight calculates charge Q with weight over the previous Q's
    such that the voltage in the current time step is added to the previous Q which is multiplied by a number <= 1
    so that the current time step voltage has more impact on the charge than previous times
    :param V_R: vector of floats representing voltage on resistor
    :param dt: vector of floats representing time differences
    :param weight: float between 0 and 1 (including)
                   0 will result in Q = V_R * time step
                   1 will result in Q as in calc_Q (without weight)
    :return: Q: vector of floats representing charge Q in time
    """

    Q = np.empty([len(V_R)])
    for i, val in enumerate(V_R):
        if i == 0:
            Q[i] = 0
        else:
            Q[i] = weight * Q[i-1] + V_R[i]*dt[i]
    return Q


def calc_coeffs_guess(data_frame):
    """
    return a good guess for the coefficients from a spread achieved from different initializations
    as if the coefficients were from a normal distribution
    :param data_frame:        DataFrame where all the guessed data is
    :return: coeffs_guess:    DataFrame of guessed coeffs
    """
    coeffs_guess = data_frame.div(data_frame['loss'], axis='rows').mean() * data_frame['loss'].mean()
    return coeffs_guess



def calc_dynamics_from_tent_map(tent_params, init, length):
    """
    explanation
    :param tent_params: explanation
    :param init: explanation
    :param length: explanation
    :return: ret_tent: explanation
    """

    f_l = lambda x: tent_params.a_l * x + tent_params.b_l
    f_r = lambda x: tent_params.a_r * x + tent_params.b_r

    ret_tent = np.empty([length + 1, 2])
    ret_tent[0, 0] = init

    for i in range(length):
        x = ret_tent[i, 0]
        if x > tent_params.x_cross:
            ret_tent[i+1, 0] = f_r(x)
        else:
            ret_tent[i+1, 0] = f_l(x)

    # now in the form of return map
    ret_tent[:-1, 1] = ret_tent[1:, 0]
    np.delete(ret_tent, length, 0)
    return np.transpose(ret_tent)


# # NOT IN USE

# Calculate Q with memory of the previous Q
# such that the previous Q integral is added to a zero mean new integration
# def calc_Q_w_memory(P):
#     V_R_tot = np.concatenate((P.V_R_memory, P.V_R))
#     # length = len(V_R)
#     # sigmoid[int(len(V_R_tot)/2):] = 1
#     Q_tot = np.convolve(V_R_tot, P.sigmoid) * P.dt
#     Q = Q_tot[len(P.V_R):2*len(P.V_R)]
#     # Q -= np.mean(Q)
#     return Q


