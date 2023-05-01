# # IMPORTS
import numpy as np
import matplotlib.pyplot as plt

class variables(object):
    def __init__(self, alpha=-1, beta=1, gamma=0.5, delta=0.3, omega=1.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega


def dudt(var, xu_i, t):
    return var.gamma * np.cos(var.omega * t) - var.delta * xu_i[1] - var.alpha * xu_i[0] - var.beta * xu_i[0] ** 3


def dxdt(xu_i):
    return xu_i[1]


def calc(var, inits, t_vec, dt):
    xu = inits
    for i, t in enumerate(t_vec):
        k1 = np.array([[dxdt(xu[i]), dudt(var, xu[i], t)]])
        k2 = np.array([[dxdt(xu[i] + dt/2 * k1[0]), dudt(var, xu[i] + dt/2 * k1[0], t + dt/2)]])
        k3 = np.array([[dxdt(xu[i] + dt/2 * k2[0]), dudt(var, xu[i] + dt/2 * k2[0], t + dt/2)]])
        k4 = np.array([[dxdt(xu[i] + dt * k3[0]), dudt(var, xu[i] + dt * k3[0], t + dt)]])

        xu_nxt = xu[i] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)
        xu = np.append(xu, xu_nxt, axis=0)

    plt.plot(t_vec, xu[:-1, 0])
    plt.plot(t_vec, xu[:-1, 1])
    plt.show()

    V_in = np.cos(var.omega * t_vec)

    return xu, V_in


def main():
    vars = variables()
    inits = np.array([[0, 0]])
    t_vec = np.linspace(0, 400, 60000)
    dt = t_vec[1] - t_vec[0]
    xu, V_in = calc(vars, inits, t_vec, dt)
    return t_vec, xu[:-1, 1], xu[:-1, 0], V_in



