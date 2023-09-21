import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Решение системы дифуров:
# dy1/dt = -2y1 + y2 + y3
# dy2/dt = y1 - 2y2 + 2y3
# dy3/dt = 2y1 - y2 - 3y3

def system_of_equations(y, t):
    dydt = np.zeros(3)
    dydt[0] = -2 * y[0] + y[1] + y[2]
    dydt[1] = y[0] - 2 * y[1] + 2 * y[2]
    dydt[2] = 2 * y[0] - y[1] - 3 * y[2]
    return dydt


if __name__ == '__main__':
    y0 = [1.0, 0.0, 0.0]  # начальные условия: y1(0) = 1, y2(0) = 0, y3(0) = 0
    t = np.linspace(0, 10, 100)  # массив времени от 0 до 10 с шагом 0.1

    solution = odeint(system_of_equations, y0, t)

    y1 = solution[:, 0]
    y2 = solution[:, 1]
    y3 = solution[:, 2]

    plt.figure(figsize=(10, 6))
    plt.plot(t, y1, label='y1(t)')
    plt.plot(t, y2, label='y2(t)')
    plt.plot(t, y3, label='y3(t)')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid()
    plt.show()
