import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Определение параметров и функций
a = 0.1
Ax = 0.2
Ay = 0.3
C1 = 0.1
C2 = 0.2
b = 0.01
X0 = 100
R0 = 10
N_optimal = 1000


# Функция f(X)
def f(X):
    return 1 + b * (X ** 2) / (X0 ** 2 + X ** 2)


# Функция R(N)
def R(N):
    if N <= N_optimal:
        return R0
    else:
        return R0 * N_optimal / N


# Функция D(Y)
def D(Y):
    return 0.01 * Y


# Система дифференциальных уравнений
def model(y, t):
    X, Y, N = y
    dXdt = a * X * Y * N - Ax * X - (C1 / Y + C2 * Y) * N
    dYdt = f(X) * R(N) * N - Ay * Y - a * X * Y * N
    dNdt = N * D(Y)
    equations = np.zeros(3)
    equations[0] = dXdt
    equations[1] = dYdt
    equations[2] = dNdt
    return equations


if __name__ == '__main__':
    # Начальные условия
    X0 = 50
    Y0 = 20
    N0 = 500
    y0 = [X0, Y0, N0]

    # Время
    t = np.linspace(0, 10, 100)

    # Решение системы уравнений
    solution = odeint(model, y0, t, full_output=True)[0]

    # Извлечение результатов
    X, Y, N = solution[:, 0], solution[:, 1], solution[:, 2]

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.plot(t, X, label='X (государство)')
    plt.plot(t, Y, label='Y (крестьяне)')
    plt.plot(t, N, label='N (численность крестьян)')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.legend()
    plt.grid()
    plt.show()
