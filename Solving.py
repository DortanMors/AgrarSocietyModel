import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Определение параметров и функций
a = 2.25
Ax = 0.2
Ay = 0.3
C1 = 2.5
C2 = C1
b = 101
r = 1
X0 = 100
Y0 = 500
N0 = 500
R0 = 10
N_optimal = 1000


# Функция f(X)
def f(X):
    return 1 + (b * (X ** 2) / (X0 ** 2 + X ** 2))


# Функция R(N)
def R(N):
    if N <= N_optimal:
        return R0
    else:
        return R0 * N_optimal / N


# Функция D(Y)
def D(Y):
    return r * (1 - Y0 / Y)


def G(X, Y, N):
    return a * X * Y * N


# Ф-ция потребления государства
def Q_x(X):
    return Ax * X


# Ф-ция потребления крестьян
def Q_y(Y):
    return Ay * Y


# Ф-ция затрат на управление
def C(X, Y, N):
    return (C1 / Y + C2 * Y) * N


# производственная функция, характеризующая совокупное сельскохозяйственное производство в государстве
def F(X, Y, N):
    return f(X) * R(N) * N


# Система дифференциальных уравнений
def model(y, t):
    X, Y, N = y
    dXdt = G(X, Y, N) - Q_x(X) - C(X, Y, N)
    dYdt = F(X, Y, N) - N * Q_y(Y) - G(X, Y, N)
    dNdt = N * D(Y)
    equations = np.zeros(3)
    equations[0] = dXdt
    equations[1] = dYdt
    equations[2] = dNdt
    return equations


if __name__ == '__main__':
    # Начальные условия
    y0 = [X0, Y0, N0]

    # Время
    t = np.linspace(0, 10, 100)

    # Решение системы уравнений
    solution = odeint(model, y0, t, full_output=True)[0]

    # Извлечение результатов
    print(solution)
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
