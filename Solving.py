import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#  Определение параметров и функций
# Доля изымаемого у крестьян продукта
a = 0.0013
# Коэффициент расходов для государства
Ax = 0.3
# Коэффициент потребления для крестьян
Ay = 0.7
# Затраты правящего классса на управление и порядок
C1 = 0.1
C2 = C1
# Параметр, описывающий, как экономическое состояние государства влияет на производительность труда
b = 0.5
# Показатель демографической культуры
r = 0.1
# Характеризует пороговое значение экономического состояния государства, при котором производительность труда достигает максимума
X0 = 1000
# Средним накоплениям материальных средств у одного крестьянина в начальный момент времени
Y0 = 0.5
# Изначальная численность крестьян
N0 = 5000
# Максимальная площадь, которую способен обрабатывать один крестьянин
R0 = 1
# Количество крестьян, соответствующее ситуации полного сельскохозяйственного освоения пригодных для обработки земель
N_optimal = 3000
T = 100
T_split = 1000


N_minimal = 1000


# Функция вклада государства в производительность труда
def f(X):
    return 1 + (b * (X ** 2) / (X0 ** 2 + X ** 2))


# Функция площади обрабатываемой одним крестьянином земли
def R(N):
    if N <= N_optimal:
        return R0
    else:
        return R0 * N_optimal / N


# Функция прироста населения
def D(Y):
    return r * (1 - notMoreThan(Y0 / Y, 1))


def notMoreThan(value, threshold):
    return value #if value < threshold else threshold


# Суммарное количество продукта, изымаемое государством у крестьян в единицу времени (например, налоги и поборы).
def G(X, Y, N):
    actual_a = 100 * a if X < 0 else a
    # value = actual_a * Y * N
    value = a * X * Y * N
    print(f'G(XYN) = {value}, a={actual_a}')
    return value


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
    value = f(X) * R(N) * N
    print(f'F(XYZ) = {value}')
    return value


# Система дифференциальных уравнений
def model(parameters, t):
    X, YN, N = parameters
    Y = YN / N
    dXdt = G(X, Y, N) - Q_x(X) - C(X, Y, N)
    dYNdt = F(X, Y, N) - N * Q_y(Y) - G(X, Y, N)
    dNdt = N * D(Y)
    equations = np.zeros(3)
    equations[0] = dXdt
    equations[1] = dYNdt
    equations[2] = dNdt
    return equations


if __name__ == '__main__':
    # Начальные условия
    parameters0 = [X0, Y0 * N0, N0]

    # Время
    t = np.linspace(0, T, T_split)

    # Решение системы уравнений
    solution = odeint(model, parameters0, t, full_output=True)[0]

    # Извлечение результатов
    X, YN, N = solution[:, 0], solution[:, 1], solution[:, 2]

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.plot(t, X, label='X (государство)')
    plt.plot(t, YN, label='YN (крестьяне)')
    plt.plot(t, N, label='N (численность крестьян)')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.legend()
    plt.grid()
    plt.show()
