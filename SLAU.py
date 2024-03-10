import numpy as np

def tridiagonal_solve(a, b, c, f):
    """Решение СЛАУ с трехдиагональной матрицей методом прогонки
    Пример типичной СЛАУ в матричном виде с N = 3 неизвестными:
    | -c_0  b_0   0  |   | x_0 |   | -f_0 |
    |  a_0 -c_1  b_1 | * | x_1 | = | -f_1 |
    |   0   a_1 -c_2 |   | x_2 |   | -f_2 |
    тогда input должен выглядеть следующим образом
    a = [a_0, a_1]
    b = [b_0, b_1]
    c = [c_0, c_1, c_2]
    f = [f_0, f_1, f_2]
    Необходимо диагональное преобладанее матрицы c_i >= a_i + b_i
    
    Args:
        a (numpy.ndarray): вектор a размерностью N - 1
        b (numpy.ndarray): вектор b размерностью N - 1
        c (numpy.ndarray): вектор c размерностью N
        f (numpy.ndarray): вектор f размерностью N

    Returns:
        numpy.ndarray: вектор неизвестных x размерностью N
    """

    N = len(c)
    x = np.zeros_like(c)
    alpha = np.zeros_like(c)
    beta = np.zeros_like(c)

    # прямой ход
    alpha[0] = b[0] / c[0]
    beta[0] = f[0] / c[0]
    for i in range(1, N-1):
        alpha[i] = b[i] / (c[i] - a[i - 1] * alpha[i - 1])
        beta[i] = (a[i - 1] * beta[i - 1] + f[i]) / (c[i] - a[i - 1] * alpha[i - 1])
    
    beta[N - 1] = (a[N - 2] * beta[N - 2] + f[N - 1]) / (c[N - 1] - a[N - 2] * alpha[N - 2])
    
    # обратный ход
    x[N - 1] = beta[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x