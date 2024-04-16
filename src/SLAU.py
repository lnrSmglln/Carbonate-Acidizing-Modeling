import numpy as np
import numba

# old version
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
    try:
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
    except IndexError:
        print("N должна быть больше 1")
        x = f[0]
    return x

@numba.njit
def tridiagonal_1D_solver(a: np.ndarray, b: np.ndarray, c: np.ndarray, f: np.ndarray, x_L: float, x_R: float) -> np.ndarray:
    """Решение СЛАУ с трехдиагональной матрицей методом прогонки
    Пример типичной СЛАУ в матричном виде с N = 3 неизвестными:
    | -c_0  b_0   0  |   | x_0 |   | -f_0 - a_0 * x_L |
    |  a_0 -c_1  b_1 | * | x_1 | = |       -f_1       |
    |   0   a_1 -c_2 |   | x_2 |   | -f_2 - b_2 * x_R |
    тогда input должен выглядеть следующим образом
    a = [a_0, a_1, a_2]
    b = [b_0, b_1, b_2]
    c = [c_0, c_1, c_2]
    f = [f_0, f_1, f_2]
    x_L, x_R
    Необходимо диагональное преобладанее матрицы c_i >= a_i + b_i
    
    Args:
        a (numpy.ndarray): вектор a размерностью N
        b (numpy.ndarray): вектор b размерностью N
        c (numpy.ndarray): вектор c размерностью N
        f (numpy.ndarray): вектор f размерностью N
        x_L (float): гран. условие слева
        x_R (float): гран. условие справа

    Returns:
        numpy.ndarray: вектор неизвестных x размерностью N
    """

    N = c.shape[0]
    x = np.zeros_like(c)
    alpha = np.zeros_like(c)
    beta = np.zeros_like(c-1)

    # прямой ход
    
    alpha[0] = b[0] / c[0]
    beta[0] = (f[0] + a[0] * x_L) / c[0]
    for i in range(1, N-1):
        alpha[i] = b[i] / (c[i] - a[i] * alpha[i - 1])
        beta[i] = (a[i] * beta[i - 1] + f[i]) / (c[i] - a[i] * alpha[i - 1])
    
    # обратный ход
    x[N - 1] = (a[N - 1] * beta[N - 2] + f[N - 1] + b[N - 1] * x_R) / (c[N - 1] - a[N - 1] * alpha[N - 2])
    for i in range(N - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    
    return x