import numpy as np
import numba
import matplotlib.pyplot as plt

WATER_CURVE_SHAPE_PARAMETER = 2.5
OIL_CURVE_SHAPE_PARAMETER = 2
S_WIR = 0.
S_ORW = 0.

# Corey-model
@numba.njit
def normalized_s(S:float) -> float:
    return (S - S_WIR) / (1 - S_WIR - S_ORW)

@numba.njit
def k1(S:float) -> float:
    """Relative permeability to water

    Args:
        S (float): Water saturation
        n (float): n power

    Returns:
        float: Relative permeability to water
    """
    if S <= S_WIR:
        return 0.0
    elif S >= (1 - S_ORW):
        return 1.0
    else:
        return np.power(normalized_s(S), WATER_CURVE_SHAPE_PARAMETER)

@numba.njit
def k2(S:float) -> float:
    """Relative permeability to oil

    Args:
        S (float): Water saturation
        n (float): n power

    Returns:
        float: Relative permeability to oil
    """
    if S <= S_WIR:
        return 1.0
    elif S >= (1 - S_ORW):
        return 0.0
    else:
        return np.power((1 - normalized_s(S)), OIL_CURVE_SHAPE_PARAMETER)

@numba.njit
def b(S:float, viscosity_1:float, viscosity_2:float) -> float:
    """Buckley-Leverett function calculation

    Args:
        S (float): Water saturation
        viscosity_1 (float): Water viscosity
        viscosity_2 (float): Oil viscosity

    Returns:
        float: Buckley-Leverett function of S
    """
    return k1(S) / (k1(S) + viscosity_1 / viscosity_2 * k2(S))

def plot_buckley_leverett(viscosity_1:float, viscosity_2:float, return_Sfront = False, return_dfs = False):
    """Plotting Buckley-Leverett main graphs and get [S]

    Args:
        viscosity_1 (float): _description_
        viscosity_2 (float): _description_
        return_Sfront (bool, optional): _description_. Defaults to False.
        return_dfs (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Графики Баклея-Леверетта
    ds = 0.01
    s = np.arange(S_WIR, (1-S_ORW) + ds, ds)
    f_s = np.zeros_like(s)
    for i in range(len(s)):
        f_s[i] = b(s[i], viscosity_1, viscosity_2)
    df_s = np.gradient(f_s) * len(s)
    s_center = s[np.where(np.isclose(df_s, np.max(df_s)))][0]

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout='constrained', dpi=150)
    # fig.suptitle("")

    axs[0].set_title("ОФП")
    axs[0].grid()
    axs[0].plot(s, [k1(i) for i in s], s, [k2(i) for i in s])
    axs[0].legend(["Water", "Oil"])
    axs[0].set_xlim([0-0.05, 1+0.05])

    axs[1].plot(s, f_s)
    axs[1].grid()
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].set_title("Ф-я Баклея-Леверетта")
    # построение касательных к функции Б-Л
    k_right = 0
    k_left = 0
    s_right = 0
    s_left = 0
    for k in np.arange(max(df_s), min(df_s), -ds):
        if np.min(k * (s - S_WIR) - f_s) < 0:
            k_right = k
            s_right = np.max(s * ((k * (s - S_WIR) - f_s) < 0))
            break
    axs[1].plot(s, k_right * (s - S_WIR), color="black", alpha=0.5, linewidth=0.7)
    axs[1].plot(s_right, b(s_right, viscosity_1, viscosity_2), marker="o", color="black", alpha=1, markersize=4)
    axs[1].text(s_right-0.05, b(s_right, viscosity_1, viscosity_2), rf'$S_+={round(s_right, 2)}$', fontsize=10, horizontalalignment='right')
    for k in np.arange(max(df_s), min(df_s), -ds):
        if np.max(k * (s+S_ORW-1) + 1 - f_s) > 0:
            k_left = k
            s_left = np.max(s * ((k * (s+S_ORW-1) + 1 - f_s) > 0))
            break
    axs[1].plot(s, k_left * (s+S_ORW-1) + 1, color="black", alpha=0.5, linewidth=0.7)
    axs[1].plot(s_left, b(s_left, viscosity_1, viscosity_2), marker="o", color="black", alpha=1, markersize=4)
    axs[1].text(s_left+0.05, b(s_left, viscosity_1, viscosity_2), rf'$S_-={round(s_left, 2)}$', fontsize=10)
    axs[1].set_xlim([0-0.05, 1+0.05])

    axs[2].plot(s, df_s)
    axs[2].grid()
    axs[2].set_title("Производная ф-и")
    axs[2].plot(s_center, np.max(df_s), marker="o", color="black", markersize=4)
    axs[2].text(s_center + 0.1, np.max(df_s) - 0.1, rf'$S_0={round(s_center, 2)}$', fontsize=10)
    axs[2].set_xlim([0-0.05, 1+0.05])

    plt.show()

    return s_right
   
    

@numba.extending.overload(np.gradient)
def np_gradient(f):
    def np_gradient_impl(f):
        out = np.empty_like(f, np.float64)
        out[1:-1] = (f[2:] - f[:-2]) / 2.0
        out[0] = f[1] - f[0]
        out[-1] = f[-1] - f[-2]
        return out

    return np_gradient_impl

def solve_exact(viscosity_1:float, viscosity_2:float, velocity, porosity, t):
    # Графики Баклея-Леверетта
    ds = 0.0001
    s = np.arange(S_WIR, (1-S_ORW) + ds, ds)
    f_s = np.zeros_like(s)
    for i in range(len(s)):
        f_s[i] = b(s[i], viscosity_1, viscosity_2)
    df_s = np.gradient(f_s) * len(s)

    # построение касательных к функции Б-Л
    k_right = 0
    k_left = 0
    s_right = 0
    s_left = 0
    for k in np.arange(max(df_s), min(df_s), -ds):
        if np.min(k * (s - S_WIR) - f_s) < 0:
            k_right = k
            s_right = np.max(s * ((k * (s - S_WIR) - f_s) < 0))
            break
    
    # for k in np.arange(max(df_s), min(df_s), -ds):
    #     if np.max(k * (s+S_ORW-1) + 1 - f_s) > 0:
    #         k_left = k
    #         s_left = np.max(s * ((k * (s+S_ORW-1) + 1 - f_s) > 0))
    #         break

    u = velocity / porosity * df_s

    D = 0
    for i in range(len(s)-1, -1, -1):
        if s[i] <= s_right:
            if D == 0:
                D = u[i]
            u[i] = D

    xx = np.zeros([len(t), len(u)])
    for n in range(len(t)):
        for j in range(len(u)):
            xx[n, j] = u[j] * t[n]

    return xx, s
