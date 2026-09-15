import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ============================================================
# 0. Matplotlib 中文字体设置
# ============================================================
# Windows 优先使用“等线”，若不存在则依次尝试微软雅黑、黑体。
# 数学公式中的英文、数字采用 STIX 字体，显示效果接近 Times New Roman。
# ============================================================

rcParams["font.sans-serif"] = [
    "DengXian",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]

rcParams["axes.unicode_minus"] = False
rcParams["mathtext.fontset"] = "stix"

# 全局字号，可根据报告插图需求自行调整
rcParams["font.size"] = 11


# ============================================================
# 1. 模糊动力系统参数
# ============================================================
#
# 微分方程：
#
#       x̃'(t) = 0.5 x̃(t)
#
# 初始三角模糊数：
#
#       x̃(0) = (0.8, 1.0, 1.2)
#
# α-截集：
#
#       [x̃(0)]_α
#       =
#       [0.8 + 0.2α, 1.2 - 0.2α]
#
# ============================================================

left = 0.8
center = 1.0
right = 1.2

k = 0.5

# 仿真时间
t0 = 0.0
t_end = 5.0

# 数值积分步长
h = 0.1

# 时间序列
t = np.arange(
    t0,
    t_end + h,
    h
)

# 用于绘制不同 α 截集
alpha_levels = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
]


# ============================================================
# 2. 三角模糊数的 α-截集
# ============================================================

def triangular_alpha_cut(alpha):
    """
    计算三角模糊数 (0.8, 1.0, 1.2) 的 α-截集。

    返回：
        lower：α-截集下端点
        upper：α-截集上端点
    """

    lower = (
        left
        + alpha * (center - left)
    )

    upper = (
        right
        - alpha * (right - center)
    )

    return lower, upper


# ============================================================
# 3. 普通常微分方程右端函数
# ============================================================
#
# x'(t) = 0.5 x(t)
#
# ============================================================

def f(t, x):
    return k * x


# ============================================================
# 4. 解析解
# ============================================================
#
# 对初值 x(0)=x0：
#
#       x(t) = x0 * exp(0.5t)
#
# ============================================================

def exact_solution(t, x0):
    return x0 * np.exp(k * t)


# ============================================================
# 5. Euler 数值方法
# ============================================================

def euler_solver(x0, t, h):
    """
    使用 Euler 法求解：

        x'(t) = 0.5x(t)

    输入：
        x0：初值
        t ：时间序列
        h ：时间步长

    输出：
        x ：数值解
    """

    x = np.zeros(len(t))

    x[0] = x0

    for n in range(len(t) - 1):

        x[n + 1] = (
            x[n]
            + h * f(t[n], x[n])
        )

    return x


# ============================================================
# 6. 四阶 Runge-Kutta（RK4）方法
# ============================================================

def rk4_solver(x0, t, h):
    """
    使用四阶 Runge-Kutta 法求解：

        x'(t) = 0.5x(t)
    """

    x = np.zeros(len(t))

    x[0] = x0

    for n in range(len(t) - 1):

        k1 = f(
            t[n],
            x[n]
        )

        k2 = f(
            t[n] + h / 2,
            x[n] + h * k1 / 2
        )

        k3 = f(
            t[n] + h / 2,
            x[n] + h * k2 / 2
        )

        k4 = f(
            t[n] + h,
            x[n] + h * k3
        )

        x[n + 1] = (
            x[n]
            + h
            * (
                k1
                + 2 * k2
                + 2 * k3
                + k4
            )
            / 6
        )

    return x


# ============================================================
# 7. 图1：α-截集的几何理解
# ============================================================
#
# 用三角隶属函数表示：
#
#            μ(x)
#             1
#             /\
#            /  \
#           /----\   ← 某个 α 水平
#          /      \
#         /________\
#
# α 水平线与隶属函数相交的两个横坐标，
# 就构成该 α 水平下的 α-截集。
#
# ============================================================

def plot_alpha_cut_geometry():

    # 构造三角隶属函数
    x_left = np.linspace(
        left,
        center,
        200
    )

    x_right = np.linspace(
        center,
        right,
        200
    )

    mu_left = (
        (x_left - left)
        / (center - left)
    )

    mu_right = (
        (right - x_right)
        / (right - center)
    )


    plt.figure(
        figsize=(9, 6)
    )


    # --------------------------------------------------------
    # 绘制三角隶属函数
    # --------------------------------------------------------

    plt.plot(
        x_left,
        mu_left,
        linewidth=2.2
    )

    plt.plot(
        x_right,
        mu_right,
        linewidth=2.2
    )


    # 填充三角模糊数区域
    x_all = np.concatenate(
        (
            x_left,
            x_right
        )
    )

    mu_all = np.concatenate(
        (
            mu_left,
            mu_right
        )
    )

    plt.fill_between(
        x_all,
        0,
        mu_all,
        alpha=0.08
    )


    # --------------------------------------------------------
    # 绘制不同 α 水平
    # --------------------------------------------------------

    geometry_alphas = [
        0.25,
        0.50,
        0.75
    ]


    for alpha in geometry_alphas:

        lower, upper = (
            triangular_alpha_cut(
                alpha
            )
        )


        # α 水平对应的区间线
        plt.plot(
            [lower, upper],
            [alpha, alpha],
            linewidth=2.5,
            label=(
                rf"$\alpha={alpha}$："
                f"[{lower:.2f}, {upper:.2f}]"
            )
        )


        # 左交点到横轴的辅助线
        plt.plot(
            [lower, lower],
            [0, alpha],
            linestyle="--",
            linewidth=1
        )


        # 右交点到横轴的辅助线
        plt.plot(
            [upper, upper],
            [0, alpha],
            linestyle="--",
            linewidth=1
        )


        # 两个交点
        plt.scatter(
            [lower, upper],
            [alpha, alpha],
            s=35,
            zorder=5
        )


    # --------------------------------------------------------
    # 标出三角模糊数三个特征点
    # --------------------------------------------------------

    plt.scatter(
        [left, center, right],
        [0, 1, 0],
        s=45,
        zorder=5
    )


    plt.text(
        left,
        -0.07,
        "0.8",
        ha="center"
    )

    plt.text(
        center,
        1.03,
        "1.0",
        ha="center"
    )

    plt.text(
        right,
        -0.07,
        "1.2",
        ha="center"
    )


    # --------------------------------------------------------
    # 图表设置
    # --------------------------------------------------------

    plt.xlabel(
        "变量 $x$"
    )

    plt.ylabel(
        "隶属度 $\\mu_{\\tilde{x}}(x)$"
    )

    plt.title(
        "三角模糊数的 α-截集几何理解"
    )

    plt.xlim(
        0.72,
        1.28
    )

    plt.ylim(
        -0.12,
        1.12
    )

    plt.yticks(
        [
            0,
            0.25,
            0.50,
            0.75,
            1.0
        ]
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 8. 图2：不同 α 水平下的解析解与模糊带
# ============================================================

def plot_exact_fuzzy_solution():

    plt.figure(
        figsize=(9, 6)
    )


    # --------------------------------------------------------
    # α = 0 对应整个模糊数的支撑区间
    # 用它形成最外层模糊带
    # --------------------------------------------------------

    lower0, upper0 = (
        triangular_alpha_cut(
            0.0
        )
    )

    exact_lower0 = (
        exact_solution(
            t,
            lower0
        )
    )

    exact_upper0 = (
        exact_solution(
            t,
            upper0
        )
    )


    plt.fill_between(
        t,
        exact_lower0,
        exact_upper0,
        alpha=0.18,
        label=r"模糊状态范围（$\alpha=0$）"
    )


    # --------------------------------------------------------
    # 绘制不同 α 水平下的上下端点
    # --------------------------------------------------------

    for alpha in alpha_levels:

        lower, upper = (
            triangular_alpha_cut(
                alpha
            )
        )

        lower_solution = (
            exact_solution(
                t,
                lower
            )
        )

        upper_solution = (
            exact_solution(
                t,
                upper
            )
        )


        # α=1 时上下端点完全重合，
        # 只绘制中心轨迹
        if np.isclose(
            alpha,
            1.0
        ):

            plt.plot(
                t,
                lower_solution,
                linewidth=2.5,
                label=r"$\alpha=1$（中心轨迹）"
            )

        else:

            plt.plot(
                t,
                lower_solution,
                linewidth=1.3,
                label=rf"$\alpha={alpha}$ 下端点"
            )

            plt.plot(
                t,
                upper_solution,
                linewidth=1.3,
                linestyle="--",
                label=rf"$\alpha={alpha}$ 上端点"
            )


    plt.xlabel(
        "时间 $t$"
    )

    plt.ylabel(
        "模糊状态 $\\tilde{x}(t)$"
    )

    plt.title(
        "不同 α 水平下模糊状态的解析解"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8,
        ncol=2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. 图3：Euler、RK4 与解析解比较
# ============================================================
#
# 为避免图线过多，
# 选择 α = 0.5 作为代表进行比较。
#
# α = 0.5：
#
#       [x̃(0)]_0.5 = [0.9, 1.1]
#
# ============================================================

def plot_method_comparison():

    alpha = 0.5

    lower0, upper0 = (
        triangular_alpha_cut(
            alpha
        )
    )


    # --------------------------------------------------------
    # 解析解
    # --------------------------------------------------------

    exact_lower = (
        exact_solution(
            t,
            lower0
        )
    )

    exact_upper = (
        exact_solution(
            t,
            upper0
        )
    )


    # --------------------------------------------------------
    # Euler 解
    # --------------------------------------------------------

    euler_lower = (
        euler_solver(
            lower0,
            t,
            h
        )
    )

    euler_upper = (
        euler_solver(
            upper0,
            t,
            h
        )
    )


    # --------------------------------------------------------
    # RK4 解
    # --------------------------------------------------------

    rk4_lower = (
        rk4_solver(
            lower0,
            t,
            h
        )
    )

    rk4_upper = (
        rk4_solver(
            upper0,
            t,
            h
        )
    )


    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )


    # 解析解
    plt.plot(
        t,
        exact_lower,
        linewidth=2.3,
        label="解析解下端点"
    )

    plt.plot(
        t,
        exact_upper,
        linewidth=2.3,
        label="解析解上端点"
    )


    # Euler
    plt.plot(
        t,
        euler_lower,
        linestyle="--",
        linewidth=1.6,
        label="Euler 法下端点"
    )

    plt.plot(
        t,
        euler_upper,
        linestyle="--",
        linewidth=1.6,
        label="Euler 法上端点"
    )


    # RK4
    plt.plot(
        t,
        rk4_lower,
        linestyle=":",
        linewidth=2.0,
        label="RK4 法下端点"
    )

    plt.plot(
        t,
        rk4_upper,
        linestyle=":",
        linewidth=2.0,
        label="RK4 法上端点"
    )


    plt.xlabel(
        "时间 $t$"
    )

    plt.ylabel(
        "状态 $x(t)$"
    )

    plt.title(
        r"Euler 法、RK4 法与解析解对比（$\alpha=0.5$）"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=9,
        ncol=2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 10. 图4：Euler 与 RK4 绝对误差比较
# ============================================================
#
# 为同时反映上下端点误差，
# 定义某一时刻的区间端点最大绝对误差：
#
# error(t)
# =
# max(
#     |x_lower_num - x_lower_exact|,
#     |x_upper_num - x_upper_exact|
# )
#
# ============================================================

def plot_error_comparison():

    alpha = 0.5

    lower0, upper0 = (
        triangular_alpha_cut(
            alpha
        )
    )


    # --------------------------------------------------------
    # 解析解
    # --------------------------------------------------------

    exact_lower = (
        exact_solution(
            t,
            lower0
        )
    )

    exact_upper = (
        exact_solution(
            t,
            upper0
        )
    )


    # --------------------------------------------------------
    # Euler
    # --------------------------------------------------------

    euler_lower = (
        euler_solver(
            lower0,
            t,
            h
        )
    )

    euler_upper = (
        euler_solver(
            upper0,
            t,
            h
        )
    )


    # --------------------------------------------------------
    # RK4
    # --------------------------------------------------------

    rk4_lower = (
        rk4_solver(
            lower0,
            t,
            h
        )
    )

    rk4_upper = (
        rk4_solver(
            upper0,
            t,
            h
        )
    )


    # --------------------------------------------------------
    # 计算 Euler 上下端点绝对误差
    # --------------------------------------------------------

    euler_error_lower = np.abs(
        euler_lower
        -
        exact_lower
    )

    euler_error_upper = np.abs(
        euler_upper
        -
        exact_upper
    )


    # 取上下端点中较大的误差
    euler_error = np.maximum(
        euler_error_lower,
        euler_error_upper
    )


    # --------------------------------------------------------
    # 计算 RK4 上下端点绝对误差
    # --------------------------------------------------------

    rk4_error_lower = np.abs(
        rk4_lower
        -
        exact_lower
    )

    rk4_error_upper = np.abs(
        rk4_upper
        -
        exact_upper
    )


    rk4_error = np.maximum(
        rk4_error_lower,
        rk4_error_upper
    )


    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(
        t,
        euler_error,
        linewidth=2,
        label="Euler 法绝对误差"
    )


    plt.plot(
        t,
        rk4_error,
        linewidth=2,
        label="RK4 法绝对误差"
    )


    plt.xlabel(
        "时间 $t$"
    )

    plt.ylabel(
        "最大端点绝对误差"
    )

    plt.title(
        r"Euler 法与 RK4 法数值误差对比（$\alpha=0.5$）"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 11. 输出简单的误差结果
# ============================================================

def print_error_results():

    alpha = 0.5

    lower0, upper0 = (
        triangular_alpha_cut(
            alpha
        )
    )


    exact_lower = (
        exact_solution(
            t,
            lower0
        )
    )

    exact_upper = (
        exact_solution(
            t,
            upper0
        )
    )


    euler_lower = (
        euler_solver(
            lower0,
            t,
            h
        )
    )

    euler_upper = (
        euler_solver(
            upper0,
            t,
            h
        )
    )


    rk4_lower = (
        rk4_solver(
            lower0,
            t,
            h
        )
    )

    rk4_upper = (
        rk4_solver(
            upper0,
            t,
            h
        )
    )


    euler_error = np.maximum(
        np.abs(
            euler_lower
            -
            exact_lower
        ),
        np.abs(
            euler_upper
            -
            exact_upper
        )
    )


    rk4_error = np.maximum(
        np.abs(
            rk4_lower
            -
            exact_lower
        ),
        np.abs(
            rk4_upper
            -
            exact_upper
        )
    )


    print(
        "α = 0.5 时："
    )

    print(
        f"Euler 法最大绝对误差："
        f"{np.max(euler_error):.8f}"
    )

    print(
        f"RK4 法最大绝对误差："
        f"{np.max(rk4_error):.8f}"
    )


# ============================================================
# 12. 主程序
# ============================================================

if __name__ == "__main__":

    # 图1：α-截集的几何理解
    plot_alpha_cut_geometry()

    # 图2：不同 α 水平下的解析解与模糊带
    plot_exact_fuzzy_solution()

    # 图3：Euler、RK4 和解析解对比
    plot_method_comparison()

    # 图4：Euler 与 RK4 绝对误差对比
    plot_error_comparison()

    # 在终端输出最大误差
    print_error_results()