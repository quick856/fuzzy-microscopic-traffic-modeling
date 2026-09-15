import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 0. 中文字体设置
# ============================================================

def setup_chinese_font():
    """
    自动寻找可用中文字体。
    Windows 下优先使用等线，其次微软雅黑、黑体。
    """

    # Windows 常见中文字体路径
    font_paths = [
        r"C:\Windows\Fonts\Deng.ttf",       # 等线
        r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    ]

    # 尝试直接加载字体文件
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)

                font_name = font_manager.FontProperties(
                    fname=path
                ).get_name()

                plt.rcParams["font.sans-serif"] = [
                    font_name,
                    "DejaVu Sans"
                ]

                plt.rcParams["axes.unicode_minus"] = False

                print(f"当前绘图中文字体：{font_name}")
                return

            except Exception:
                pass

    # 如果找不到具体字体文件，则根据字体名称搜索
    candidate_fonts = [
        "DengXian",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei"
    ]

    available_fonts = {
        f.name for f in font_manager.fontManager.ttflist
    }

    for name in candidate_fonts:
        if name in available_fonts:
            plt.rcParams["font.sans-serif"] = [
                name,
                "DejaVu Sans"
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(f"当前绘图中文字体：{name}")
            return

    print(
        "警告：没有检测到常用中文字体，"
        "若图中文字显示为方框，请安装等线、微软雅黑或思源黑体。"
    )


setup_chinese_font()


# ============================================================
# 1. 全局设置
# ============================================================

# 是否保存图片
SAVE_FIG = True

# 是否运行结束后显示所有图片
SHOW_FIG = True

# 图片保存目录
OUTPUT_DIR = "fuzzy_idm_results"

if SAVE_FIG:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# 时间步长
DT = 0.05

# 跟驰车辆数量
#
# 前车：
#   车辆0
#
# 跟驰车辆：
#   车辆1 ~ 车辆5
#
# 使用多辆车是为了进一步分析交通流扰动传播。
N_FOLLOWERS = 5


# α 水平
ALPHA_LEVELS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# 每个模糊参数在一个 α 截集内的采样点数量
#
# 5 个 a
# × 5 个 b
# × 5 个 T
# = 125 个参数组合
#
# GRID_N 越大，α-截集包络近似越精确，
# 但计算量也越大。
GRID_N = 5


# ============================================================
# 2. 单位转换
# ============================================================

def kmh_to_ms(v):
    """
    km/h 转换为 m/s
    """

    return np.asarray(
        v,
        dtype=float
    ) / 3.6


def ms_to_kmh(v):
    """
    m/s 转换为 km/h
    """

    return np.asarray(
        v,
        dtype=float
    ) * 3.6


# ============================================================
# 3. IDM 确定性参数
# ============================================================
#
# 严格对应原 IDM 公式：
#
# a_n(t)
# =
# a_max [
#     1
#     - (V_n / V_des)^beta
#     - (S_des / S_n)^2
# ]
#
# S_des
# =
# S_jam
# +
# max(
#     0,
#     V_n T
#     -
#     V_n ΔV_n /
#     (2 sqrt(a_max a_comf))
# )
#
# ΔV_n = V_(n-1) - V_n
#
# ============================================================


# 期望速度 Ṽ_n
# 图中显示 km/h，计算时转换为 m/s
V_DES_KMH = 108.0
V_DES = kmh_to_ms(V_DES_KMH)


# 静止期望间距 S_jam
S_JAM = 2.0


# 加速度指数 β
BETA = 4.0


# 车辆长度 L
VEHICLE_LENGTH = 5.0


# ============================================================
# 4. 三参数模糊数
# ============================================================
#
# 三角模糊数：
#
# a_max = (0.8, 1.0, 1.2)
#
# a_comf = (1.2, 1.5, 1.8)
#
# T = (1.2, 1.5, 1.8)
#
# 中心值即 Crisp IDM 参数。
# ============================================================


# 最大加速度 a_max
A_MAX_TRI = (
    0.8,
    1.0,
    1.2
)


# 舒适减速度 a_comf
A_COMF_TRI = (
    1.2,
    1.5,
    1.8
)


# 期望车头时距 T
T_TRI = (
    1.2,
    1.5,
    1.8
)


# Crisp IDM 使用中心值
A_MAX_CRISP = A_MAX_TRI[1]
A_COMF_CRISP = A_COMF_TRI[1]
T_CRISP = T_TRI[1]


# ============================================================
# 5. 三角模糊数 α-截集
# ============================================================

def triangular_alpha_cut(triangle, alpha):
    """
    三角模糊数：

        (left, center, right)

    的 α-截集：

        [
            left + α(center-left),
            right - α(right-center)
        ]
    """

    left, center, right = triangle

    lower = (
        left
        + alpha
        * (center - left)
    )

    upper = (
        right
        - alpha
        * (right - center)
    )

    return lower, upper


# ============================================================
# 6. 构造 α 水平下的三参数集合 Ω_α
# ============================================================

def create_parameter_grid(alpha, grid_n=GRID_N):
    """
    给定 α 水平后：

        [a_max]_α
        [a_comf]_α
        [T]_α

    构造三参数笛卡尔积：

        Ω_α
        =
        [a_max]_α
        ×
        [a_comf]_α
        ×
        [T]_α

    返回所有确定性参数组合。
    """

    a_lower, a_upper = triangular_alpha_cut(
        A_MAX_TRI,
        alpha
    )

    b_lower, b_upper = triangular_alpha_cut(
        A_COMF_TRI,
        alpha
    )

    T_lower, T_upper = triangular_alpha_cut(
        T_TRI,
        alpha
    )

    # 如果 α=1，上下界相同，只需要一个参数值
    def make_samples(lower, upper):

        if np.isclose(
            lower,
            upper
        ):
            return np.array([
                (lower + upper) / 2
            ])

        return np.linspace(
            lower,
            upper,
            grid_n
        )

    a_samples = make_samples(
        a_lower,
        a_upper
    )

    b_samples = make_samples(
        b_lower,
        b_upper
    )

    T_samples = make_samples(
        T_lower,
        T_upper
    )

    # 三参数笛卡尔积
    a_grid, b_grid, T_grid = np.meshgrid(
        a_samples,
        b_samples,
        T_samples,
        indexing="ij"
    )

    return (
        a_grid.ravel(),
        b_grid.ravel(),
        T_grid.ravel()
    )


# ============================================================
# 7. IDM 加速度函数
# ============================================================

def idm_acceleration(
    v_follower,
    v_leader,
    gap,
    a_max,
    a_comf,
    T
):
    """
    严格按照本文采用的 IDM 形式计算跟驰车辆加速度。

    ----------------------------------------------------------
    相对速度：
    ----------------------------------------------------------

        ΔV_n
        =
        V_(n-1)
        -
        V_n

    即：

        ΔV > 0：
            前车速度大于本车

        ΔV < 0：
            本车速度大于前车，
            本车正在接近前车


    ----------------------------------------------------------
    期望动态间距：
    ----------------------------------------------------------

        S_des
        =
        S_jam
        +
        max[
            0,
            V_n T
            -
            V_n ΔV
            /
            (2 sqrt(a_max a_comf))
        ]


    ----------------------------------------------------------
    实际加速度：
    ----------------------------------------------------------

        a_n
        =
        a_max
        [
            1
            -
            (V_n / V_des)^β
            -
            (S_des / S_n)^2
        ]
    """

    # --------------------------------------------------------
    # 防止实际间距接近 0 时出现除零
    #
    # 这里只是数值保护，
    # 并不代表车辆允许出现 0.1 m 的实际安全间距。
    # --------------------------------------------------------

    gap_effective = np.maximum(
        gap,
        0.1
    )

    # 相对速度
    delta_v = (
        v_leader
        -
        v_follower
    )

    # 期望动态间距
    desired_gap = (
        S_JAM
        +
        np.maximum(
            0.0,

            v_follower * T

            -

            (
                v_follower
                * delta_v
            )
            /
            (
                2.0
                *
                np.sqrt(
                    a_max
                    *
                    a_comf
                )
            )
        )
    )

    # IDM 实际加速度
    acceleration = (
        a_max
        *
        (
            1.0

            -

            (
                v_follower
                /
                V_DES
            ) ** BETA

            -

            (
                desired_gap
                /
                gap_effective
            ) ** 2
        )
    )

    return acceleration


# ============================================================
# 8. 前车典型交通场景
# ============================================================

def generate_leader_profile(
    scenario,
    t
):
    """
    构造三种典型前车运动场景：

    1. accelerating
       匀加速前车

    2. braking
       减速停车

    3. periodic
       周期性速度扰动
    """

    # --------------------------------------------------------
    # 场景1：匀加速前车
    #
    # 初始速度：54 km/h
    # 加速度：0.5 m/s²
    # 最大速度：72 km/h
    # --------------------------------------------------------

    if scenario == "accelerating":

        v_initial = kmh_to_ms(
            54.0
        )

        leader_acc = 0.5

        v_max = kmh_to_ms(
            72.0
        )

        leader_v = np.minimum(
            v_initial
            +
            leader_acc * t,

            v_max
        )


    # --------------------------------------------------------
    # 场景2：前车减速停车
    #
    # 初始速度：72 km/h
    # t = 10 s 开始制动
    # 制动减速度：1.5 m/s²
    # --------------------------------------------------------

    elif scenario == "braking":

        v_initial = kmh_to_ms(
            72.0
        )

        braking_start = 10.0

        deceleration = 1.5

        leader_v = np.where(

            t < braking_start,

            v_initial,

            np.maximum(
                0.0,

                v_initial
                -
                deceleration
                *
                (
                    t
                    -
                    braking_start
                )
            )
        )


    # --------------------------------------------------------
    # 场景3：周期性速度扰动
    #
    # 平均速度：72 km/h
    # 振幅：7.2 km/h
    #
    # V_leader
    # =
    # 72
    # +
    # 7.2 sin(0.2t)
    #
    # --------------------------------------------------------

    elif scenario == "periodic":

        mean_speed = kmh_to_ms(
            72.0
        )

        amplitude = kmh_to_ms(
            7.2
        )

        omega = 0.2

        leader_v = (
            mean_speed
            +
            amplitude
            *
            np.sin(
                omega * t
            )
        )

    else:

        raise ValueError(
            f"未知场景：{scenario}"
        )


    # --------------------------------------------------------
    # 根据速度积分得到前车位置
    #
    # 使用梯形积分：
    #
    # x(k+1)
    # =
    # x(k)
    # +
    # 0.5
    # *
    # [v(k)+v(k+1)]
    # *
    # Δt
    # --------------------------------------------------------

    leader_x = np.zeros_like(
        t
    )

    dt_array = np.diff(
        t
    )

    leader_x[1:] = np.cumsum(
        0.5
        *
        (
            leader_v[:-1]
            +
            leader_v[1:]
        )
        *
        dt_array
    )

    return (
        leader_v,
        leader_x
    )


# ============================================================
# 9. 计算初始平衡间距
# ============================================================

def equilibrium_gap(
    speed,
    T=T_CRISP
):
    """
    根据经典 IDM 在：

        ΔV = 0
        a_n = 0

    条件下近似计算初始平衡间距。

    这样可以减少仿真开始时不必要的瞬态扰动。
    """

    speed = float(
        speed
    )

    denominator = (
        1.0
        -
        (
            speed
            /
            float(V_DES)
        ) ** BETA
    )

    denominator = max(
        denominator,
        1e-6
    )

    desired_gap = (
        S_JAM
        +
        speed * T
    )

    gap = (
        desired_gap
        /
        math.sqrt(
            denominator
        )
    )

    return gap


# ============================================================
# 10. 确定性 IDM / 参数组合 IDM 仿真核心
# ============================================================

def simulate_platoon(
    t,
    leader_v,
    leader_x,
    a_values,
    b_values,
    T_values,
    n_followers=N_FOLLOWERS
):
    """
    同时仿真多个 IDM 参数组合。

    ----------------------------------------------------------
    输入：
    ----------------------------------------------------------

    a_values:
        多组 a_max

    b_values:
        多组 a_comf

    T_values:
        多组安全车头时距 T


    ----------------------------------------------------------
    如果只输入一个参数组合：
    ----------------------------------------------------------

        a=[1.0]
        b=[1.5]
        T=[1.5]

    则得到 Crisp IDM。


    ----------------------------------------------------------
    如果输入 Ω_α 中所有参数组合：
    ----------------------------------------------------------

    则一次计算整个 α 水平下的 IDM 轨迹集合。
    """

    a_values = np.asarray(
        a_values,
        dtype=float
    )

    b_values = np.asarray(
        b_values,
        dtype=float
    )

    T_values = np.asarray(
        T_values,
        dtype=float
    )


    # 参数组合数量
    n_parameter_sets = len(
        a_values
    )

    n_time = len(
        t
    )

    dt = (
        t[1]
        -
        t[0]
    )


    # --------------------------------------------------------
    # 初始速度
    #
    # 所有跟驰车辆开始时与前车同速。
    # --------------------------------------------------------

    v = np.full(
        (
            n_parameter_sets,
            n_followers
        ),
        leader_v[0],
        dtype=float
    )


    # --------------------------------------------------------
    # 初始间距
    #
    # 使用 Crisp 参数中心值对应的平衡间距。
    # --------------------------------------------------------

    initial_gap = equilibrium_gap(
        leader_v[0],
        T_CRISP
    )


    # --------------------------------------------------------
    # 初始位置
    #
    # 前车初始位置：
    #     x0 = 0
    #
    # 第一辆跟驰车：
    #     -(L + S)
    #
    # 第二辆：
    #     -2(L + S)
    #
    # ...
    # --------------------------------------------------------

    base_positions = np.array(
        [
            -(i + 1)
            *
            (
                VEHICLE_LENGTH
                +
                initial_gap
            )

            for i in range(
                n_followers
            )
        ],
        dtype=float
    )


    x = np.zeros(
        (
            n_parameter_sets,
            n_followers
        ),
        dtype=float
    )

    x[:] = base_positions


    # --------------------------------------------------------
    # 保存轨迹
    #
    # 维度：
    #
    # 参数组合
    # ×
    # 跟驰车辆
    # ×
    # 时间
    # --------------------------------------------------------

    velocity_history = np.empty(
        (
            n_parameter_sets,
            n_followers,
            n_time
        )
    )

    position_history = np.empty_like(
        velocity_history
    )

    gap_history = np.empty_like(
        velocity_history
    )

    acceleration_history = np.empty_like(
        velocity_history
    )


    # ========================================================
    # 时间积分
    # ========================================================

    for k in range(
        n_time
    ):

        velocity_history[:, :, k] = v
        position_history[:, :, k] = x


        current_acc = np.empty_like(
            v
        )

        current_gap = np.empty_like(
            v
        )


        # ----------------------------------------------------
        # 在同一个时刻计算每辆跟驰车加速度
        # ----------------------------------------------------

        for j in range(
            n_followers
        ):

            # 第一辆跟驰车直接跟随外部前车
            if j == 0:

                v_leader_now = leader_v[k]
                x_leader_now = leader_x[k]

            # 后续车辆以前一辆跟驰车作为前车
            else:

                v_leader_now = v[:, j - 1]
                x_leader_now = x[:, j - 1]


            # 实际净间距
            gap = (
                x_leader_now
                -
                x[:, j]
                -
                VEHICLE_LENGTH
            )

            current_gap[:, j] = gap


            # IDM 加速度
            current_acc[:, j] = idm_acceleration(

                v_follower=v[:, j],

                v_leader=v_leader_now,

                gap=gap,

                a_max=a_values,

                a_comf=b_values,

                T=T_values
            )

        stopped_mask = (
                (v <= 1e-6)
                &
                (current_acc < 0.0)
        )

        current_acc = np.where(
            stopped_mask,
            0.0,
            current_acc
        )
        gap_history[:, :, k] = (
            current_gap
        )

        acceleration_history[:, :, k] = (
            current_acc
        )


        # ----------------------------------------------------
        # 最后一个时间点无需继续更新
        # ----------------------------------------------------

        if k == n_time - 1:
            break


        # ----------------------------------------------------
        # 更新速度
        #
        # V(k+1)
        # =
        # V(k)
        # +
        # a(k)Δt
        #
        # 速度最低限制为 0。
        # ----------------------------------------------------

        v_new = np.maximum(

            0.0,

            v
            +
            current_acc
            *
            dt
        )


        # ----------------------------------------------------
        # 更新位置
        #
        # 使用梯形近似：
        #
        # x(k+1)
        # =
        # x(k)
        # +
        # (V(k)+V(k+1))/2
        # *
        # Δt
        # ----------------------------------------------------

        x = (
            x
            +
            0.5
            *
            (
                v
                +
                v_new
            )
            *
            dt
        )

        v = v_new


    return {
        "v": velocity_history,
        "x": position_history,
        "gap": gap_history,
        "acc": acceleration_history
    }


# ============================================================
# 11. Crisp IDM
# ============================================================

def simulate_crisp_idm(
    t,
    leader_v,
    leader_x
):
    """
    确定性 IDM：

        a_max = 1.0
        a_comf = 1.5
        T = 1.5
    """

    return simulate_platoon(

        t=t,

        leader_v=leader_v,

        leader_x=leader_x,

        a_values=[
            A_MAX_CRISP
        ],

        b_values=[
            A_COMF_CRISP
        ],

        T_values=[
            T_CRISP
        ]
    )


# ============================================================
# 12. 对参数组合轨迹取 α-截集上下包络
# ============================================================

def extract_fuzzy_envelope(
    raw_result
):
    """
    对所有参数组合产生的轨迹逐时刻取：

        min
        max

    得到 Fuzzy-IDM 的 α-截集数值近似。
    """

    result = {}

    for variable in [
        "v",
        "x",
        "gap",
        "acc"
    ]:

        data = raw_result[
            variable
        ]

        result[
            variable + "_lower"
        ] = np.min(
            data,
            axis=0
        )

        result[
            variable + "_upper"
        ] = np.max(
            data,
            axis=0
        )

    return result


# ============================================================
# 13. 三参数 Fuzzy-IDM
# ============================================================

def simulate_fuzzy_idm(
    t,
    leader_v,
    leader_x
):
    """
    对每一个 α 水平：

    1. 构造 Ω_α
    2. 求解 Ω_α 中不同 a、b、T 对应的确定性 IDM
    3. 对轨迹逐时刻取 min/max
    4. 得到模糊速度、间距、加速度等 α-截集
    """

    fuzzy_results = {}

    # 部分 α 水平保留原始参数轨迹，
    # 用于后面绘制速度-间距相图。
    raw_results = {}


    for alpha in ALPHA_LEVELS:

        print(
            f"正在计算 α = {alpha:.2f}"
        )

        a_values, b_values, T_values = (
            create_parameter_grid(
                alpha
            )
        )

        print(
            f"    参数组合数量："
            f"{len(a_values)}"
        )

        raw = simulate_platoon(

            t=t,

            leader_v=leader_v,

            leader_x=leader_x,

            a_values=a_values,

            b_values=b_values,

            T_values=T_values
        )


        fuzzy_results[
            alpha
        ] = extract_fuzzy_envelope(
            raw
        )


        # α=0、0.5、1 保留完整轨迹
        if alpha in [
            0.0,
            0.5,
            1.0
        ]:

            raw_results[
                alpha
            ] = raw


    return (
        fuzzy_results,
        raw_results
    )


# ============================================================
# 14. 图片保存辅助函数
# ============================================================

def finish_figure(
    filename
):
    """
    调整版式并保存图片。
    """

    plt.tight_layout()

    if SAVE_FIG:

        path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"图片已保存：{path}"
        )

    if not SHOW_FIG:
        plt.close()


# ============================================================
# 15. 速度演化图
# ============================================================

def plot_velocity(
    t,
    leader_v,
    crisp,
    fuzzy,
    scene_name,
    file_prefix
):
    """
    对比：

    前车
    Crisp IDM
    Fuzzy-IDM α=0
    Fuzzy-IDM α=0.5

    这里重点展示最后一辆跟驰车，
    因为扰动传播到车队末端后最能体现交通流稳定性。
    """

    vehicle_index = (
        N_FOLLOWERS - 1
    )

    plt.figure(
        figsize=(9, 5.5)
    )


    # α=0 最外层模糊带
    lower_0 = (
        fuzzy[0.0]["v_lower"][
            vehicle_index
        ]
        * 3.6
    )

    upper_0 = (
        fuzzy[0.0]["v_upper"][
            vehicle_index
        ]
        * 3.6
    )

    plt.fill_between(
        t,
        lower_0,
        upper_0,
        alpha=0.18,
        label="Fuzzy-IDM：α=0"
    )


    # α=0.5 模糊带
    lower_05 = (
        fuzzy[0.5]["v_lower"][
            vehicle_index
        ]
        * 3.6
    )

    upper_05 = (
        fuzzy[0.5]["v_upper"][
            vehicle_index
        ]
        * 3.6
    )

    plt.fill_between(
        t,
        lower_05,
        upper_05,
        alpha=0.30,
        label="Fuzzy-IDM：α=0.5"
    )


    # 前车速度
    plt.plot(
        t,
        ms_to_kmh(
            leader_v
        ),
        linestyle="--",
        linewidth=1.8,
        label="前车"
    )


    # Crisp IDM
    plt.plot(
        t,
        ms_to_kmh(
            crisp["v"][
                0,
                vehicle_index
            ]
        ),
        linewidth=2.0,
        label="确定性 IDM"
    )


    plt.xlabel(
        "时间 / s"
    )

    plt.ylabel(
        "速度 / (km/h)"
    )

    plt.title(
        f"{scene_name}：速度演化"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        file_prefix
        +
        "_velocity.png"
    )


# ============================================================
# 16. 间距演化图
# ============================================================

def plot_gap(
    t,
    crisp,
    fuzzy,
    scene_name,
    file_prefix
):
    """
    实际净间距 S_n(t) 对比。
    """

    vehicle_index = (
        N_FOLLOWERS - 1
    )

    plt.figure(
        figsize=(9, 5.5)
    )


    plt.fill_between(

        t,

        fuzzy[0.0]["gap_lower"][
            vehicle_index
        ],

        fuzzy[0.0]["gap_upper"][
            vehicle_index
        ],

        alpha=0.18,

        label="Fuzzy-IDM：α=0"
    )


    plt.fill_between(

        t,

        fuzzy[0.5]["gap_lower"][
            vehicle_index
        ],

        fuzzy[0.5]["gap_upper"][
            vehicle_index
        ],

        alpha=0.30,

        label="Fuzzy-IDM：α=0.5"
    )


    plt.plot(

        t,

        crisp["gap"][
            0,
            vehicle_index
        ],

        linewidth=2.0,

        label="确定性 IDM"
    )


    plt.xlabel(
        "时间 / s"
    )

    plt.ylabel(
        "实际净间距 / m"
    )

    plt.title(
        f"{scene_name}：车辆间距演化"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        file_prefix
        +
        "_gap.png"
    )


# ============================================================
# 17. 加速度演化图
# ============================================================

def plot_acceleration(
    t,
    crisp,
    fuzzy,
    scene_name,
    file_prefix
):
    """
    加速度演化可用于分析跟驰行为的波动程度。
    """

    vehicle_index = (
        N_FOLLOWERS - 1
    )

    plt.figure(
        figsize=(9, 5.5)
    )


    plt.fill_between(

        t,

        fuzzy[0.0]["acc_lower"][
            vehicle_index
        ],

        fuzzy[0.0]["acc_upper"][
            vehicle_index
        ],

        alpha=0.18,

        label="Fuzzy-IDM：α=0"
    )


    plt.fill_between(

        t,

        fuzzy[0.5]["acc_lower"][
            vehicle_index
        ],

        fuzzy[0.5]["acc_upper"][
            vehicle_index
        ],

        alpha=0.30,

        label="Fuzzy-IDM：α=0.5"
    )


    plt.plot(

        t,

        crisp["acc"][
            0,
            vehicle_index
        ],

        linewidth=2.0,

        label="确定性 IDM"
    )


    plt.axhline(
        0,
        linewidth=1,
        linestyle="--"
    )

    plt.xlabel(
        "时间 / s"
    )

    plt.ylabel(
        "加速度 / (m/s²)"
    )

    plt.title(
        f"{scene_name}：加速度演化"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        file_prefix
        +
        "_acceleration.png"
    )


# ============================================================
# 18. 不同 α 水平速度-间距相图
# ============================================================

def plot_phase_diagram(
    raw_results,
    crisp,
    scene_name,
    file_prefix
):
    """
    绘制最后一辆跟驰车的：

        速度 - 实际间距

    相图。

    α=0 和 α=0.5：
        绘制不同参数组合产生的状态点云

    α=1：
        参数退化为中心值，因此是一条确定轨迹
    """

    vehicle_index = (
        N_FOLLOWERS - 1
    )


    plt.figure(
        figsize=(8, 6)
    )


    # 为减少点数，每隔一定时间采样一次
    step = 10


    for alpha in [
        0.0,
        0.5
    ]:

        raw = raw_results[
            alpha
        ]

        gap = (
            raw["gap"][
                :,
                vehicle_index,
                ::step
            ]
        ).ravel()

        velocity = (
            raw["v"][
                :,
                vehicle_index,
                ::step
            ]
            *
            3.6
        ).ravel()


        plt.scatter(
            gap,
            velocity,
            s=8,
            alpha=0.12,
            label=f"Fuzzy-IDM：α={alpha}"
        )


    # α=1 即中心参数轨迹
    plt.plot(

        crisp["gap"][
            0,
            vehicle_index
        ],

        ms_to_kmh(
            crisp["v"][
                0,
                vehicle_index
            ]
        ),

        linewidth=2,

        label="α=1 / 确定性 IDM"
    )


    plt.xlabel(
        "实际净间距 / m"
    )

    plt.ylabel(
        "速度 / (km/h)"
    )

    plt.title(
        f"{scene_name}：速度-间距相图"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        file_prefix
        +
        "_phase.png"
    )


# ============================================================
# 19. 不同 α 水平下速度模糊宽度
# ============================================================

def plot_uncertainty_width(
    t,
    fuzzy,
    scene_name,
    file_prefix
):
    """
    定义速度模糊区间宽度：

        W_V(t, α)
        =
        V_upper
        -
        V_lower

    用于定量分析模糊性的传播。
    """

    vehicle_index = (
        N_FOLLOWERS - 1
    )

    plt.figure(
        figsize=(9, 5.5)
    )


    for alpha in ALPHA_LEVELS:

        lower = fuzzy[
            alpha
        ]["v_lower"][
            vehicle_index
        ]

        upper = fuzzy[
            alpha
        ]["v_upper"][
            vehicle_index
        ]

        width = (
            upper
            -
            lower
        ) * 3.6


        plt.plot(
            t,
            width,
            linewidth=1.7,
            label=f"α={alpha}"
        )


    plt.xlabel(
        "时间 / s"
    )

    plt.ylabel(
        "速度模糊区间宽度 / (km/h)"
    )

    plt.title(
        f"{scene_name}：速度不确定性传播"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        file_prefix
        +
        "_uncertainty_width.png"
    )


# ============================================================
# 20. 周期扰动下的交通流稳定性指标
# ============================================================

def calculate_stability_metrics(
    t,
    leader_v,
    crisp,
    fuzzy_alpha0_raw,
    analysis_start=20.0
):
    """
    使用周期扰动场景分析交通流稳定性。

    ----------------------------------------------------------
    定义速度扰动放大系数：
    ----------------------------------------------------------

        G_i
        =
        σ(V_i)
        /
        σ(V_leader)

    其中 σ 表示去均值后的速度标准差。

    如果：

        G_i < 1

    表示速度扰动沿车队传播时被衰减。

    如果：

        G_i > 1

    表示扰动出现放大趋势。


    注意：
    ----------------------------------------------------------
    这是基于数值仿真的“扰动传播指标”，
    可用于讨论交通流稳定性，
    但不能等价于严格的解析 string stability 证明。
    """

    indices = np.where(
        t >= analysis_start
    )[0]


    # --------------------------------------------------------
    # 前车速度标准差
    # --------------------------------------------------------

    leader_std = np.std(
        leader_v[
            indices
        ]
    )


    # --------------------------------------------------------
    # Crisp IDM：
    # 每辆跟驰车速度标准差
    # --------------------------------------------------------

    crisp_velocity = np.take(

        crisp["v"][0],

        indices,

        axis=1
    )

    crisp_speed_std = np.std(
        crisp_velocity,
        axis=1
    )


    crisp_gain = (
        crisp_speed_std
        /
        leader_std
    )


    # --------------------------------------------------------
    # Fuzzy-IDM α=0：
    #
    # 每一组 a,b,T 参数组合，
    # 都得到一个速度扰动放大系数。
    # --------------------------------------------------------

    fuzzy_velocity = np.take(

        fuzzy_alpha0_raw["v"],

        indices,

        axis=2
    )


    fuzzy_speed_std = np.std(
        fuzzy_velocity,
        axis=2
    )


    fuzzy_gain = (
        fuzzy_speed_std
        /
        leader_std
    )


    gain_lower = np.min(
        fuzzy_gain,
        axis=0
    )

    gain_upper = np.max(
        fuzzy_gain,
        axis=0
    )


    # --------------------------------------------------------
    # 加速度标准差
    #
    # 用于衡量车辆跟驰行为的波动性。
    # --------------------------------------------------------

    crisp_acc = np.take(

        crisp["acc"][0],

        indices,

        axis=1
    )

    crisp_acc_std = np.std(
        crisp_acc,
        axis=1
    )


    fuzzy_acc = np.take(

        fuzzy_alpha0_raw["acc"],

        indices,

        axis=2
    )

    fuzzy_acc_std = np.std(
        fuzzy_acc,
        axis=2
    )


    acc_std_lower = np.min(
        fuzzy_acc_std,
        axis=0
    )

    acc_std_upper = np.max(
        fuzzy_acc_std,
        axis=0
    )


    return {
        "crisp_gain": crisp_gain,
        "gain_lower": gain_lower,
        "gain_upper": gain_upper,

        "crisp_acc_std": crisp_acc_std,
        "acc_std_lower": acc_std_lower,
        "acc_std_upper": acc_std_upper
    }


# ============================================================
# 21. 绘制速度扰动放大系数
# ============================================================

def plot_stability_gain(
    metrics
):
    """
    横轴：
        前车 + 5 辆跟驰车辆

    纵轴：
        速度扰动放大系数 G
    """

    vehicle_numbers = np.arange(
        0,
        N_FOLLOWERS + 1
    )


    # 前车自身 G=1
    crisp_gain = np.concatenate(
        (
            [1.0],
            metrics[
                "crisp_gain"
            ]
        )
    )


    fuzzy_lower = np.concatenate(
        (
            [1.0],
            metrics[
                "gain_lower"
            ]
        )
    )


    fuzzy_upper = np.concatenate(
        (
            [1.0],
            metrics[
                "gain_upper"
            ]
        )
    )


    plt.figure(
        figsize=(8.5, 5.5)
    )


    plt.fill_between(
        vehicle_numbers,
        fuzzy_lower,
        fuzzy_upper,
        alpha=0.22,
        label="Fuzzy-IDM α=0 范围"
    )


    plt.plot(
        vehicle_numbers,
        crisp_gain,
        marker="o",
        linewidth=2,
        label="确定性 IDM"
    )


    # G=1 稳定性参考线
    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="扰动不放大边界 G=1"
    )


    plt.xticks(
        vehicle_numbers,
        [
            "前车",
            "跟驰车1",
            "跟驰车2",
            "跟驰车3",
            "跟驰车4",
            "跟驰车5"
        ]
    )


    plt.xlabel(
        "车辆序号"
    )

    plt.ylabel(
        "速度扰动放大系数 G"
    )

    plt.title(
        "周期扰动场景：速度扰动沿车队传播"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        "periodic_stability_gain.png"
    )


# ============================================================
# 22. 跟驰行为波动性
# ============================================================

def plot_acceleration_fluctuation(
    metrics
):
    """
    用加速度标准差衡量车辆控制行为波动程度。

    标准差越大：
        表示车辆加速、减速变化越明显。
    """

    vehicles = np.arange(
        1,
        N_FOLLOWERS + 1
    )


    plt.figure(
        figsize=(8.5, 5.5)
    )


    plt.fill_between(
        vehicles,

        metrics[
            "acc_std_lower"
        ],

        metrics[
            "acc_std_upper"
        ],

        alpha=0.22,

        label="Fuzzy-IDM α=0 范围"
    )


    plt.plot(
        vehicles,

        metrics[
            "crisp_acc_std"
        ],

        marker="o",

        linewidth=2,

        label="确定性 IDM"
    )


    plt.xticks(
        vehicles,
        [
            "跟驰车1",
            "跟驰车2",
            "跟驰车3",
            "跟驰车4",
            "跟驰车5"
        ]
    )


    plt.xlabel(
        "车辆序号"
    )

    plt.ylabel(
        "加速度标准差 / (m/s²)"
    )

    plt.title(
        "周期扰动场景：跟驰行为波动性"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        "periodic_acceleration_fluctuation.png"
    )


# ============================================================
# 23. α=1 与 Crisp IDM 一致性检查
# ============================================================

def check_alpha_one(
    crisp,
    fuzzy
):
    """
    α=1 时：

        a_max = 1.0
        a_comf = 1.5
        T = 1.5

    因此理论上应该完全退化为 Crisp IDM。
    """

    fuzzy_alpha1 = fuzzy[
        1.0
    ]


    speed_error = np.max(
        np.abs(
            crisp["v"][0]
            -
            fuzzy_alpha1[
                "v_lower"
            ]
        )
    )


    gap_error = np.max(
        np.abs(
            crisp["gap"][0]
            -
            fuzzy_alpha1[
                "gap_lower"
            ]
        )
    )


    print(
        "\nα=1 一致性检查"
    )

    print(
        "最大速度误差："
        f"{speed_error:.10e} m/s"
    )

    print(
        "最大间距误差："
        f"{gap_error:.10e} m"
    )


# ============================================================
# 24. 安全间距检查
# ============================================================

def check_gap_safety(
    crisp,
    fuzzy,
    scene_name
):
    """
    检查仿真过程中是否出现车辆碰撞。
    """

    crisp_min_gap = np.min(
        crisp["gap"]
    )

    fuzzy_min_gap = np.min(
        fuzzy[0.0][
            "gap_lower"
        ]
    )


    print(
        f"\n{scene_name}："
    )

    print(
        f"确定性 IDM 最小净间距："
        f"{crisp_min_gap:.3f} m"
    )

    print(
        f"Fuzzy-IDM α=0 最小净间距："
        f"{fuzzy_min_gap:.3f} m"
    )


    if fuzzy_min_gap <= 0:

        print(
            "警告：某些模糊参数组合产生了非正间距，"
            "需要检查参数范围或模型安全约束。"
        )

    else:

        print(
            "未检测到车辆碰撞。"
        )


# ============================================================
# 25. 输出周期扰动稳定性结果
# ============================================================

def print_stability_metrics(
    metrics
):
    """
    输出最后一辆跟驰车的关键稳定性指标。
    """

    last = (
        N_FOLLOWERS - 1
    )


    crisp_G = metrics[
        "crisp_gain"
    ][last]


    fuzzy_G_lower = metrics[
        "gain_lower"
    ][last]


    fuzzy_G_upper = metrics[
        "gain_upper"
    ][last]


    crisp_acc_std = metrics[
        "crisp_acc_std"
    ][last]


    fuzzy_acc_lower = metrics[
        "acc_std_lower"
    ][last]


    fuzzy_acc_upper = metrics[
        "acc_std_upper"
    ][last]


    print(
        "\n========================================"
    )

    print(
        "周期扰动场景稳定性分析"
    )

    print(
        "========================================"
    )


    print(
        "\n最后一辆跟驰车速度扰动放大系数："
    )

    print(
        f"Crisp IDM："
        f"{crisp_G:.4f}"
    )

    print(
        f"Fuzzy-IDM α=0："
        f"[{fuzzy_G_lower:.4f}, "
        f"{fuzzy_G_upper:.4f}]"
    )


    print(
        "\n最后一辆跟驰车加速度标准差："
    )

    print(
        f"Crisp IDM："
        f"{crisp_acc_std:.4f} m/s²"
    )

    print(
        f"Fuzzy-IDM α=0："
        f"[{fuzzy_acc_lower:.4f}, "
        f"{fuzzy_acc_upper:.4f}] m/s²"
    )


    print(
        "\n稳定性判断："
    )


    # --------------------------------------------------------
    # 如果整个模糊区间都低于1：
    # 所有当前采样参数组合都表现为扰动衰减。
    # --------------------------------------------------------

    if fuzzy_G_upper < 1.0:

        print(
            "Fuzzy-IDM 的扰动放大系数上界仍小于 1，"
            "说明在当前参数范围和该周期扰动下，"
            "速度扰动总体沿车队衰减。"
        )


    # --------------------------------------------------------
    # 如果区间跨越1：
    # 参数不确定性改变了稳定性判断。
    # --------------------------------------------------------

    elif (
        fuzzy_G_lower
        <
        1.0
        <
        fuzzy_G_upper
    ):

        print(
            "Fuzzy-IDM 的扰动放大系数区间跨越 G=1，"
            "说明参数不确定性可能使系统在扰动衰减和扰动放大"
            "两种状态之间变化。"
        )


    # --------------------------------------------------------
    # 如果整个区间均大于1：
    # 所有当前采样组合都表现出扰动放大趋势。
    # --------------------------------------------------------

    else:

        print(
            "Fuzzy-IDM 的扰动放大系数下界已大于 1，"
            "说明当前参数范围下存在明显的扰动放大趋势。"
        )


# ============================================================
# 26. 单个场景完整运行函数
# ============================================================

def run_scenario(
    scenario
):
    """
    完整运行：

    1. 前车轨迹
    2. Crisp IDM
    3. 三参数 Fuzzy-IDM
    4. 结果检查
    5. 绘制结果
    """

    scene_names = {
        "accelerating": "匀加速前车场景",
        "braking": "前车减速停车场景",
        "periodic": "周期性扰动场景"
    }


    file_prefixes = {
        "accelerating": "accelerating",
        "braking": "braking",
        "periodic": "periodic"
    }


    # 不同场景使用不同仿真时长
    if scenario == "periodic":
        t_end = 80.0
    else:
        t_end = 50.0


    t = np.arange(
        0.0,
        t_end + DT,
        DT
    )


    scene_name = scene_names[
        scenario
    ]

    file_prefix = file_prefixes[
        scenario
    ]


    print(
        "\n\n"
        "========================================"
    )

    print(
        f"开始仿真：{scene_name}"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # 前车轨迹
    # --------------------------------------------------------

    leader_v, leader_x = (
        generate_leader_profile(
            scenario,
            t
        )
    )


    # --------------------------------------------------------
    # Crisp IDM
    # --------------------------------------------------------

    print(
        "\n正在运行确定性 IDM..."
    )

    crisp = simulate_crisp_idm(
        t,
        leader_v,
        leader_x
    )


    # --------------------------------------------------------
    # Fuzzy-IDM
    # --------------------------------------------------------

    print(
        "\n正在运行三参数 Fuzzy-IDM..."
    )

    fuzzy, raw_results = (
        simulate_fuzzy_idm(
            t,
            leader_v,
            leader_x
        )
    )


    # --------------------------------------------------------
    # 检验 α=1 是否退化为 Crisp IDM
    # --------------------------------------------------------

    check_alpha_one(
        crisp,
        fuzzy
    )


    # --------------------------------------------------------
    # 安全间距检查
    # --------------------------------------------------------

    check_gap_safety(
        crisp,
        fuzzy,
        scene_name
    )


    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    plot_velocity(
        t,
        leader_v,
        crisp,
        fuzzy,
        scene_name,
        file_prefix
    )


    plot_gap(
        t,
        crisp,
        fuzzy,
        scene_name,
        file_prefix
    )


    plot_acceleration(
        t,
        crisp,
        fuzzy,
        scene_name,
        file_prefix
    )


    plot_phase_diagram(
        raw_results,
        crisp,
        scene_name,
        file_prefix
    )


    plot_uncertainty_width(
        t,
        fuzzy,
        scene_name,
        file_prefix
    )


    # --------------------------------------------------------
    # 周期扰动：
    # 额外进行交通流稳定性与波动性分析
    # --------------------------------------------------------

    if scenario == "periodic":

        metrics = (
            calculate_stability_metrics(
                t,
                leader_v,
                crisp,
                raw_results[0.0],
                analysis_start=20.0
            )
        )


        plot_stability_gain(
            metrics
        )


        plot_acceleration_fluctuation(
            metrics
        )


        print_stability_metrics(
            metrics
        )


    return {
        "t": t,
        "leader_v": leader_v,
        "leader_x": leader_x,
        "crisp": crisp,
        "fuzzy": fuzzy,
        "raw": raw_results
    }


# ============================================================
# 27. 主程序
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "Crisp IDM 与三参数 Fuzzy-IDM 数值仿真"
    )

    print(
        "========================================"
    )


    print(
        "\n确定性参数："
    )

    print(
        f"期望速度 Ṽ = "
        f"{V_DES_KMH:.1f} km/h"
    )

    print(
        f"静止间距 S_jam = "
        f"{S_JAM:.1f} m"
    )

    print(
        f"加速度指数 β = "
        f"{BETA:.1f}"
    )


    print(
        "\n三参数模糊数："
    )

    print(
        f"a_max = {A_MAX_TRI} m/s²"
    )

    print(
        f"a_comf = {A_COMF_TRI} m/s²"
    )

    print(
        f"T = {T_TRI} s"
    )


    # --------------------------------------------------------
    # 场景1：匀加速前车
    # --------------------------------------------------------

    result_accelerating = (
        run_scenario(
            "accelerating"
        )
    )


    # --------------------------------------------------------
    # 场景2：前车减速停车
    # --------------------------------------------------------

    result_braking = (
        run_scenario(
            "braking"
        )
    )


    # --------------------------------------------------------
    # 场景3：周期性扰动
    # --------------------------------------------------------

    result_periodic = (
        run_scenario(
            "periodic"
        )
    )


    print(
        "\n========================================"
    )

    print(
        "全部仿真完成"
    )

    print(
        "========================================"
    )


    if SAVE_FIG:

        print(
            f"\n所有图片已保存到："
            f"{OUTPUT_DIR}"
        )


    # 所有图片最后统一显示
    if SHOW_FIG:
        plt.show()