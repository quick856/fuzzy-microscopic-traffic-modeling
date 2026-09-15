import os
import math
import csv
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

    font_paths = [
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\Dengl.ttf",
        r"C:\Windows\Fonts\Dengb.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]

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

    candidate_fonts = [
        "DengXian",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN"
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
        "警告：未检测到常用中文字体。"
        "如果中文显示为方框，请安装等线、微软雅黑或思源黑体。"
    )


setup_chinese_font()


# ============================================================
# 1. 全局设置
# ============================================================

SAVE_FIG = True
SHOW_FIG = True

OUTPUT_DIR = "fuzzy_idm_results_v2"

if SAVE_FIG:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# 时间步长
DT = 0.05


# 跟驰车辆数量
N_FOLLOWERS = 5


# α 水平
ALPHA_LEVELS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00
]


# 三参数 α-截集离散采样数量
GRID_N = 5


# 是否进行 GRID_N 收敛性验证
RUN_GRID_CONVERGENCE = True


# ============================================================
# 2. 初始条件模式
# ============================================================
#
# 推荐：
#
# "parameter_equilibrium"
#
# 每一组 a、b、T 参数组合，
# 均按照自身的 IDM 平衡间距进行初始化。
#
# 优点：
# t=0 时各参数组合均满足平衡状态，
# 可以尽量避免由于初始状态不匹配造成的人工瞬态。
#
#
# 如果希望：
# 所有模糊参数组合从完全相同的实际位置和速度出发，
# 可以改成：
#
# "common_crisp_state"gap_history
#
# ============================================================

INITIALIZATION_MODE = "common_crisp_state"


# ============================================================
# 3. 周期扰动实验设置
# ============================================================

# 前车平均速度
PERIODIC_MEAN_KMH = 72.0

# 周期扰动振幅
PERIODIC_AMPLITUDE_KMH = 7.2

# 角频率 rad/s
PERIODIC_OMEGA = 0.2

# 一个扰动周期
PERIODIC_PERIOD = (
    2.0
    * np.pi
    / PERIODIC_OMEGA
)

# 前60秒保持匀速，让车队先达到稳定状态
PERIODIC_WARMUP = 60.0

# 正弦扰动开始后，
# 前2个周期视为过渡过程，不用于稳定性统计
PERIODIC_TRANSIENT_CYCLES = 2

# 后5个完整周期用于统计
PERIODIC_ANALYSIS_CYCLES = 5

# 周期场景总时间
PERIODIC_END = (
    PERIODIC_WARMUP
    +
    (
        PERIODIC_TRANSIENT_CYCLES
        +
        PERIODIC_ANALYSIS_CYCLES
    )
    *
    PERIODIC_PERIOD
)

# 稳定性分析起点
PERIODIC_ANALYSIS_START = (
    PERIODIC_WARMUP
    +
    PERIODIC_TRANSIENT_CYCLES
    *
    PERIODIC_PERIOD
)


# ============================================================
# 4. 单位转换
# ============================================================

def kmh_to_ms(v):
    """
    km/h → m/s
    """

    return np.asarray(
        v,
        dtype=float
    ) / 3.6


def ms_to_kmh(v):
    """
    m/s → km/h
    """

    return np.asarray(
        v,
        dtype=float
    ) * 3.6


# ============================================================
# 5. IDM 固定参数
# ============================================================
#
# 严格对应所采用的 IDM：
#
# a_n(t)
# =
# a_max [
#     1
#     - (V_n / V_des)^β
#     - (S_des / S_n)^2
# ]
#
#
# S_des
# =
# S_jam
# +
# max[
#     0,
#     V_n T
#     -
#     V_n ΔV_n
#     /
#     (2 sqrt(a_max a_comf))
# ]
#
#
# ΔV_n
# =
# V_(n-1)
# -
# V_n
#
# ============================================================


# 期望速度 Ṽ_n(t)
V_DES_KMH = 108.0
V_DES = float(
    kmh_to_ms(
        V_DES_KMH
    )
)


# 静止期望间距
S_JAM = 2.0


# 加速度指数
BETA = 4.0


# 车辆长度
VEHICLE_LENGTH = 5.0


# ============================================================
# 6. 三参数模糊数
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


# Crisp IDM 使用三角模糊数中心值
A_MAX_CRISP = A_MAX_TRI[1]
A_COMF_CRISP = A_COMF_TRI[1]
T_CRISP = T_TRI[1]


# ============================================================
# 7. 三角模糊数 α-截集
# ============================================================

def triangular_alpha_cut(
    triangle,
    alpha
):
    """
    三角模糊数：

        (left, center, right)

    α-截集：

        [
            left + α(center-left),
            right - α(right-center)
        ]
    """

    left, center, right = triangle

    lower = (
        left
        +
        alpha
        *
        (
            center
            -
            left
        )
    )

    upper = (
        right
        -
        alpha
        *
        (
            right
            -
            center
        )
    )

    return lower, upper


# ============================================================
# 8. 构造 Ω_α 参数空间
# ============================================================

def create_parameter_grid(
    alpha,
    grid_n=GRID_N
):
    """
    对给定 α 水平构造：

        Ω_α
        =
        [a_max]_α
        ×
        [a_comf]_α
        ×
        [T]_α

    每个参数区间均匀取 grid_n 个点。
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


    def make_samples(
        lower,
        upper
    ):

        if np.isclose(
            lower,
            upper
        ):
            return np.array(
                [
                    (
                        lower
                        +
                        upper
                    )
                    /
                    2.0
                ]
            )

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
# 9. IDM 加速度函数
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
    IDM 加速度函数。

    相对速度：

        ΔV_n
        =
        V_(n-1)
        -
        V_n

    因此：

        ΔV < 0
        表示跟驰车速度高于前车，
        两车距离正在减小。
    """


    # 数值保护，避免除零
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

            v_follower
            *
            T

            -

            (
                v_follower
                *
                delta_v
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


    # IDM 加速度
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
# 10. IDM 平衡间距
# ============================================================

def equilibrium_gap(
    speed,
    T
):
    """
    在：

        ΔV = 0
        a_n = 0

    条件下求 IDM 平衡净间距。

    当 ΔV=0 时：

        S_des
        =
        S_jam
        +
        V T

    平衡条件：

        1
        -
        (V/V_des)^β
        -
        (S_des/S)^2
        =
        0

    所以：

        S
        =
        S_des
        /
        sqrt(
            1-(V/V_des)^β
        )

    注意：

    平衡间距与 T 有关，
    但此时不直接依赖 a_max 与 a_comf。
    """

    speed = np.asarray(
        speed,
        dtype=float
    )

    T = np.asarray(
        T,
        dtype=float
    )


    denominator = (
        1.0
        -
        (
            speed
            /
            V_DES
        ) ** BETA
    )


    denominator = np.maximum(
        denominator,
        1e-8
    )


    desired_gap = (
        S_JAM
        +
        speed
        *
        T
    )


    return (
        desired_gap
        /
        np.sqrt(
            denominator
        )
    )


# ============================================================
# 11. 前车轨迹
# ============================================================

def generate_leader_profile(
    scenario,
    t
):
    """
    三种场景：

    1. accelerating
       前车匀加速后保持恒速

    2. braking
       前车减速直至停车

    3. periodic
       先匀速预热，再施加周期扰动
    """


    # --------------------------------------------------------
    # 场景1：匀加速
    # --------------------------------------------------------

    if scenario == "accelerating":

        v_initial = float(
            kmh_to_ms(
                54.0
            )
        )

        leader_acc = 0.5

        v_max = float(
            kmh_to_ms(
                72.0
            )
        )

        leader_v = np.minimum(
            v_initial
            +
            leader_acc
            *
            t,
            v_max
        )


    # --------------------------------------------------------
    # 场景2：减速停车
    # --------------------------------------------------------

    elif scenario == "braking":

        v_initial = float(
            kmh_to_ms(
                72.0
            )
        )

        braking_start = 10.0

        deceleration = 1.5


        leader_v = np.where(

            t
            <
            braking_start,

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
    # 场景3：周期扰动
    # --------------------------------------------------------

    elif scenario == "periodic":

        mean_speed = float(
            kmh_to_ms(
                PERIODIC_MEAN_KMH
            )
        )

        amplitude = float(
            kmh_to_ms(
                PERIODIC_AMPLITUDE_KMH
            )
        )


        leader_v = np.full_like(
            t,
            mean_speed,
            dtype=float
        )


        disturbance_mask = (
            t
            >=
            PERIODIC_WARMUP
        )


        disturbance_time = (
            t[
                disturbance_mask
            ]
            -
            PERIODIC_WARMUP
        )


        # 预热结束时正弦从0开始，
        # 这样速度在扰动开始点连续。
        leader_v[
            disturbance_mask
        ] = (
            mean_speed
            +
            amplitude
            *
            np.sin(
                PERIODIC_OMEGA
                *
                disturbance_time
            )
        )


    else:

        raise ValueError(
            f"未知场景：{scenario}"
        )


    # --------------------------------------------------------
    # 积分得到前车位置
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
# 12. 构造初始车队
# ============================================================

def create_initial_platoon(
    leader_initial_speed,
    T_values,
    n_followers
):
    """
    根据 INITIALIZATION_MODE 构造初始位置。


    parameter_equilibrium：
    ----------------------
    每个参数组合使用自己的 T，
    因而具有自己的初始平衡间距。


    common_crisp_state：
    -------------------
    所有参数组合使用 Crisp T=1.5
    对应的相同初始间距。
    """


    n_parameter_sets = len(
        T_values
    )


    if (
        INITIALIZATION_MODE
        ==
        "parameter_equilibrium"
    ):

        gaps = equilibrium_gap(
            leader_initial_speed,
            T_values
        )

    elif (
        INITIALIZATION_MODE
        ==
        "common_crisp_state"
    ):

        crisp_gap = float(
            equilibrium_gap(
                leader_initial_speed,
                T_CRISP
            )
        )

        gaps = np.full(
            n_parameter_sets,
            crisp_gap
        )

    else:

        raise ValueError(
            "INITIALIZATION_MODE 必须为 "
            "'parameter_equilibrium' "
            "或 'common_crisp_state'"
        )


    x = np.zeros(
        (
            n_parameter_sets,
            n_followers
        )
    )


    # 每个参数组合均按自己的平衡间距
    # 构造整列车队。
    for j in range(
        n_followers
    ):

        x[:, j] = (
            -
            (
                j
                +
                1
            )
            *
            (
                VEHICLE_LENGTH
                +
                gaps
            )
        )


    return x


# ============================================================
# 13. 核心车队仿真
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
    可同时运行一组或多组 IDM 参数。

    若只有一个参数组合：
        → Crisp IDM

    若有 Ω_α 中的多组参数：
        → Fuzzy-IDM α-截集轨迹集合
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
    # 所有车辆初始速度均等于前车
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
    # 初始位置
    # --------------------------------------------------------

    x = create_initial_platoon(
        leader_initial_speed=leader_v[0],
        T_values=T_values,
        n_followers=n_followers
    )


    # --------------------------------------------------------
    # 保存轨迹
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
        # 同一时刻下先计算全部车辆加速度
        # ----------------------------------------------------

        for j in range(
            n_followers
        ):


            if j == 0:

                v_leader_now = (
                    leader_v[k]
                )

                x_leader_now = (
                    leader_x[k]
                )

            else:

                v_leader_now = (
                    v[:, j - 1]
                )

                x_leader_now = (
                    x[:, j - 1]
                )


            gap = (
                x_leader_now
                -
                x[:, j]
                -
                VEHICLE_LENGTH
            )


            current_gap[:, j] = gap


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


        if k == n_time - 1:
            break


        # ----------------------------------------------------
        # 更新速度
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
        # 梯形积分：
        #
        # x(k+1)
        # =
        # x(k)
        # +
        # 0.5[v(k)+v(k+1)]Δt
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
# 14. Crisp IDM
# ============================================================

def simulate_crisp_idm(
    t,
    leader_v,
    leader_x
):

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
# 15. 提取 Fuzzy-IDM α-截集包络
# ============================================================

def extract_fuzzy_envelope(
    raw_result
):

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
            variable
            +
            "_lower"
        ] = np.min(
            data,
            axis=0
        )


        result[
            variable
            +
            "_upper"
        ] = np.max(
            data,
            axis=0
        )


    return result


# ============================================================
# 16. 三参数 Fuzzy-IDM
# ============================================================

def simulate_fuzzy_idm(
    t,
    leader_v,
    leader_x,
    grid_n=GRID_N
):
    """
    对各 α：

    Ω_α
    ↓
    参数离散采样
    ↓
    求解确定性 IDM ODE 族
    ↓
    逐时刻 min/max
    ↓
    数值近似 α-截集
    """


    fuzzy_results = {}

    raw_results = {}


    for alpha in ALPHA_LEVELS:

        print(
            f"正在计算 α={alpha:.2f}"
        )


        (
            a_values,
            b_values,
            T_values
        ) = create_parameter_grid(
            alpha,
            grid_n=grid_n
        )


        print(
            f"    参数组合："
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


        # 相图/状态分布仅需要保存几个代表 α
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
# 17. 图片保存
# ============================================================

def finish_figure(
    filename
):

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
# 18. 速度演化
# ============================================================

def plot_velocity(
    t,
    leader_v,
    crisp,
    fuzzy,
    scene_name,
    file_prefix,
    vehicle_indices=None
):

    if vehicle_indices is None:

        vehicle_indices = [
            0
        ]


    plt.figure(
        figsize=(9, 5.5)
    )


    for vehicle_index in vehicle_indices:

        vehicle_number = (
            vehicle_index
            +
            1
        )


        # α=0
        plt.fill_between(

            t,

            ms_to_kmh(
                fuzzy[0.0][
                    "v_lower"
                ][vehicle_index]
            ),

            ms_to_kmh(
                fuzzy[0.0][
                    "v_upper"
                ][vehicle_index]
            ),

            alpha=0.12,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0"
        )


        # α=0.5
        plt.fill_between(

            t,

            ms_to_kmh(
                fuzzy[0.5][
                    "v_lower"
                ][vehicle_index]
            ),

            ms_to_kmh(
                fuzzy[0.5][
                    "v_upper"
                ][vehicle_index]
            ),

            alpha=0.20,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0.5"
        )


        plt.plot(

            t,

            ms_to_kmh(
                crisp[
                    "v"
                ][
                    0,
                    vehicle_index
                ]
            ),

            linewidth=1.2,

            label=f"跟驰车{vehicle_number} 确定性 IDM"
        )


    plt.plot(

        t,

        ms_to_kmh(
            leader_v
        ),

        linestyle="--",

        linewidth=1.8,

        label="前车"
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
# 19. 间距演化
# ============================================================

def plot_gap(
    t,
    crisp,
    fuzzy,
    scene_name,
    file_prefix,
    vehicle_indices=None
):

    if vehicle_indices is None:

        vehicle_indices = [
            N_FOLLOWERS
            -
            1
        ]


    plt.figure(
        figsize=(9, 5.5)
    )


    for vehicle_index in vehicle_indices:

        vehicle_number = (
            vehicle_index
            +
            1
        )


        plt.fill_between(

            t,

            fuzzy[0.0][
                "gap_lower"
            ][vehicle_index],

            fuzzy[0.0][
                "gap_upper"
            ][vehicle_index],

            alpha=0.12,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0"
        )


        plt.fill_between(

            t,

            fuzzy[0.5][
                "gap_lower"
            ][vehicle_index],

            fuzzy[0.5][
                "gap_upper"
            ][vehicle_index],

            alpha=0.20,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0.5"
        )


        plt.plot(

            t,

            crisp[
                "gap"
            ][
                0,
                vehicle_index
            ],

            linewidth=1.2,

            label=f"跟驰车{vehicle_number} 确定性 IDM"
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
# 20. 加速度演化
# ============================================================

def plot_acceleration(
    t,
    crisp,
    fuzzy,
    scene_name,
    file_prefix,
    vehicle_indices=None
):

    if vehicle_indices is None:

        vehicle_indices = [
            N_FOLLOWERS
            -
            1
        ]


    plt.figure(
        figsize=(9, 5.5)
    )


    for vehicle_index in vehicle_indices:

        vehicle_number = (
            vehicle_index
            +
            1
        )


        plt.fill_between(

            t,

            fuzzy[0.0][
                "acc_lower"
            ][vehicle_index],

            fuzzy[0.0][
                "acc_upper"
            ][vehicle_index],

            alpha=0.12,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0"
        )


        plt.fill_between(

            t,

            fuzzy[0.5][
                "acc_lower"
            ][vehicle_index],

            fuzzy[0.5][
                "acc_upper"
            ][vehicle_index],

            alpha=0.20,

            label=f"跟驰车{vehicle_number} Fuzzy-IDM：α=0.5"
        )


        plt.plot(

            t,

            crisp[
                "acc"
            ][
                0,
                vehicle_index
            ],

            linewidth=1.2,

            label=f"跟驰车{vehicle_number} 确定性 IDM"
        )


    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0
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
# 21. 速度—间距状态分布
# ============================================================

def plot_phase_diagram(
    t,
    raw_results,
    crisp,
    scene_name,
    file_prefix,
    start_time=0.0,
    vehicle_indices=None
):
    """
    原来的“速度—间距相图”改为：

        速度—间距状态分布

    原因：
    Fuzzy-IDM 图中包含多个参数组合产生的状态点，
    并不是单一确定性相轨迹。
    """


    if vehicle_indices is None:

        vehicle_indices = [
            N_FOLLOWERS
            -
            1
        ]


    plt.figure(
        figsize=(8, 6)
    )


    start_index = np.searchsorted(
        t,
        start_time
    )


    # 长时间周期场景适当降低点密度
    if len(t) > 3000:
        step = 30
    else:
        step = 10


    for vehicle_index in vehicle_indices:

        vehicle_number = (
            vehicle_index
            +
            1
        )

        for alpha in [
            0.0,
            0.5
        ]:

            raw = raw_results[
                alpha
            ]


            gap = (
                raw[
                    "gap"
                ][
                    :,
                    vehicle_index,
                    start_index::step
                ]
            ).ravel()


            velocity = (
                raw[
                    "v"
                ][
                    :,
                    vehicle_index,
                    start_index::step
                ]
                *
                3.6
            ).ravel()


            plt.scatter(
                gap,
                velocity,
                s=6,
                alpha=0.08,
                label=f"跟驰车{vehicle_number} Fuzzy-IDM：α={alpha}"
            )


        plt.plot(

            crisp[
                "gap"
            ][
                0,
                vehicle_index,
                start_index:
            ],

            ms_to_kmh(
                crisp[
                    "v"
                ][
                    0,
                    vehicle_index,
                    start_index:
                ]
            ),

            linewidth=1.2,

            label=f"跟驰车{vehicle_number} α=1 / 确定性 IDM"
        )


    plt.xlabel(
        "实际净间距 / m"
    )

    plt.ylabel(
        "速度 / (km/h)"
    )

    plt.title(
        f"{scene_name}：速度—间距状态分布"
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
# 22. 速度模糊区间宽度
# ============================================================

def plot_uncertainty_width(
    t,
    fuzzy,
    scene_name,
    file_prefix,
    vehicle_indices=None
):

    if vehicle_indices is None:

        vehicle_indices = [
            N_FOLLOWERS
            -
            1
        ]


    plt.figure(
        figsize=(9, 5.5)
    )


    for vehicle_index in vehicle_indices:

        vehicle_number = (
            vehicle_index
            +
            1
        )

        linestyle = "-"

        if vehicle_number == N_FOLLOWERS:

            linestyle = "--"

        for color_index, alpha in enumerate(
            ALPHA_LEVELS
        ):

            lower = fuzzy[
                alpha
            ][
                "v_lower"
            ][
                vehicle_index
            ]


            upper = fuzzy[
                alpha
            ][
                "v_upper"
            ][
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
                linestyle=linestyle,
                color=f"C{color_index}",
                label=f"跟驰车{vehicle_number} α={alpha}"
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
# 23. α-截集嵌套性检查
# ============================================================

def check_alpha_nesting(
    fuzzy,
    scene_name,
    tolerance=1e-9
):
    """
    检查：

        α 越大，
        α-截集是否越窄。

    即：

        [V]_1
        ⊆
        [V]_0.75
        ⊆
        ...
        ⊆
        [V]_0
    """


    print(
        f"\n{scene_name}：α-截集嵌套性检查"
    )


    all_ok = True


    for i in range(
        len(ALPHA_LEVELS) - 1
    ):

        alpha_low = ALPHA_LEVELS[i]

        alpha_high = ALPHA_LEVELS[i + 1]


        low_L = fuzzy[
            alpha_low
        ][
            "v_lower"
        ]

        low_U = fuzzy[
            alpha_low
        ][
            "v_upper"
        ]


        high_L = fuzzy[
            alpha_high
        ][
            "v_lower"
        ]

        high_U = fuzzy[
            alpha_high
        ][
            "v_upper"
        ]


        lower_ok = np.all(
            high_L
            >=
            low_L
            -
            tolerance
        )


        upper_ok = np.all(
            high_U
            <=
            low_U
            +
            tolerance
        )


        pair_ok = (
            lower_ok
            and
            upper_ok
        )


        print(
            f"α={alpha_high:.2f} "
            f"⊆ "
            f"α={alpha_low:.2f}："
            f"{'通过' if pair_ok else '未通过'}"
        )


        if not pair_ok:
            all_ok = False


    if all_ok:

        print(
            "全部 α-截集满足嵌套关系。"
        )

    else:

        print(
            "警告：存在 α-截集嵌套性异常，"
            "需要增加参数采样密度或检查数值实现。"
        )


# ============================================================
# 24. α=1 与 Crisp IDM 一致性
# ============================================================

def check_alpha_one(
    crisp,
    fuzzy
):

    alpha1 = fuzzy[
        1.0
    ]


    speed_error = np.max(
        np.abs(
            crisp[
                "v"
            ][0]
            -
            alpha1[
                "v_lower"
            ]
        )
    )


    gap_error = np.max(
        np.abs(
            crisp[
                "gap"
            ][0]
            -
            alpha1[
                "gap_lower"
            ]
        )
    )


    acc_error = np.max(
        np.abs(
            crisp[
                "acc"
            ][0]
            -
            alpha1[
                "acc_lower"
            ]
        )
    )


    print(
        "\nα=1 与 Crisp IDM 一致性检查"
    )

    print(
        f"最大速度误差："
        f"{speed_error:.3e} m/s"
    )

    print(
        f"最大间距误差："
        f"{gap_error:.3e} m"
    )

    print(
        f"最大加速度误差："
        f"{acc_error:.3e} m/s²"
    )


# ============================================================
# 25. 安全性检查
# ============================================================

def check_gap_safety(
    crisp,
    fuzzy,
    scene_name
):

    crisp_min_gap = np.min(
        crisp[
            "gap"
        ]
    )


    fuzzy_min_gap = np.min(
        fuzzy[
            0.0
        ][
            "gap_lower"
        ]
    )


    print(
        f"\n{scene_name}：安全间距检查"
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
            "警告：存在非正净间距，可能发生碰撞。"
        )

    else:

        print(
            "所有采样轨迹均保持正净间距。"
        )


# ============================================================
# 26. 周期稳定性指标
# ============================================================

def calculate_stability_metrics(
    t,
    leader_v,
    crisp,
    fuzzy_alpha0_raw
):
    """
    只使用稳定周期区间：

        t >= PERIODIC_ANALYSIS_START

    统计最后5个完整周期。


    速度扰动放大系数：

        G_i
        =
        σ(V_i)
        /
        σ(V_leader)


    G < 1：
        当前周期扰动被衰减

    G > 1：
        当前周期扰动被放大


    注意：
    这是数值扰动传播指标，
    不是严格的解析 string stability 证明。
    """


    analysis_mask = (
        t
        >=
        PERIODIC_ANALYSIS_START
    )


    indices = np.where(
        analysis_mask
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
    # Crisp IDM
    # --------------------------------------------------------

    crisp_velocity = np.take(
        crisp[
            "v"
        ][0],
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
    # Fuzzy-IDM α=0
    # --------------------------------------------------------

    fuzzy_velocity = np.take(
        fuzzy_alpha0_raw[
            "v"
        ],
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
    # 加速度波动性
    # --------------------------------------------------------

    crisp_acc = np.take(
        crisp[
            "acc"
        ][0],
        indices,
        axis=1
    )


    crisp_acc_std = np.std(
        crisp_acc,
        axis=1
    )


    fuzzy_acc = np.take(
        fuzzy_alpha0_raw[
            "acc"
        ],
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

        "crisp_gain":
            crisp_gain,

        "gain_lower":
            gain_lower,

        "gain_upper":
            gain_upper,

        "crisp_acc_std":
            crisp_acc_std,

        "acc_std_lower":
            acc_std_lower,

        "acc_std_upper":
            acc_std_upper,

        "leader_std":
            leader_std
    }


# ============================================================
# 27. 速度扰动沿车队传播
# ============================================================

def plot_stability_gain(
    metrics
):

    vehicle_numbers = np.arange(
        0,
        N_FOLLOWERS + 1
    )


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
        linewidth=2.0,
        label="确定性 IDM"
    )


    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="扰动不放大边界 G=1"
    )


    labels = [
        "前车"
    ] + [
        f"跟驰车{i}"
        for i in range(
            1,
            N_FOLLOWERS + 1
        )
    ]


    plt.xticks(
        vehicle_numbers,
        labels
    )


    plt.xlabel(
        "车辆序号"
    )

    plt.ylabel(
        "速度扰动放大系数 G"
    )

    plt.title(
        "周期性扰动场景：速度扰动沿车队传播"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        "periodic_stability_gain.png"
    )


# ============================================================
# 28. 跟驰行为波动性
# ============================================================

def plot_acceleration_fluctuation(
    metrics
):

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

        linewidth=2.0,

        label="确定性 IDM"
    )


    labels = [
        f"跟驰车{i}"
        for i in range(
            1,
            N_FOLLOWERS + 1
        )
    ]


    plt.xticks(
        vehicles,
        labels
    )


    plt.xlabel(
        "车辆序号"
    )

    plt.ylabel(
        "加速度标准差 / (m/s²)"
    )

    plt.title(
        "周期性扰动场景：跟驰行为波动性"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    finish_figure(
        "periodic_acceleration_fluctuation.png"
    )


# ============================================================
# 29. 输出周期稳定性指标
# ============================================================

def print_stability_metrics(
    metrics
):

    print(
        "\n========================================"
    )

    print(
        "周期扰动稳定阶段分析"
    )

    print(
        "========================================"
    )


    print(
        f"扰动周期："
        f"{PERIODIC_PERIOD:.2f} s"
    )

    print(
        f"预热时间："
        f"{PERIODIC_WARMUP:.2f} s"
    )

    print(
        f"稳定性统计起点："
        f"{PERIODIC_ANALYSIS_START:.2f} s"
    )

    print(
        f"统计完整周期数："
        f"{PERIODIC_ANALYSIS_CYCLES}"
    )


    print(
        "\n速度扰动放大系数 G："
    )


    for i in range(
        N_FOLLOWERS
    ):

        print(
            f"跟驰车{i+1}："
            f"Crisp={metrics['crisp_gain'][i]:.4f}，"
            f"Fuzzy α=0="
            f"[{metrics['gain_lower'][i]:.4f}, "
            f"{metrics['gain_upper'][i]:.4f}]"
        )


    print(
        "\n加速度标准差："
    )


    for i in range(
        N_FOLLOWERS
    ):

        print(
            f"跟驰车{i+1}："
            f"Crisp="
            f"{metrics['crisp_acc_std'][i]:.4f} m/s²，"
            f"Fuzzy α=0="
            f"[{metrics['acc_std_lower'][i]:.4f}, "
            f"{metrics['acc_std_upper'][i]:.4f}] m/s²"
        )


    fuzzy_max_G = np.max(
        metrics[
            "gain_upper"
        ]
    )


    print(
        "\n稳定性判断："
    )


    if fuzzy_max_G < 1.0:

        print(
            "当前扰动频率与模糊参数范围内，"
            "Fuzzy-IDM 的速度扰动放大系数上界均小于1，"
            "说明该数值实验中速度扰动沿车队总体呈衰减趋势。"
        )

    else:

        print(
            "至少部分模糊参数组合的 G 达到或超过1，"
            "说明当前条件下存在速度扰动不衰减或放大的可能。"
        )


# ============================================================
# 30. 保存周期稳定性指标 CSV
# ============================================================

def save_stability_metrics_csv(
    metrics
):

    if not SAVE_FIG:
        return


    path = os.path.join(
        OUTPUT_DIR,
        "periodic_stability_metrics.csv"
    )


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "车辆",
                "Crisp速度扰动放大系数G",
                "Fuzzy_G下界",
                "Fuzzy_G上界",
                "Crisp加速度标准差",
                "Fuzzy加速度标准差下界",
                "Fuzzy加速度标准差上界"
            ]
        )


        for i in range(
            N_FOLLOWERS
        ):

            writer.writerow(
                [
                    f"跟驰车{i+1}",

                    metrics[
                        "crisp_gain"
                    ][i],

                    metrics[
                        "gain_lower"
                    ][i],

                    metrics[
                        "gain_upper"
                    ][i],

                    metrics[
                        "crisp_acc_std"
                    ][i],

                    metrics[
                        "acc_std_lower"
                    ][i],

                    metrics[
                        "acc_std_upper"
                    ][i]
                ]
            )


    print(
        f"稳定性指标已保存：{path}"
    )


# ============================================================
# 31. GRID_N 收敛性分析
# ============================================================

def run_grid_convergence():
    """
    对减速停车场景进行：

        GRID_N = 3,5,7,9

    收敛性检查。

    比较 α=0 下最后一辆跟驰车：

        最大速度模糊宽度

    随参数网格密度的变化。
    """


    print(
        "\n\n========================================"
    )

    print(
        "开始 GRID_N 收敛性分析"
    )

    print(
        "========================================"
    )


    grid_values = [
        3,
        5,
        7,
        9
    ]


    t = np.arange(
        0.0,
        50.0 + DT,
        DT
    )


    leader_v, leader_x = (
        generate_leader_profile(
            "braking",
            t
        )
    )


    max_widths = []


    for grid_n in grid_values:

        print(
            f"\nGRID_N={grid_n}"
        )


        (
            a_values,
            b_values,
            T_values
        ) = create_parameter_grid(
            alpha=0.0,
            grid_n=grid_n
        )


        raw = simulate_platoon(

            t=t,

            leader_v=leader_v,

            leader_x=leader_x,

            a_values=a_values,

            b_values=b_values,

            T_values=T_values
        )


        vehicle_index = (
            N_FOLLOWERS
            -
            1
        )


        velocity_data = raw[
            "v"
        ][
            :,
            vehicle_index,
            :
        ]


        lower = np.min(
            velocity_data,
            axis=0
        )


        upper = np.max(
            velocity_data,
            axis=0
        )


        width_kmh = (
            upper
            -
            lower
        ) * 3.6


        max_width = np.max(
            width_kmh
        )


        max_widths.append(
            max_width
        )


        print(
            f"参数组合数量："
            f"{len(a_values)}"
        )

        print(
            f"最大速度模糊宽度："
            f"{max_width:.6f} km/h"
        )


    # --------------------------------------------------------
    # 打印相邻网格相对变化
    # --------------------------------------------------------

    print(
        "\n相邻网格结果变化："
    )


    for i in range(
        1,
        len(grid_values)
    ):

        previous = max_widths[
            i - 1
        ]

        current = max_widths[
            i
        ]


        relative_change = (
            abs(
                current
                -
                previous
            )
            /
            max(
                abs(
                    previous
                ),
                1e-12
            )
            *
            100.0
        )


        print(
            f"GRID_N "
            f"{grid_values[i-1]}"
            f" → "
            f"{grid_values[i]}："
            f"{relative_change:.4f}%"
        )


    # --------------------------------------------------------
    # 收敛图
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 5.2)
    )


    plt.plot(
        grid_values,
        max_widths,
        marker="o",
        linewidth=2.0
    )


    plt.xlabel(
        "每个模糊参数区间的采样点数 GRID_N"
    )

    plt.ylabel(
        "最大速度模糊区间宽度 / (km/h)"
    )

    plt.title(
        "减速停车场景：参数网格收敛性分析"
    )

    plt.xticks(
        grid_values
    )

    plt.grid(
        alpha=0.25
    )


    finish_figure(
        "grid_convergence.png"
    )


# ============================================================
# 32. 单场景运行
# ============================================================

def run_scenario(
    scenario
):

    scene_names = {

        "accelerating":
            "匀加速前车场景",

        "braking":
            "前车减速停车场景",

        "periodic":
            "周期性扰动场景"
    }


    scene_name = scene_names[
        scenario
    ]


    # --------------------------------------------------------
    # 仿真时间
    # --------------------------------------------------------

    if scenario == "periodic":

        t_end = PERIODIC_END

    else:

        t_end = 50.0


    t = np.arange(
        0.0,
        t_end + DT,
        DT
    )


    print(
        "\n\n========================================"
    )

    print(
        f"开始仿真：{scene_name}"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # 前车
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
            leader_x,
            grid_n=GRID_N
        )
    )


    # --------------------------------------------------------
    # 数值正确性检查
    # --------------------------------------------------------

    check_alpha_one(
        crisp,
        fuzzy
    )


    check_alpha_nesting(
        fuzzy,
        scene_name
    )


    check_gap_safety(
        crisp,
        fuzzy,
        scene_name
    )


    # --------------------------------------------------------
    # 原有可视化全部保留
    # --------------------------------------------------------

    if scenario == "periodic":

        velocity_vehicle_indices = [
            0,
            N_FOLLOWERS - 1
        ]

        plot_vehicle_indices = [
            0,
            N_FOLLOWERS
            -
            1
        ]

    else:

        velocity_vehicle_indices = [
            0,
            N_FOLLOWERS - 1
        ]

        plot_vehicle_indices = [
            0,
            N_FOLLOWERS - 1
        ]


    plot_velocity(
        t,
        leader_v,
        crisp,
        fuzzy,
        scene_name,
        scenario,
        vehicle_indices=velocity_vehicle_indices
    )


    plot_gap(
        t,
        crisp,
        fuzzy,
        scene_name,
        scenario,
        vehicle_indices=plot_vehicle_indices
    )


    plot_acceleration(
        t,
        crisp,
        fuzzy,
        scene_name,
        scenario,
        vehicle_indices=plot_vehicle_indices
    )


    # 周期场景只画稳定扰动阶段的状态分布，
    # 避免把60秒预热段大量重复状态点画进去。
    if scenario == "periodic":

        phase_start = (
            PERIODIC_WARMUP
        )

    else:

        phase_start = 0.0


    plot_phase_diagram(
        t,
        raw_results,
        crisp,
        scene_name,
        scenario,
        start_time=phase_start,
        vehicle_indices=plot_vehicle_indices
    )


    plot_uncertainty_width(
        t,
        fuzzy,
        scene_name,
        scenario,
        vehicle_indices=plot_vehicle_indices
    )


    # --------------------------------------------------------
    # 周期扰动额外分析
    # --------------------------------------------------------

    if scenario == "periodic":

        metrics = (
            calculate_stability_metrics(
                t,
                leader_v,
                crisp,
                raw_results[
                    0.0
                ]
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


        save_stability_metrics_csv(
            metrics
        )


    return {

        "t":
            t,

        "leader_v":
            leader_v,

        "leader_x":
            leader_x,

        "crisp":
            crisp,

        "fuzzy":
            fuzzy,

        "raw":
            raw_results
    }


# ============================================================
# 33. 主程序
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
        f"\n初始条件模式："
        f"{INITIALIZATION_MODE}"
    )


    print(
        "\n确定性 IDM 中心参数："
    )

    print(
        f"a_max = "
        f"{A_MAX_CRISP:.2f} m/s²"
    )

    print(
        f"a_comf = "
        f"{A_COMF_CRISP:.2f} m/s²"
    )

    print(
        f"T = "
        f"{T_CRISP:.2f} s"
    )


    print(
        "\n三参数三角模糊数："
    )

    print(
        f"a_max = "
        f"{A_MAX_TRI} m/s²"
    )

    print(
        f"a_comf = "
        f"{A_COMF_TRI} m/s²"
    )

    print(
        f"T = "
        f"{T_TRI} s"
    )


    print(
        f"\nGRID_N = "
        f"{GRID_N}"
    )


    print(
        f"每个 α 水平最多参数组合："
        f"{GRID_N ** 3}"
    )


    # ========================================================
    # 场景1：匀加速
    # ========================================================

    result_accelerating = (
        run_scenario(
            "accelerating"
        )
    )


    # ========================================================
    # 场景2：减速停车
    # ========================================================

    result_braking = (
        run_scenario(
            "braking"
        )
    )


    # ========================================================
    # 场景3：周期性扰动
    # ========================================================

    result_periodic = (
        run_scenario(
            "periodic"
        )
    )


    # ========================================================
    # 参数网格收敛性验证
    # ========================================================

    if RUN_GRID_CONVERGENCE:

        run_grid_convergence()


    print(
        "\n\n========================================"
    )

    print(
        "全部仿真完成"
    )

    print(
        "========================================"
    )


    if SAVE_FIG:

        print(
            f"\n结果保存目录："
            f"{OUTPUT_DIR}"
        )


    if SHOW_FIG:
        plt.show()
