import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 0. 中文字体设置
# ============================================================

def setup_chinese_font():
    """优先使用 Windows 等线，其次微软雅黑、黑体。"""

    font_paths = [
        r"C:\Windows\Fonts\Deng.ttf",       # 等线
        r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(
                    fname=path
                ).get_name()

                plt.rcParams["font.sans-serif"] = [
                    name,
                    "DejaVu Sans"
                ]

                plt.rcParams["axes.unicode_minus"] = False
                return

            except Exception:
                pass

    candidates = [
        "DengXian",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN"
    ]

    available = {
        f.name
        for f in font_manager.fontManager.ttflist
    }

    for name in candidates:
        if name in available:

            plt.rcParams["font.sans-serif"] = [
                name,
                "DejaVu Sans"
            ]

            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["axes.unicode_minus"] = False

    print(
        "警告：未检测到常用中文字体，"
        "若中文乱码请安装等线、微软雅黑或思源黑体。"
    )


setup_chinese_font()


# ============================================================
# 1. 全局设置
# ============================================================

DT = 0.05

# 每个 α 水平下：
# a、b、T 各取 5 个采样点
# 最多形成 5³ = 125 组参数
GRID_N = 5

# 前车 + 5 辆跟驰车
N_FOLLOWERS = 5

ALPHA_LEVELS = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
]


# ------------------------------------------------------------
# 预热时间
#
# 目的：
# 所有参数组合首先从相同的确定性状态出发，
# 在恒速前车下运行 120 s。
#
# 这段过程不绘图。
#
# 这样可以避免原代码中：
# t=0 时不同参数组合并不处于各自平衡状态，
# 导致加速度模糊带突然很宽的问题。
# ------------------------------------------------------------

WARMUP_TIME = 120.0


SAVE_FIG = True
SHOW_FIG = True

OUTPUT_DIR = "fuzzy_idm_key_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 单位转换
# ============================================================

def kmh_to_ms(v):
    return np.asarray(
        v,
        dtype=float
    ) / 3.6


def ms_to_kmh(v):
    return np.asarray(
        v,
        dtype=float
    ) * 3.6


# ============================================================
# 3. IDM 参数
# ============================================================

# 期望速度
V_DES_KMH = 108.0
V_DES = kmh_to_ms(
    V_DES_KMH
)

# 静止期望间距
S_JAM = 2.0

# 车辆长度
VEHICLE_LENGTH = 5.0

# ------------------------------------------------------------
# 加速度指数 β
#
# 标准 IDM 中通常取：
#
# β = 4
#
# 当前不进行模糊化。
# ------------------------------------------------------------

BETA = 4.0


# ============================================================
# 4. Crisp IDM 中心参数
# ============================================================

A_MAX_CRISP = 1.0      # m/s²
A_COMF_CRISP = 1.5     # m/s²
T_CRISP = 1.5          # s


# ============================================================
# 5. 三参数模糊化
# ============================================================
#
# 当前采用：
#
# 中心值 ±20%
#
# 作为代表性模糊范围。
#
# a_max：
# (0.8, 1.0, 1.2)
#
# a_comf：
# (1.2, 1.5, 1.8)
#
# T：
# (1.2, 1.5, 1.8)
#
# ============================================================

FUZZY_RELATIVE_WIDTH = 0.20


def make_triangle(
    center,
    relative_width=FUZZY_RELATIVE_WIDTH
):

    return (
        center
        *
        (
            1.0
            -
            relative_width
        ),

        center,

        center
        *
        (
            1.0
            +
            relative_width
        )
    )


A_MAX_TRI = make_triangle(
    A_MAX_CRISP
)

A_COMF_TRI = make_triangle(
    A_COMF_CRISP
)

T_TRI = make_triangle(
    T_CRISP
)


# ============================================================
# 6. 三角模糊数 α-截集
# ============================================================

def triangular_alpha_cut(
    triangle,
    alpha
):

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
# 7. 构造 α 水平下的参数组合
# ============================================================

def create_parameter_grid(
    alpha,
    grid_n=GRID_N
):
    """
    Ω_α
    =
    [a_max]_α
    ×
    [a_comf]_α
    ×
    [T]_α
    """

    sample_arrays = []

    for triangle in [
        A_MAX_TRI,
        A_COMF_TRI,
        T_TRI
    ]:

        lower, upper = (
            triangular_alpha_cut(
                triangle,
                alpha
            )
        )

        if np.isclose(
            lower,
            upper
        ):

            samples = np.array(
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

        else:

            samples = np.linspace(
                lower,
                upper,
                grid_n
            )

        sample_arrays.append(
            samples
        )


    a_grid, b_grid, T_grid = (
        np.meshgrid(
            sample_arrays[0],
            sample_arrays[1],
            sample_arrays[2],
            indexing="ij"
        )
    )


    return (
        a_grid.ravel(),
        b_grid.ravel(),
        T_grid.ravel()
    )


# ============================================================
# 8. IDM 加速度函数
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
    严格采用你的公式定义：

    ΔV_n
    =
    V_(n-1)
    -
    V_n

    S_des
    =
    S_jam
    +
    max[
        0,
        V_n*T
        -
        V_n*ΔV_n
        /
        (2√(a_max*a_comf))
    ]
    """

    # 防止除 0
    gap_safe = np.maximum(
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

            v_follower
            *
            delta_v

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
                gap_safe
            ) ** 2
        )
    )


    return acceleration


# ============================================================
# 9. Crisp 中心参数对应的平衡间距
# ============================================================

def equilibrium_gap(
    speed,
    T=T_CRISP
):
    """
    在：

    ΔV = 0
    a_n = 0

    条件下计算中心参数对应的初始净间距。
    """

    denominator = (

        1.0

        -

        (
            float(speed)
            /
            float(V_DES)
        ) ** BETA
    )


    denominator = max(
        denominator,
        1e-8
    )


    desired_gap = (
        S_JAM
        +
        float(speed)
        *
        T
    )


    return (
        desired_gap
        /
        math.sqrt(
            denominator
        )
    )


# ============================================================
# 10. 前车场景
# ============================================================

def build_leader_profile(
    scenario
):
    """
    所有场景：

    先预热 120 s，
    然后才正式开始交通扰动。

    图中的 t=0
    表示扰动开始，
    而不是程序真正的初始时刻。
    """


    # --------------------------------------------------------
    # 匀加速前车
    # --------------------------------------------------------

    if scenario == "accelerating":

        initial_speed = kmh_to_ms(
            54.0
        )

        event_duration = 60.0


    # --------------------------------------------------------
    # 减速停车
    # --------------------------------------------------------

    elif scenario == "braking":

        initial_speed = kmh_to_ms(
            72.0
        )

        event_duration = 60.0


    # --------------------------------------------------------
    # 周期扰动
    # --------------------------------------------------------

    elif scenario == "periodic":

        initial_speed = kmh_to_ms(
            72.0
        )

        omega = 0.2

        period = (
            2.0
            *
            np.pi
            /
            omega
        )

        # 6 个完整周期
        event_duration = (
            6.0
            *
            period
        )


    else:

        raise ValueError(
            f"未知场景：{scenario}"
        )


    # --------------------------------------------------------
    # 总时间
    # --------------------------------------------------------

    t = np.arange(

        0.0,

        WARMUP_TIME
        +
        event_duration
        +
        DT
        /
        2.0,

        DT
    )


    # 扰动开始之后的时间
    tau = np.maximum(
        t
        -
        WARMUP_TIME,
        0.0
    )


    # ========================================================
    # 场景1：匀加速
    # ========================================================

    if scenario == "accelerating":

        event_speed = np.minimum(

            initial_speed
            +
            0.5
            *
            tau,

            kmh_to_ms(
                72.0
            )
        )


    # ========================================================
    # 场景2：减速停车
    # ========================================================

    elif scenario == "braking":

        event_speed = np.maximum(

            initial_speed

            -

            1.5
            *
            tau,

            0.0
        )


    # ========================================================
    # 场景3：周期扰动
    # ========================================================

    else:

        event_speed = (

            kmh_to_ms(
                72.0
            )

            +

            kmh_to_ms(
                7.2
            )

            *
            np.sin(
                0.2
                *
                tau
            )
        )


    # --------------------------------------------------------
    # 预热过程中前车保持恒速
    # --------------------------------------------------------

    leader_v = np.where(

        t
        <
        WARMUP_TIME,

        initial_speed,

        event_speed
    )


    # --------------------------------------------------------
    # 积分得到前车位置
    # --------------------------------------------------------

    leader_x = np.zeros_like(
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

        np.diff(
            t
        )
    )


    event_mask = (
        t
        >=
        WARMUP_TIME
    )


    return {

        "scenario":
            scenario,

        "t_full":
            t,

        "event_mask":
            event_mask,

        "event_time":
            t[event_mask]
            -
            WARMUP_TIME,

        "leader_v":
            leader_v,

        "leader_x":
            leader_x
    }


# ============================================================
# 11. 多车辆跟驰仿真
# ============================================================

def simulate_platoon(
    profile,
    a_values,
    b_values,
    T_values
):
    """
    一个参数组合对应一支 5 辆跟驰车的同质车队。

    多个参数组合并行计算。
    """


    t = profile[
        "t_full"
    ]

    leader_v = profile[
        "leader_v"
    ]

    leader_x = profile[
        "leader_x"
    ]


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


    n_param = len(
        a_values
    )

    n_time = len(
        t
    )


    # --------------------------------------------------------
    # 所有参数组合从相同确定性状态出发
    # --------------------------------------------------------

    initial_speed = float(
        leader_v[0]
    )


    initial_gap = equilibrium_gap(
        initial_speed
    )


    v = np.full(

        (
            n_param,
            N_FOLLOWERS
        ),

        initial_speed,

        dtype=float
    )


    base_positions = np.array(

        [

            -(i + 1)

            *

            (
                VEHICLE_LENGTH
                +
                initial_gap
            )

            for i
            in range(
                N_FOLLOWERS
            )
        ],

        dtype=float
    )


    x = np.tile(

        base_positions,

        (
            n_param,
            1
        )
    )


    # --------------------------------------------------------
    # 保存轨迹
    # --------------------------------------------------------

    velocity_history = np.empty(

        (
            n_param,
            N_FOLLOWERS,
            n_time
        )
    )


    gap_history = np.empty_like(
        velocity_history
    )


    acceleration_history = np.empty_like(
        velocity_history
    )


    # ========================================================
    # 时间推进
    # ========================================================

    for k in range(
        n_time
    ):


        velocity_history[
            :,
            :,
            k
        ] = v


        current_gap = np.empty_like(
            v
        )


        current_acc = np.empty_like(
            v
        )


        # ----------------------------------------------------
        # 同一时刻先计算所有车辆加速度
        # ----------------------------------------------------

        for j in range(
            N_FOLLOWERS
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
                    v[
                        :,
                        j - 1
                    ]
                )

                x_leader_now = (
                    x[
                        :,
                        j - 1
                    ]
                )


            # 实际净间距
            gap = (

                x_leader_now

                -

                x[
                    :,
                    j
                ]

                -

                VEHICLE_LENGTH
            )


            current_gap[
                :,
                j
            ] = gap


            current_acc[
                :,
                j
            ] = idm_acceleration(

                v_follower=
                    v[
                        :,
                        j
                    ],

                v_leader=
                    v_leader_now,

                gap=
                    gap,

                a_max=
                    a_values,

                a_comf=
                    b_values,

                T=
                    T_values
            )


        gap_history[
            :,
            :,
            k
        ] = current_gap


        acceleration_history[
            :,
            :,
            k
        ] = current_acc


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
            DT
        )


        # ----------------------------------------------------
        # 梯形积分更新位置
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
            DT
        )


        v = v_new


    return {

        "v":
            velocity_history,

        "gap":
            gap_history,

        "acc":
            acceleration_history
    }


# ============================================================
# 12. Crisp IDM
# ============================================================

def simulate_crisp(
    profile
):

    return simulate_platoon(

        profile,

        np.array(
            [
                A_MAX_CRISP
            ]
        ),

        np.array(
            [
                A_COMF_CRISP
            ]
        ),

        np.array(
            [
                T_CRISP
            ]
        )
    )


# ============================================================
# 13. Fuzzy-IDM α-截集包络
# ============================================================

def extract_envelope(
    raw
):

    result = {}


    for key in [
        "v",
        "gap",
        "acc"
    ]:

        result[
            key
            +
            "_lower"
        ] = np.min(
            raw[key],
            axis=0
        )


        result[
            key
            +
            "_upper"
        ] = np.max(
            raw[key],
            axis=0
        )


    return result


def simulate_fuzzy(
    profile,
    keep_alpha0_raw=False
):

    fuzzy = {}

    alpha0_raw = None


    for alpha in ALPHA_LEVELS:

        print(
            f"    α={alpha:.2f}"
        )


        a_values, b_values, T_values = (
            create_parameter_grid(
                alpha
            )
        )


        raw = simulate_platoon(

            profile,

            a_values,

            b_values,

            T_values
        )


        fuzzy[
            alpha
        ] = extract_envelope(
            raw
        )


        # 周期场景需要保留 α=0 原始轨迹
        # 用于稳定性分析
        if (
            keep_alpha0_raw
            and
            np.isclose(
                alpha,
                0.0
            )
        ):

            alpha0_raw = raw


    return (
        fuzzy,
        alpha0_raw
    )


# ============================================================
# 14. 正确性检查
# ============================================================

def validate_result(
    profile,
    crisp,
    fuzzy
):

    mask = profile[
        "event_mask"
    ]


    # --------------------------------------------------------
    # α=1 应退化为 Crisp IDM
    # --------------------------------------------------------

    speed_error = np.max(

        np.abs(

            crisp[
                "v"
            ][0][
                :,
                mask
            ]

            -

            fuzzy[
                1.0
            ][
                "v_lower"
            ][
                :,
                mask
            ]
        )
    )


    # --------------------------------------------------------
    # α-截集嵌套性检查
    # --------------------------------------------------------

    nesting_ok = True


    for outer_alpha, inner_alpha in zip(

        ALPHA_LEVELS[:-1],

        ALPHA_LEVELS[1:]
    ):


        lo_outer = fuzzy[
            outer_alpha
        ][
            "v_lower"
        ][
            :,
            mask
        ]


        hi_outer = fuzzy[
            outer_alpha
        ][
            "v_upper"
        ][
            :,
            mask
        ]


        lo_inner = fuzzy[
            inner_alpha
        ][
            "v_lower"
        ][
            :,
            mask
        ]


        hi_inner = fuzzy[
            inner_alpha
        ][
            "v_upper"
        ][
            :,
            mask
        ]


        if (

            np.any(
                lo_inner
                <
                lo_outer
                -
                1e-8
            )

            or

            np.any(
                hi_inner
                >
                hi_outer
                +
                1e-8
            )
        ):

            nesting_ok = False
            break


    # --------------------------------------------------------
    # 最小安全间距
    # --------------------------------------------------------

    min_crisp_gap = np.min(

        crisp[
            "gap"
        ][0][
            :,
            mask
        ]
    )


    min_fuzzy_gap = np.min(

        fuzzy[
            0.0
        ][
            "gap_lower"
        ][
            :,
            mask
        ]
    )


    print(
        f"    α=1/Crisp 最大速度误差："
        f"{speed_error:.3e} m/s"
    )


    print(
        f"    α-截集嵌套："
        f"{'通过' if nesting_ok else '未通过'}"
    )


    print(
        f"    Crisp 最小净间距："
        f"{min_crisp_gap:.3f} m"
    )


    print(
        f"    Fuzzy α=0 最小净间距："
        f"{min_fuzzy_gap:.3f} m"
    )


    if min_fuzzy_gap <= 0:

        print(
            "    警告：出现非正净间距。"
        )


# ============================================================
# 15. 绘图公共设置
# ============================================================

SCENE_NAMES = {

    "accelerating":
        "匀加速前车场景",

    "braking":
        "前车减速停车场景",

    "periodic":
        "周期性扰动场景"
}


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
            f"    已保存：{path}"
        )


    if not SHOW_FIG:

        plt.close()


# ============================================================
# 16. 图1～3：三个场景速度演化
# ============================================================

def plot_speed(
    profile,
    crisp,
    fuzzy
):

    scenario = profile[
        "scenario"
    ]


    mask = profile[
        "event_mask"
    ]


    t = profile[
        "event_time"
    ]


    # 最后一辆跟驰车
    j = (
        N_FOLLOWERS
        -
        1
    )


    plt.figure(
        figsize=(
            9,
            5.4
        )
    )


    # α=0
    plt.fill_between(

        t,

        ms_to_kmh(
            fuzzy[
                0.0
            ][
                "v_lower"
            ][
                j,
                mask
            ]
        ),

        ms_to_kmh(
            fuzzy[
                0.0
            ][
                "v_upper"
            ][
                j,
                mask
            ]
        ),

        alpha=0.18,

        label=
            "Fuzzy-IDM：α=0"
    )


    # α=0.5
    plt.fill_between(

        t,

        ms_to_kmh(
            fuzzy[
                0.5
            ][
                "v_lower"
            ][
                j,
                mask
            ]
        ),

        ms_to_kmh(
            fuzzy[
                0.5
            ][
                "v_upper"
            ][
                j,
                mask
            ]
        ),

        alpha=0.30,

        label=
            "Fuzzy-IDM：α=0.5"
    )


    # 前车
    plt.plot(

        t,

        ms_to_kmh(
            profile[
                "leader_v"
            ][
                mask
            ]
        ),

        "--",

        linewidth=1.8,

        label="前车"
    )


    # Crisp
    plt.plot(

        t,

        ms_to_kmh(
            crisp[
                "v"
            ][
                0,
                j,
                mask
            ]
        ),

        linewidth=2.2,

        label="确定性 IDM"
    )


    plt.xlabel(
        "扰动开始后的时间 / s"
    )

    plt.ylabel(
        "速度 / (km/h)"
    )

    plt.title(
        f"{SCENE_NAMES[scenario]}：速度演化"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()


    finish_figure(
        f"{scenario}_speed.png"
    )


# ============================================================
# 17. 图4：停车场景实际间距
# ============================================================

def plot_braking_gap(
    profile,
    crisp,
    fuzzy
):

    mask = profile[
        "event_mask"
    ]

    t = profile[
        "event_time"
    ]

    j = (
        N_FOLLOWERS
        -
        1
    )


    plt.figure(
        figsize=(
            9,
            5.4
        )
    )


    plt.fill_between(

        t,

        fuzzy[
            0.0
        ][
            "gap_lower"
        ][
            j,
            mask
        ],

        fuzzy[
            0.0
        ][
            "gap_upper"
        ][
            j,
            mask
        ],

        alpha=0.18,

        label=
            "Fuzzy-IDM：α=0"
    )


    plt.fill_between(

        t,

        fuzzy[
            0.5
        ][
            "gap_lower"
        ][
            j,
            mask
        ],

        fuzzy[
            0.5
        ][
            "gap_upper"
        ][
            j,
            mask
        ],

        alpha=0.30,

        label=
            "Fuzzy-IDM：α=0.5"
    )


    plt.plot(

        t,

        crisp[
            "gap"
        ][
            0,
            j,
            mask
        ],

        linewidth=2.2,

        label=
            "确定性 IDM"
    )


    plt.axhline(

        S_JAM,

        linestyle="--",

        linewidth=1.4,

        label=
            f"静止期望间距 {S_JAM:.1f} m"
    )


    plt.xlabel(
        "扰动开始后的时间 / s"
    )

    plt.ylabel(
        "实际净间距 / m"
    )

    plt.title(
        "前车减速停车场景：车辆间距演化"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()


    finish_figure(
        "braking_gap.png"
    )


# ============================================================
# 18. 周期扰动稳定性指标
# ============================================================

def periodic_metrics(
    profile,
    crisp,
    alpha0_raw
):
    """
    周期：

        2π / 0.2 ≈ 31.4 s

    总计 6 个周期。

    前 2 个周期作为瞬态，
    后 4 个周期用于统计。

    稳定性指标：

        G_i
        =
        σ(V_i)
        /
        σ(V_leader)

    G < 1：
        当前正弦扰动逐渐衰减。

    注意：
        这是数值稳定性指标，
        不是严格解析 string stability 证明。
    """


    period = (
        2.0
        *
        np.pi
        /
        0.2
    )


    event_tau = (

        profile[
            "t_full"
        ]

        -

        WARMUP_TIME
    )


    analysis_mask = (

        event_tau
        >=
        2.0
        *
        period
    )


    # --------------------------------------------------------
    # 前车速度标准差
    # --------------------------------------------------------

    leader_std = np.std(

        profile[
            "leader_v"
        ][
            analysis_mask
        ]
    )


    # --------------------------------------------------------
    # Crisp IDM
    # --------------------------------------------------------

    crisp_v = (

        crisp[
            "v"
        ][0][
            :,
            analysis_mask
        ]
    )


    crisp_gain = (

        np.std(
            crisp_v,
            axis=1
        )

        /

        leader_std
    )


    # --------------------------------------------------------
    # Fuzzy-IDM α=0
    # --------------------------------------------------------

    fuzzy_v = (

        alpha0_raw[
            "v"
        ][
            :,
            :,
            analysis_mask
        ]
    )


    fuzzy_gain_each = (

        np.std(
            fuzzy_v,
            axis=2
        )

        /

        leader_std
    )


    gain_lower = np.min(
        fuzzy_gain_each,
        axis=0
    )


    gain_upper = np.max(
        fuzzy_gain_each,
        axis=0
    )


    # --------------------------------------------------------
    # 加速度标准差
    # --------------------------------------------------------

    crisp_acc = (

        crisp[
            "acc"
        ][0][
            :,
            analysis_mask
        ]
    )


    crisp_acc_std = np.std(
        crisp_acc,
        axis=1
    )


    fuzzy_acc = (

        alpha0_raw[
            "acc"
        ][
            :,
            :,
            analysis_mask
        ]
    )


    fuzzy_acc_std_each = np.std(
        fuzzy_acc,
        axis=2
    )


    acc_std_lower = np.min(
        fuzzy_acc_std_each,
        axis=0
    )


    acc_std_upper = np.max(
        fuzzy_acc_std_each,
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
            acc_std_upper
    }


# ============================================================
# 19. 图5：速度扰动沿车队传播
# ============================================================

def plot_stability_gain(
    metrics
):

    x = np.arange(
        N_FOLLOWERS
        +
        1
    )


    labels = [

        "前车",
        "跟驰车1",
        "跟驰车2",
        "跟驰车3",
        "跟驰车4",
        "跟驰车5"
    ]


    crisp_gain = np.r_[

        1.0,

        metrics[
            "crisp_gain"
        ]
    ]


    fuzzy_lower = np.r_[

        1.0,

        metrics[
            "gain_lower"
        ]
    ]


    fuzzy_upper = np.r_[

        1.0,

        metrics[
            "gain_upper"
        ]
    ]


    plt.figure(
        figsize=(
            8.7,
            5.4
        )
    )


    plt.fill_between(

        x,

        fuzzy_lower,

        fuzzy_upper,

        alpha=0.22,

        label=
            "Fuzzy-IDM：α=0 范围"
    )


    plt.plot(

        x,

        crisp_gain,

        marker="o",

        linewidth=2.1,

        label=
            "确定性 IDM"
    )


    plt.axhline(

        1.0,

        linestyle="--",

        linewidth=1.4,

        label=
            "扰动不放大参考线 G=1"
    )


    plt.xticks(
        x,
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
# 20. 图6：跟驰行为波动性
# ============================================================

def plot_acceleration_fluctuation(
    metrics
):

    x = np.arange(
        1,
        N_FOLLOWERS
        +
        1
    )


    labels = [

        "跟驰车1",
        "跟驰车2",
        "跟驰车3",
        "跟驰车4",
        "跟驰车5"
    ]


    plt.figure(
        figsize=(
            8.7,
            5.4
        )
    )


    plt.fill_between(

        x,

        metrics[
            "acc_std_lower"
        ],

        metrics[
            "acc_std_upper"
        ],

        alpha=0.22,

        label=
            "Fuzzy-IDM：α=0 范围"
    )


    plt.plot(

        x,

        metrics[
            "crisp_acc_std"
        ],

        marker="o",

        linewidth=2.1,

        label=
            "确定性 IDM"
    )


    plt.xticks(
        x,
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
# 21. 图7：三种场景最大速度不确定性
# ============================================================

def plot_peak_uncertainty(
    all_results
):

    scene_order = [

        "accelerating",
        "braking",
        "periodic"
    ]


    labels = [

        "匀加速",
        "减速停车",
        "周期扰动"
    ]


    peak_widths = []

    peak_times = []


    for scenario in scene_order:


        profile = all_results[
            scenario
        ][
            "profile"
        ]


        fuzzy = all_results[
            scenario
        ][
            "fuzzy"
        ]


        mask = profile[
            "event_mask"
        ]


        t = profile[
            "event_time"
        ]


        j = (
            N_FOLLOWERS
            -
            1
        )


        width = ms_to_kmh(

            fuzzy[
                0.0
            ][
                "v_upper"
            ][
                j,
                mask
            ]

            -

            fuzzy[
                0.0
            ][
                "v_lower"
            ][
                j,
                mask
            ]
        )


        idx = int(
            np.argmax(
                width
            )
        )


        peak_widths.append(
            float(
                width[idx]
            )
        )


        peak_times.append(
            float(
                t[idx]
            )
        )


    x = np.arange(
        len(
            labels
        )
    )


    plt.figure(
        figsize=(
            7.5,
            5.2
        )
    )


    bars = plt.bar(
        x,
        peak_widths
    )


    plt.xticks(
        x,
        labels
    )


    plt.ylabel(
        "α=0 最大速度模糊宽度 / (km/h)"
    )


    plt.title(
        "不同交通场景下的最大速度不确定性"
    )


    plt.grid(
        axis="y",
        alpha=0.25
    )


    for bar, width, time_value in zip(

        bars,

        peak_widths,

        peak_times
    ):

        plt.text(

            bar.get_x()
            +
            bar.get_width()
            /
            2.0,

            bar.get_height(),

            f"{width:.2f}\n"
            f"(t={time_value:.1f}s)",

            ha="center",

            va="bottom"
        )


    finish_figure(
        "scenario_peak_uncertainty.png"
    )


# ============================================================
# 22. 可选：GRID_N 收敛性检查
# ============================================================
#
# 正式写报告之前建议运行一次。
#
# 默认关闭，
# 因为 GRID_N=7 会明显增加计算量。
# ============================================================

RUN_GRID_CONVERGENCE = False


def grid_convergence_check():

    profile = build_leader_profile(
        "braking"
    )


    mask = profile[
        "event_mask"
    ]


    j = (
        N_FOLLOWERS
        -
        1
    )


    print(
        "\nGRID_N 收敛性检查："
    )


    for n in [
        3,
        5,
        7
    ]:


        a_values, b_values, T_values = (
            create_parameter_grid(
                0.0,
                grid_n=n
            )
        )


        raw = simulate_platoon(

            profile,

            a_values,

            b_values,

            T_values
        )


        width = ms_to_kmh(

            np.max(
                raw[
                    "v"
                ][
                    :,
                    j,
                    mask
                ],
                axis=0
            )

            -

            np.min(
                raw[
                    "v"
                ][
                    :,
                    j,
                    mask
                ],
                axis=0
            )
        )


        print(

            f"    GRID_N={n}，"

            f"参数组合={n**3}，"

            f"最大速度模糊宽度="
            f"{np.max(width):.4f} km/h"
        )


# ============================================================
# 23. 主程序
# ============================================================

if __name__ == "__main__":


    print(
        "============================================"
    )

    print(
        "Crisp IDM 与三参数 Fuzzy-IDM 仿真"
    )

    print(
        "============================================"
    )


    print(
        f"β = {BETA}"
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

    print(
        f"GRID_N = {GRID_N}"
    )

    print(
        f"预热时间 = {WARMUP_TIME:.0f} s"
    )


    all_results = {}


    # ========================================================
    # 三个交通场景
    # ========================================================

    for scenario in [

        "accelerating",
        "braking",
        "periodic"
    ]:


        print(
            f"\n--- {SCENE_NAMES[scenario]} ---"
        )


        profile = build_leader_profile(
            scenario
        )


        print(
            "  运行 Crisp IDM..."
        )


        crisp = simulate_crisp(
            profile
        )


        print(
            "  运行三参数 Fuzzy-IDM..."
        )


        fuzzy, alpha0_raw = simulate_fuzzy(

            profile,

            keep_alpha0_raw=(
                scenario
                ==
                "periodic"
            )
        )


        # ----------------------------------------------------
        # 正确性检查
        # ----------------------------------------------------

        validate_result(

            profile,

            crisp,

            fuzzy
        )


        all_results[
            scenario
        ] = {

            "profile":
                profile,

            "crisp":
                crisp,

            "fuzzy":
                fuzzy,

            "alpha0_raw":
                alpha0_raw
        }


        # ----------------------------------------------------
        # 三个场景各一张速度图
        # ----------------------------------------------------

        plot_speed(

            profile,

            crisp,

            fuzzy
        )


        # ----------------------------------------------------
        # 停车场景额外保留间距图
        # ----------------------------------------------------

        if scenario == "braking":

            plot_braking_gap(

                profile,

                crisp,

                fuzzy
            )


    # ========================================================
    # 周期扰动稳定性分析
    # ========================================================

    periodic = all_results[
        "periodic"
    ]


    metrics = periodic_metrics(

        periodic[
            "profile"
        ],

        periodic[
            "crisp"
        ],

        periodic[
            "alpha0_raw"
        ]
    )


    # 图5
    plot_stability_gain(
        metrics
    )


    # 图6
    plot_acceleration_fluctuation(
        metrics
    )


    # 图7
    plot_peak_uncertainty(
        all_results
    )


    # ========================================================
    # 输出主要数值指标
    # ========================================================

    print(
        "\n周期扰动核心指标："
    )


    for i in range(
        N_FOLLOWERS
    ):


        print(

            f"  跟驰车{i+1}："

            f"G_crisp="
            f"{metrics['crisp_gain'][i]:.4f}，"

            f"G_fuzzy∈["
            f"{metrics['gain_lower'][i]:.4f}, "
            f"{metrics['gain_upper'][i]:.4f}]，"

            f"σa_crisp="
            f"{metrics['crisp_acc_std'][i]:.4f} m/s²"
        )


    # ========================================================
    # 可选网格收敛检查
    # ========================================================

    if RUN_GRID_CONVERGENCE:

        grid_convergence_check()


    print(
        f"\n共保留 7 张核心图，"
        f"保存目录：{OUTPUT_DIR}"
    )


    if SHOW_FIG:

        plt.show()