"""用统一的局部 Lyapunov 探针比较三种互斥模糊化实验。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STATE_SUMMARY = Path(r"D:\Desktop\state_fuzzy_2d_lyapunov_20260929\results\state_fuzzy_lyapunov_summary.json")
PARAMETER_SUMMARY = HERE / "results" / "parameter" / "parameter_fuzzy_stability_summary.json"
GEOMETRY_SUMMARY = HERE / "results" / "geometry" / "geometry_fuzzy_stability_summary.json"
E0 = np.array([0.20, 0.08, 0.01, 0.05, 0.02, 0.005], dtype=float)
DT = 0.01
DURATION = 8.0


def linear_probe(a: np.ndarray, p: np.ndarray) -> float:
    state = E0.copy()
    initial = float(state @ p @ state)
    for _ in range(round(DURATION / DT)):
        k1 = a @ state
        k2 = a @ (state + 0.5*DT*k1)
        k3 = a @ (state + 0.5*DT*k2)
        k4 = a @ (state + DT*k3)
        state = state + DT*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return float((state @ p @ state) / initial)


def main() -> None:
    state = json.loads(STATE_SUMMARY.read_text(encoding="utf-8"))
    parameter = json.loads(PARAMETER_SUMMARY.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY_SUMMARY.read_text(encoding="utf-8"))
    a = np.asarray(state["lyapunov"]["jacobian"], dtype=float)
    p = np.asarray(state["lyapunov"]["p_matrix"], dtype=float)
    state_ratio = linear_probe(a, p)
    rows = [
        {"experiment": "状态模糊化（仅初始六状态）",
         "worst_spectral_abscissa": max(item["real"] for item in state["lyapunov"]["jacobian_eigenvalues"]),
         "minimum_lambda_P": min(state["lyapunov"]["p_eigenvalues"]),
         "common_probe_final_ratio": state_ratio,
         "standardized_diagnostic_score": 80.0 + 20.0*(1.0-np.sqrt(state_ratio)),
         "all_hurwitz": state["lyapunov"]["hurwitz"]},
        {"experiment": "参数模糊化（a,b,T,ε,ζ,γ）",
         "worst_spectral_abscissa": parameter["stability"]["worst_spectral_abscissa"],
         "minimum_lambda_P": parameter["stability"]["minimum_p_eigenvalue"],
         "common_probe_final_ratio": parameter["stability"]["worst_probe_final_ratio"],
         "standardized_diagnostic_score": parameter["stability_score"]["score_out_of_100"],
         "all_hurwitz": parameter["stability"]["all_hurwitz"]},
        {"experiment": "环境几何模糊化（标线与中心线）",
         "worst_spectral_abscissa": geometry["stability"]["worst_spectral_abscissa"],
         "minimum_lambda_P": geometry["stability"]["minimum_p_eigenvalue"],
         "common_probe_final_ratio": geometry["stability"]["worst_probe_final_ratio"],
         "standardized_diagnostic_score": geometry["stability_score"]["score_out_of_100"],
         "all_hurwitz": geometry["stability"]["all_hurwitz"]},
    ]
    out = HERE / "results" / "three_experiment_stability_comparison.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                         "axes.unicode_minus":False, "font.size":10.5, "savefig.dpi":300})
    names = ["状态模糊", "参数模糊", "环境几何模糊"]
    spectral = [row["worst_spectral_abscissa"] for row in rows]
    ratios = [row["common_probe_final_ratio"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(names, spectral, color=["#4D8EB9","#C06442","#2C8C7B"])
    axes[0].axhline(0, color="#46515B", linewidth=1)
    axes[0].set_title("最不利谱横坐标（小于 0 表示 Hurwitz）", fontweight="bold")
    axes[0].set_ylabel("max Re(λ(A))")
    axes[1].bar(names, ratios, color=["#4D8EB9","#C06442","#2C8C7B"])
    axes[1].set_yscale("log")
    axes[1].set_title("统一局部探针 8 s 后的 V(8)/V(0)", fontweight="bold")
    axes[1].set_ylabel("越小表示该探针收敛越充分")
    for ax in axes:
        ax.grid(True, axis="y", color="#CBD2D9", alpha=.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("三类模糊化实验的局部稳定性对照", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(HERE / "results" / "13_three_experiment_stability_comparison.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
