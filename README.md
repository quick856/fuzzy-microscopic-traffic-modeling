# fuzzy-microscopic-traffic-modeling

本仓库用于长期整理硕士科研学习项目：**模糊微分方程在微观交通流建模中的应用**。

项目当前以模糊微分方程（Fuzzy Differential Equations, FDE）理论学习和 Fuzzy-IDM 微观交通流模型仿真为起点，后续逐步扩展到 Fuzzy-FVDM / OVM 等新的模糊微观交通模型，以及多模态感知不确定性与 Multimodal Fuzzy Neural ODE 的框架构思。

这个仓库不仅保存代码，也用于持续记录阶段报告、仿真结果、阅读笔记、每日科研日志和组会汇报材料。

## 当前研究路线

### Stage 1: Fuzzy Differential Equations

- 模糊数与 α-截集
- H差、gH差、H导数、gH导数
- 模糊初值问题
- Euler / RK4 数值求解

### Stage 2: Fuzzy-IDM

- 经典 IDM
- `a_max`、`a_comf`、`T` 三参数模糊化
- α-截集数值求解
- 匀加速场景
- 减速停车场景
- 周期扰动场景
- 交通流稳定性与跟驰行为波动性分析

### Stage 3: 新的模糊微观交通模型

- OVM / FVDM 等候选模型
- Crisp 模型实现
- Fuzzy 模型构造
- 数值仿真与对比

### Stage 4: Multimodal Fuzzy Neural ODE

- 相机 / LiDAR 多模态感知
- 感知不确定性
- FNN 模糊嵌入
- Fuzzy Neural ODE 连续动力学
- 数值求解和端到端训练


## 仓库目录说明

```text
fuzzy-microscopic-traffic-modeling/
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ docs/
│  ├─ reports/          # 阶段报告、文档初稿、组会文字材料
│  ├─ notes/            # 学习笔记、公式推导、临时整理文本
│  ├─ literature/       # 文献清单、PDF、阅读笔记
│  └─ presentations/    # 组会 PPT、模板和展示材料
├─ src/
│  ├─ fde/              # 模糊微分方程基础代码
│  ├─ fuzzy_idm/        # Fuzzy-IDM 相关代码
│  ├─ fuzzy_fvdm/       # 后续 Fuzzy-FVDM / OVM 代码
│  └─ neural_ode/       # 后续 Neural ODE 相关代码
├─ experiments/
│  ├─ fuzzy_idm/        # Fuzzy-IDM 实验指标、CSV、运行配置
│  ├─ fuzzy_fvdm/       # 后续 FVDM/OVM 实验
│  └─ logs/             # 实验日志和运行记录
├─ figures/
│  ├─ fde/              # FDE 算例图
│  ├─ fuzzy_idm/        # Fuzzy-IDM 仿真图
│  ├─ fuzzy_fvdm/       # 后续模型图
│  └─ neural_ode/       # 神经微分方程框架图
├─ worklog/             # 每日科研日志
├─ configs/             # 参数配置文件
└─ data/                # 数据说明与占位文件；大型原始数据不提交
```
