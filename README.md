# fuzzy-microscopic-traffic-modeling

本仓库用于长期整理硕士科研学习项目：**模糊微分方程在微观交通流建模中的应用**。

项目当前以模糊微分方程（Fuzzy Differential Equations, FDE）理论学习和 Fuzzy-IDM 微观交通流模型仿真为起点，后续逐步扩展到 Fuzzy-FVDM / OVM 等新的模糊微观交通模型，以及多模态感知不确定性与 Multimodal Fuzzy Neural ODE 的框架构思。

这个仓库不仅保存代码，也用于持续记录阶段报告、仿真结果、阅读笔记、每日科研日志和组会汇报材料。希望 Git 提交历史本身能够形成一条清晰的科研工作时间线。

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

## 当前进度

- [x] 完成模糊微分方程基础概念学习
- [x] 完成简单 FDE 算例与 Euler / RK4 数值求解对比
- [x] 完成经典 IDM 模型公式与参数含义整理
- [x] 完成三参数 Fuzzy-IDM 初步建模
- [x] 完成匀加速、减速停车、周期扰动三类仿真
- [x] 完成第一次组会阶段汇报 PPT
- [x] 完成阶段三 Multimodal Fuzzy Neural ODE 构思初稿
- [ ] 系统阅读 FVDM / OVM 等候选微观交通模型
- [ ] 实现 Crisp FVDM / OVM 仿真
- [ ] 设计 Fuzzy-FVDM / Fuzzy-OVM 参数模糊化方案
- [ ] 整理多模态感知不确定性与 Neural ODE 相关文献
- [ ] 设计 Multimodal Fuzzy Neural ODE 的简化实验方案

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

## 每日工作流

每天开始：

```bash
git pull
```

工作完成：

```bash
git status
git add .
git commit -m "YYYY-MM-DD: 今日核心工作"
git push
```

提交前先更新当天的 `worklog/YYYY-MM-DD.md`，让代码、文档和日志保持同步。

## Git 提交规范

提交信息建议使用：

```text
YYYY-MM-DD: 核心工作内容
```

示例：

```text
2026-09-15: review microscopic traffic models for fuzzy extension
2026-09-16: implement crisp FVDM simulation
2026-09-17: add fuzzy parameters and alpha-cut solver
2026-09-18: add braking and periodic disturbance experiments
```

避免使用：

```text
update
修改
final
final2
最新版
test
```

如果一次工作包含多个独立内容，建议拆成多个 commit。例如文献整理、代码实现、PPT 修改可以分开提交。

## 安全与数据管理

- 不提交密码、token、API key、私钥和 `.env` 文件。
- 不提交 Python 虚拟环境、临时 Office 文件和系统缓存。
- 大型原始数据集放在 `data/raw/` 或 `data/datasets/`，默认由 `.gitignore` 忽略。
- `*.pptx`、`*.docx`、`*.pdf` 建议通过 Git LFS 管理，避免仓库体积过快膨胀。

