# SignalAI — 测控通信信号智能分析原型

面向测控、通信信息化业务的 AI 应用原型，用两个可独立训练、评测的任务验证
"信号 + 时序数据 → 深度学习模型" 的完整链路：

| 任务 | 输入 | 输出 | 模型 |
| --- | --- | --- | --- |
| 通信信号调制识别 | 1024 点基带 IQ 序列 | 8 类调制方式 | 1D CNN |
| 遥测异常检测 | 6 通道遥测时序（64 点窗口） | 正常 / 异常及异常分数 | 1D 卷积自编码器 |

## 快速开始

```powershell
# 1) 创建环境（国内网络走清华镜像）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2) 一键跑通：数据生成 -> 训练 -> 评测
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

或者手动执行：

```powershell
conda activate signalai
python -m signalai gen-data
python -m signalai train-mod --epochs 15
python -m signalai train-ae  --epochs 30
python -m signalai eval
```

> Windows 下若直接调用 `envs\signalai\python.exe` 而不激活环境，torch 会因
> 动态库搜索路径缺失报 `DLL load failed`；请用 `conda run -n signalai ...`
> 或先 `conda activate signalai`。

## 目录结构

```text
SignalAI/
├─ src/signalai/
│  ├─ simulate.py    # IQ 调制信号与遥测时序仿真、异常注入
│  ├─ dataset.py     # 数据集持久化、窗口切分、训练/验证划分
│  ├─ models.py      # ModulationCNN / TelemetryAE
│  ├─ train_mod.py   # 调制识别训练
│  ├─ train_ae.py    # 遥测自编码器训练与阈值确定
│  ├─ evaluate.py    # 指标计算与图表
│  └─ cli.py         # 命令行入口
├─ scripts/          # 环境安装与一键运行脚本
├─ tests/            # 冒烟测试
├─ artifacts/        # 数据、模型权重、指标 JSON、图表
└─ docs/report.md    # 技术报告
```

## 数据仿真

**通信信号**（`simulate.py`）：

- 8 类：BPSK、QPSK、8PSK、16QAM、64QAM、GMSK、AM-DSB、FM
- 成形：根升余弦（β=0.35，8 倍过采样），GMSK 为高斯滤波 MSK（BT=0.3）
- 信道损伤：随机载波频偏（±0.4% 采样率）、随机相偏、符号定时偏移、
  Wiener 相位噪声、幅度起伏
- 噪声：AWGN，训练 SNR 在 [-5, 20] dB 内随机；评测覆盖 [-10, 20] dB 七个档位
- 每条样本归一化到单位功率

**遥测时序**：

- 6 通道：滚转/俯仰/偏航姿态角、母线电压、温度、下行电平
- 正常段：多周期正弦叠加通道相关噪声
- 异常类型：spike（尖峰）、drift（漂移）、stuck（卡死）、dropout（丢失）、
  step（阶跃），单通道注入，长度 60–200 点

## 方法

- **调制识别**：三层 Conv1d + BN + MaxPool，全局池化后接两层全连接；
  以 2×1024 原始 IQ 为输入，交叉熵训练，CosineAnnealing 学习率。
- **遥测异常检测**：只用正常数据训练 1D 卷积自编码器，重建 MSE 作为异常分数；
  阈值取验证集正常窗口分数的 99.5 分位。

## 实验结果

| 任务 | 指标 |
| --- | --- |
| 调制识别（8 类） | 总体 71.4%；SNR ≥ 0 dB 时 82.0%；20 dB 时 84.6% |
| 遥测异常检测（窗口级） | AUROC 0.818；F1 0.700 |
| 遥测异常检测（序列级） | AUROC 0.894；误报率 2.5% |
| 序列级检出率 | drift 100% / spike 100% / step 93% / dropout 75% / stuck 35% |

分 SNR 曲线、混淆矩阵与异常分数分布见 `artifacts/fig_*.png`；
完整分析与误差讨论见 [docs/report.md](docs/report.md)。

## 局限

- 数据为**仿真信号**，用于验证方法链路，不等同于真实射频链路或卫星遥测数据；
  真实场景需接入 RadioML 2016.10a 等公开数据或实测数据重新训练与标定。
- 调制识别对低 SNR（<= -5 dB）与同族调制（如 PSK 阶数）的区分能力有限，
  具体见指标表中的分 SNR 准确率。
- 遥测异常检测假设异常类型已在仿真中建模；对未建模故障模式的泛化能力未验证。

## 测试

```powershell
conda run -n signalai python -m pytest -q
```
