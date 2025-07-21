---
title: dmp算法
date: 2025-03-20
lastMod: 2025-07-21T12:34:43.263Z
summary: 介绍一下dmp算法
category: MP类算法
tags: [技能学习, MP类算法, 模仿学习]
---

# 一、DMP 的基本概念

在机器人的规划控制中，我们需要事先规划参考轨迹，例如关节角度曲线、机械臂末端轨迹等，如果任务参数比较复杂多变，那么通过编程来规划参考轨迹就比较复杂了，而`示教`是一种比较简单直观的方法，我们可以让有经验的人带着机器人先完成一次任务，`然后让机器人自动学习其中的过程，从而省去编程的复杂过程。`我们希望能有一种方法，能够使用<font color="#ff0000">少量的参数来建模示教的轨迹，通过这些参数能够快速地复现示教轨迹，</font>同时，还希望在<font color="#ff0000">复现示教轨迹的时候能够增加一些任务参数来泛化和改变原始轨迹</font>，例如改变关节角度曲线的幅值和频率，改变机械臂末端轨迹的起始和目标位置等等，这样才能更加灵活的使用示教方法来完成实际任务。

论文作者最基本的出发点是想利用一个具有自稳定性的二阶动态系统来构造一个“吸引点”模型（attractor model），使得可以通过改变这个“吸引点”来改变系统的最终状态，从而达到修改轨迹目标位置的目的，而最简单最常用的二阶系统就是弹簧阻尼系统（PD 控制器）：
$${\ddot{y}}=\alpha_{y}(\beta_{y}(g-y)-{\dot{y}})$$
其中，y 表示系统状态（比如关节角度、速度、加速度等等），g 表示目标状态，最后系统会收敛到这个状态上。$\displaystyle \alpha _{y}和\beta _{y}$ 是两个常数，相当于 PD 控制器中的 P 参数和 D 参数。

使用弹簧阻尼系统虽然能够让系统收敛到目标状态 𝑔，却`无法控制收敛的过程量`，比如轨迹的形状。那么作者就想在这个 PD 控制器上`叠加一个非线性项来控制收敛过程量`，因此，在直观上，我们可以把<font color="#ff0000"> DMP 看作是一个 PD 控制器与一个轨迹形状学习器的叠加</font>：
$$\ddot{y}=\alpha_{y}(\beta_{y}(g-y)-\dot{y})+f$$
其中，上式右边的第一项就是PD控制器，而第二项 $f$ 就是轨迹形状学习器，是一个非线性函数。这样，我们就可以通过改变目标状态 𝑔和非线性项 𝑓来调整我们的轨迹终点和轨迹形状了，这可以称之为空间上的改变。但还不够，我们还需要改变轨迹的速度，从而获得不同收敛速度的轨迹，这个可以通过在速度曲线 $\displaystyle \dot{y}$ 上增加一个放缩项 𝜏来实现：
$$\tau^{2}\ddot{y}=\alpha_{y}\bigl(\beta_{y}\bigl(g-y\bigr)-\tau\dot{y}\bigr)+f$$
直观上来看，我们可以知道，上面这个公式表明在这个动态的控制过程中，系统最后会收敛到状态 g ，并且非线性项 f 的参与会直接影响 $\displaystyle \dot{y}$ ，也就是直接影响到收敛的过程。通过给定不同的 τ 和 g ，我们就可以在示教曲线的基础上得到不同目标状态以及不同速度的轨迹了。

那么，这个非线性项 𝑓具体怎么构造呢？要知道，我们叠加 𝑓的目的`就是为了改变轨迹的形状为我们期望的形状`，那么，<font color="#ff0000">什么样的非线性函数可以很容易的拟合各种形状的轨迹，而且方便我们进行参数的调整呢？</font>原大佬作者是这样考虑的，`通过多个非线性基函数的归一化线性叠加来实现`，因此，非线性函数 𝑓可以表示为：
$$f(t)={\frac{\sum_{i=1}^{N}\Psi_{i}(t)w_{i}}{\sum_{i=1}^{N}\Psi_{i}(t)}}$$
其中，$\displaystyle \Uppsi_{i}$ 是基函数，其实，在后面我们可以看到，这个基函数所使用的就是高斯基函数（径向基函数）， $\displaystyle w_{i}$ 为每个基函数对应的权重， 𝑁为基函数的个数。`我们通过给定不同的基函数和对应权重就可加权得到复杂的轨迹了`。

<font color=" #ff0000 ">此时，非线性项 𝑓 与时间 𝑡是高度相关的，无法直接叠加其他的动态系统进来，也无法同时建模多个自由度的轨迹并让它们在时间上与控制系统保持同步。</font>

因此，经典 DMP 方法引入了`规范系统`这个概念。

# 二、DMP 方法中的规范系统

---

**什么是规范系统？**

规范系统（Canonical System）是动态运动原语（DMP）中的一个核心组件，它是一个独立的动态系统，用于生成一个抽象的时间变量（通常是相位变量 $s$ 或 $\phi$），以驱动运动的进展。它实现了 DMP 的时间解耦，是与变换系统配合的关键部分。

- 术语来源：
  - “Canonical” 在数学和物理中意为“标准”或“典型”，这里指一个标准化的时间演化机制。
  - 在 DMP 的原始文献中（如 Schaal 和 Ijspeert 的论文），始终使用“Canonical System”。

**规范系统的数学形式**

DMP 的规范系统根据运动类型分为两种形式：

离散型 DMP 的规范系统：

- 形式：  
  $\tau \dot{s} = -\alpha_s s$

  - $s$：相位变量，初始值 $s(0) = 1$。
  - $\tau$：时间尺度，控制运动时长。
  - $\alpha_s$：正的衰减率（通常设为 1）。

- 解：  
  $s(t) = e^{-\alpha_s t / \tau}$
  - $s$ 从 1 单调衰减到接近 0。

连续型 DMP 的规范系统：

- 形式：  
  $\tau \dot{\phi} = 1$

  - $\phi$：相位变量，初始值 $\phi(0) = \phi_0$。
  - $\tau$：周期的时间尺度。

- 解：  
  $\phi(t) = \frac{t}{\tau} + \phi_0$
  - $\phi$ 线性增加，模 $2\pi$ 处理以保持周期性。

扩展形式：  
在改进版本中（如 Ijspeert et al., 2013），可使用振荡器：  
$\dot{\phi} = \omega + \text{coupling terms}$  
支持自适应频率的节奏运动。

**规范系统在 DMP 中的作用**

实现时间解耦 (Temporal Decoupling)：

- 作用：
  - 生成相位变量 $s$ 或 $\phi$，将运动进展与物理时间 $t$ 分离。
  - 变换系统中的强迫项 $f(s)$ 或 $f(\phi)$ 依赖相位变量，而非 $t$。
- 意义：
  - 时长无关性：调整 $\tau$ 可改变速度，轨迹形状不变。
  - 鲁棒性：运动暂停时，$s$ 或 $\phi$ 保持值，恢复后继续演化。

驱动运动进程 (Driving the Motion)：

- 作用：
  - 离散型：$s$ 的衰减控制从开始到结束的进度。
  - 连续型：$\phi$ 的增加驱动周期性重复。
- 意义：
  - 为强迫项提供时间基准，决定扰动的时机。

确保轨迹平滑性和一致性：

- 作用：
  - 规范系统保证相位变量的连续和平滑演化。
- 意义：
  - 避免时间跳跃，确保轨迹稳定。

支持学习与泛化：

- 作用：
  - $s$ 或 $\phi$ 标准化（$s \in [0, 1]$ 或 $\phi \in [0, 2\pi]$），便于定义强迫项基函数。
  - 学习到的 $f(s)$ 或 $f(\phi)$ 与时间无关，可复用。
- 意义：
  - 简化从示范中提取轨迹形状，提高可重用性。

统一离散与连续运动：

- 作用：
  - 通过调整规范系统形式，在离散型和连续型间切换。
- 意义：
  - 提供统一框架，适应多种任务。

**在 DMP 中的工作机制**

离散型示例：

1. 初始化：$s = 1$。
2. 演化：$s = e^{-t/\tau}$。
3. 驱动强迫项：$f(s)$ 根据 $s$ 施加扰动。
4. 结束：$s \to 0$，$f(s) \to 0$，收敛到 $g$。

连续型示例：

1. 初始化：$\phi = \phi_0$。
2. 演化：$\phi = t/\tau + \phi_0$。
3. 驱动强迫项：$f(\phi)$ 周期性变化。
4. 持续运行：运动围绕 $y^*$ 循环。

**与变换系统的关系**

- 变换系统：  
  $\tau \dot{v} = \alpha_z (\beta_z (g - y) - v) + f(s)$
  - 生成轨迹（$y$ 和 $v$）。
- 规范系统：
  - 提供 $s$ 或 $\phi$，驱动 $f(s)$ 或 $f(\phi)$。
- 耦合：通过 $\tau$ 和强迫项耦合，规范系统决定“何时”，变换系统决定“如何”。

**为什么需要规范系统？**

- 没有规范系统：
  - $f(t)$ 依赖 $t$，轨迹与时间绑定，难以调整或应对扰动。
- 有规范系统：
  - 解耦时间与轨迹，增强灵活性和鲁棒性。

**总结**

- 含义：生成相位变量的动态系统，标准化运动进程。
- 正确术语：规范系统（Canonical System）。
- 作用：时间解耦、驱动进程、确保平滑、支持学习、统一运动类型。

---

# 三、离散型 DMP 和节律性 DMP

**离散型 DMP 和节律性 DMP 的区别（表格表示）**

离散型 DMP（Discrete DMP）和节律性 DMP（Rhythmic DMP）是动态运动原语的两种形式，分别用于单次运动和周期性运动。以下表格对比两者的区别：

| **特性**     | **离散型 DMP**             | **节律性 DMP**                            |
| ------------ | -------------------------- | ----------------------------------------- |
| **运动类型** | 单次、点到点运动           | 周期性、节奏性运动                        |
| **规范系统** | $\tau\dot{s}=-\alpha_ss$   | $\tau\dot{\phi}=1$ 或 $\dot{\phi}=\omega$ |
| **相位变量** | $s$: 从 1 衰减到 0         | $\phi$: 线性增加或循环                    |
| **目标**     | 固定目标 $g$               | 振荡中心 $y^*$                            |
| **强迫项**   | $f(s)$: 随 $s$ 衰减到 0    | $f(\phi)$: 持续周期性驱动                 |
| **收敛性**   | 收敛到点（$s=0$）          | 收敛到极限环                              |
| **时间特性** | 有限时长，运动有起点和终点 | 无限时长，运动持续循环                    |
| **应用场景** | 抓取物体、单次移动         | 行走、搅拌、周期性任务                    |

**详细说明**

- **运动类型**: 离散型 DMP 用于从起点到终点的单次移动，节律性 DMP 用于持续的周期运动。
- **规范系统**: 离散型的 $s$ 通过 $\tau\dot{s}=-\alpha_ss$ 衰减，节律性的 $\phi$ 通过 $\tau\dot{\phi}=1$ 或 $\dot{\phi}=\omega$ 循环。
- **相位变量**: $s$ 单调递减，$\phi$ 持续增加并可模 $2\pi$。
- **目标**: 离散型目标是固定点 $g$，节律型围绕 $y^*$ 振荡。
- **强迫项**: $f(s)$ 最终消失，$f(\phi)$ 保持振荡。
- **收敛性**: 离散型收敛到点，节律型收敛到周期轨道。
- **时间特性**: 离散型有明确时长，节律型无限重复。
- **应用场景**: 离散型适合抓取，节律型适合行走。

**总结**

离散型 DMP 和节律性 DMP 通过规范系统和强迫项的不同设计，分别实现单次运动和周期运动，满足不同任务需求。

---

**DMP 模型学习和轨迹复现**

动态运动原语（DMP）是一种灵活的运动生成框架，通过学习示范轨迹并复现相似运动，广泛应用于机器人控制。以下详细描述 DMP（以离散型为例）的学习过程和轨迹复现步骤，节律型 DMP 的过程类似但细节稍有不同。

**DMP 模型概述**

离散型 DMP 的核心包括：

- 变换系统：$\tau\dot{v}=\alpha_z(\beta_z(g-y)-v)+f(s)$, $\tau\dot{y}=v$
- 规范系统：$\tau\dot{s}=-\alpha_ss$, 解为 $s(t)=e^{-\alpha_st/\tau}$
- 强迫项：$f(s)=\frac{\sum_{k=1}^{K}w_k\psi_k(s)}{\sum_{k=1}^{K}\psi_k(s)}s(g-y_0)$, $\psi_k(s)=e^{-h_k(s-c_k)^2}$  
  学习的目标是确定强迫项的权重 $w_k$，以复现示范轨迹。

**学习过程**

DMP 通过模仿学习从示范轨迹中提取强迫项，具体步骤如下：

1. **采集示范轨迹**

   - 获取一组时间序列数据：位置 $y_d(t)$，通常通过传感器或人工示范记录。
   - 计算导数：速度 $\dot{y}_d(t)$ 和加速度 $\ddot{y}_d(t)$（可用数值差分）。
   - 示例：$y_d(t)=[0,0.2,0.5,0.8,1]$，$t=[0,0.25,0.5,0.75,1]$。

2. **初始化参数**

   - 设定 $\tau$（示范时长，如 1 秒），$g=y_d(\tau)$（目标，如 1），$y_0=y_d(0)$（起点，如 0）。
   - 选择 $\alpha_z$, $\beta_z$（如 $\alpha_z=8$, $\beta_z=2$），$\alpha_s=1$。
   - 定义基函数：$K$（如 10），$c_k=\frac{k-1}{K-1}$，$h_k=\frac{1}{2(c_{k+1}-c_k)^2}$。

3. **计算目标强迫项 $f_d(t)$**

   - 根据变换系统反推：  
     $f_d(t)=\tau^2\ddot{y}_d+\alpha_z(\beta_z(g-y_d)-\tau\dot{y}_d)$
   - 将 $t$ 映射到 $s$：$s(t)=e^{-\alpha_st/\tau}$，得到 $f_d(s)$。
   - 示例：若 $y_d=0.5$, $\dot{y}_d=1$, $\ddot{y}_d=0$, $\tau=1$, 则 $f_d=8(2(1-0.5)-1)+0=0$。

4. **拟合权重 $w_k$**

   - 使用局部加权回归（LWR）：  
     $w_k=\frac{\sum_s\psi_k(s)s(f_d(s)/(g-y_0))}{\sum_s\psi_k(s)s^2}$
   - 离散化：对 $s$ 的采样点计算 $\psi_k(s)$ 和 $f_d(s)$，求解 $w_k$。
   - 意义：$w_k$ 使 $f(s)$ 逼近 $f_d(s)$，保留示范轨迹的形状。

5. **验证学习结果**
   - 用学习后的 $w_k$ 重新计算 $f(s)$，检查是否匹配 $f_d(s)$。
   - 若误差较大，可增加 $K$ 或调整 $h_k$。

**轨迹复现过程**

学习完成后，DMP 用训练好的模型生成新轨迹，步骤如下：

1. **初始化状态**

   - 设置初始条件：$y(0)=y_0$（可与示范不同），$v(0)=0$，$s(0)=1$。
   - 指定新目标 $g'$（可调整）和 $\tau'$（可缩放时长）。
   - 示例：$y_0=0$, $g'=1.5$, $\tau'=2$。

2. **时间步进**

   - 选择步长 $dt$（如 0.01），总步数 $N=\tau'/dt$。
   - $t$ 从 0 到 $\tau'$ 迭代。

3. **更新规范系统**

   - $\dot{s}=-\alpha_ss/\tau'$
   - $s(t+dt)=s(t)+\dot{s}dt$, 或 $s(t)=e^{-\alpha_st/\tau'}$。

4. **计算强迫项**

   - 对当前 $s$：  
     $f(s)=s(g'-y_0)\frac{\sum w_k\psi_k(s)}{\sum\psi_k(s)}$
   - $w_k$ 使用学习结果，$\psi_k(s)=e^{-h_k(s-c_k)^2}$。

5. **更新变换系统**

   - $\dot{v}=(\alpha_z(\beta_z(g'-y)-v)+f(s))/\tau'$
   - $v(t+dt)=v(t)+\dot{v}dt$
   - $\dot{y}=v$
   - $y(t+dt)=y(t)+vdt$

6. **生成轨迹**
   - 重复步骤 3-5，记录 $y(t)$，形成完整轨迹。
   - 结果：$y$ 从 $y_0$ 平滑移到 $g'$，形状与示范相似。

**节律性 DMP 的差异**

- **学习**:
  - 规范系统：$\tau\dot{\phi}=1$，$\phi(t)=t/\tau+\phi_0$。
  - 强迫项：$f(\phi)=\sum w_k\psi_k(\phi)$，$\psi_k(\phi)=e^{-h_k(\cos(\phi-c_k)-1)}$。
  - $f_d(t)$ 周期性，$w_k$ 拟合振荡模式。
- **复现**:
  - $\phi$ 持续增加，$f(\phi)$ 驱动周期轨迹，无衰减。

**示例**

- **离散型**: 示范 $y_d(t)=t^2$（$t=0$ 到 1），学习 $w_k$，复现时 $g'=2$，轨迹仍呈抛物线形状。
- **节律型**: 示范 $y_d(t)=\sin(2\pi t)$，学习后复现周期正弦波。

**总结**

DMP 通过从示范轨迹反推 $f_d$ 并拟合 $w_k$ 实现学习，利用规范系统和变换系统复现轨迹。离散型强迫项衰减，支持单次运动；节律型强迫项周期性，支持振荡运动。两者均可泛化到新目标和时长。

---

# 四、稳定性证明

## 1.1 BIBO 稳定性证明

目标：证明系统对有界输入 $f$ 产生有界输出 $y$ 和 $z$。

系统方程  
点吸引子系统：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
$\tau \dot{y} = z$  
$\tau \dot{x} = -\alpha_x x$  
其中 $f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$，$\Psi_i(x)$ 是高斯基函数。

步骤 1：无 $f$ 时的稳定性  
$\tau \dot{z} = \alpha_z \beta_z (g - y) - \alpha_z z$  
$\tau \dot{y} = z$  
定义误差 $e = y - g$，则 $\dot{e} = \dot{y} = z / \tau$。代入得：  
$\tau \dot{z} = -\alpha_z \beta_z e - \alpha_z z$  
状态空间形式：  
$\begin{bmatrix} \dot{e} \\ \dot{z} \end{bmatrix} = \begin{bmatrix} 0 & 1/\tau \\ -\alpha_z \beta_z / \tau & -\alpha_z / \tau \end{bmatrix} \begin{bmatrix} e \\ z \end{bmatrix} = A \begin{bmatrix} e \\ z \end{bmatrix}$  
特征方程：  
$\det(sI - A) = s^2 + \frac{\alpha_z}{\tau} s + \frac{\alpha_z \beta_z}{\tau^2} = 0$  
根为：  
$s = \frac{-\alpha_z \pm \sqrt{\alpha_z^2 - 4 \alpha_z \beta_z}}{2\tau}$  
设 $\alpha_z = 4$，$\beta_z = 1$，$\tau = 1$：  
$s = \frac{-4 \pm \sqrt{16 - 16}}{2} = -2$  
双重实根 $s = -2$，系统稳定且无振荡，解为 $e(t) = (c_1 + c_2 t) e^{-2t}$，指数收敛到 0。

步骤 2：加入有界 $f$  
重写系统：  
$\tau \dot{z} = \alpha_z \beta_z \left( \left( g + \frac{f}{\alpha_z \beta_z} \right) - y \right) - \alpha_z z$  
令 $u = g + \frac{f}{\alpha_z \beta_z}$，则：  
$\begin{bmatrix} \dot{e} \\ \dot{z} \end{bmatrix} = A \begin{bmatrix} e \\ z \end{bmatrix} + \begin{bmatrix} 0 \\ \alpha_z \beta_z / \tau \end{bmatrix} (u - g)$

- $f(x)$ 有界：$\Psi_i(x)$ 是高斯函数（0 到 1 之间），$w_i$ 是有限权重，$x$ 从 1 衰减到 0，$g - y_0$ 是常数。设 $|f(x)| \leq M$，则 $|u - g| \leq \frac{M}{\alpha_z \beta_z}$。
- 线性系统解：  
  $\begin{bmatrix} e(t) \\ z(t) \end{bmatrix} = e^{At} \begin{bmatrix} e(0) \\ z(0) \end{bmatrix} + \int_0^t e^{A(t-\sigma)} \begin{bmatrix} 0 \\ \alpha_z \beta_z / \tau \end{bmatrix} (u(\sigma) - g) d\sigma$
- $e^{At}$ 指数衰减（因 $A$ 的特征值负实部），积分项受 $|u - g|$ 界限约束。Friedland (1986) 指出，对稳定线性系统，有界输入产生有界输出，具体界限可用卷积积分估计：  
  $|e(t)| \leq |e^{At} e(0)| + \frac{\alpha_z \beta_z}{\tau} \int_0^t |e^{A(t-\sigma)}| \frac{M}{\alpha_z \beta_z} d\sigma$  
  $e^{At}$ 的范数 $\sim e^{-2t}$，积分收敛到常数，$e(t)$ 和 $z(t)$ 有界。

步骤 3：点吸引子收敛  
$x \to 0$ 时，$f(x) \to 0$（因 $x$ 因子），系统退化为无输入形式，指数收敛到 $g$。极限环时，$f(\phi, r)$ 周期性，输出为有界振荡。  
结论：系统 BIBO 稳定。

## 1.2 收缩稳定性证明

目标：用收缩理论证明系统稳定（参考 Perk & Slotine, 2006）。

收缩理论基础（Slotine & Li, 1991）  
系统 $\dot{x} = f(x, t)$ 是收缩的，如果其雅可比矩阵 $\frac{\partial f}{\partial x}$ 一致负定，即存在 $\lambda > 0$ 使：  
$\frac{\partial f}{\partial x} + \left( \frac{\partial f}{\partial x} \right)^T \leq -2\lambda I$  
则所有轨迹指数收敛。

层次结构

- 规范系统：$\tau \dot{x} = -\alpha_x x$
  - $\frac{\partial f}{\partial x} = -\alpha_x / \tau < 0$，收缩。
- 变换系统：  
  $\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
  $\tau \dot{y} = z$  
  雅可比矩阵：  
  $J = \begin{bmatrix} -\alpha_z \beta_z / \tau & -\alpha_z / \tau \\ 1/\tau & 0 \end{bmatrix}$  
  对称化：  
  $J + J^T = \begin{bmatrix} -2\alpha_z \beta_z / \tau & (1 - \alpha_z) / \tau \\ (1 - \alpha_z) / \tau & 0 \end{bmatrix}$  
  特征值需负。设 $\alpha_z = 4$，$\beta_z = 1$，$\tau = 1$：  
  $J + J^T = \begin{bmatrix} -8 & -3 \\ -3 & 0 \end{bmatrix}$  
  特征多项式：$\lambda^2 + 8\lambda + 9 = 0$，判别式 $64 - 36 = 28$。根：  
  $\lambda = \frac{-8 \pm \sqrt{28}}{2} < 0$  
  $f(x)$ 视为外部输入，因其有界且随 $x$ 衰减，不破坏收缩性。

结论：规范系统驱动变换系统，二者耦合后仍收缩，稳定。

# 五、不变特性的严格证明

文章引用的参考文献为 Jackson (1989)。

## 2.1 幅度缩放证明

目标：证明 $y \to k y$ 保持轨迹形状。

变换  
原系统：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
$\tau \dot{y} = z$  
$f(x) = \frac{\sum \Psi_i(x) w_i}{\sum \Psi_i(x)} x (g - y_0)$  
缩放后：$g \to k g$，$y_0 \to k y_0$，假设 $y \to k y$，$z \to k z$：

- $\dot{y} = z/\tau \to k \dot{y} = (k z)/\tau$，成立。
- $\tau (k \dot{z}) = \alpha_z (\beta_z (k g - k y) - k z) + f'(x)$
- $f'(x) = \frac{\sum \Psi_i(x) w_i}{\sum \Psi_i(x)} x (k g - k y_0) = k f(x)$
- 代入：$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$，形式不变。

拓扑等价  
轨迹 $y(t)$ 变为 $k y(t)$，是线性拉伸，保持方向和相对形状，为同胚映射。

## 2.2 时间缩放证明

目标：$\tau \to k \tau$，路径不变。

变换  
令 $t' = t/k$，$\tau' = k \tau$：

- $\dot{y} = \frac{dy}{dt} = \frac{dy}{dt' k} = \frac{z}{\tau} = \frac{z}{k \tau'} = \frac{\dot{y}}{k}$
- $\dot{z} = \frac{1}{\tau} [\alpha_z (\beta_z (g - y) - z) + f(x)] = \frac{\dot{z}}{k}$
- $\dot{x} = -\alpha_x x / (k \tau') = \dot{x}' / k$  
  轨迹 $y(t)$ 的形状由 $y$ 和 $z$ 的关系决定，不变，仅时间拉伸。

结论：两种缩放均为拓扑等价变换。

总结

- 稳定性：BIBO 用线性系统理论证明，收缩用雅可比分析，均严格成立。
- 不变性：幅度和时间缩放通过代入验证，满足拓扑等价。

---

# 六、状态空间理论基础复习

## 什么是状态空间表示？

状态空间方法用一组一阶微分方程描述系统，而不是直接用高阶微分方程。系统的“状态”是一组变量，包含了描述系统当前行为的所有信息。  
对于线性时不变（LTI）系统，状态空间模型通常写成：  
$\dot{x}(t) = A x(t) + B u(t)$  
$y(t) = C x(t) + D u(t)$

- $x(t)$：状态向量（n 维）。
- $u(t)$：输入向量（m 维）。
- $y(t)$：输出向量（p 维）。
- $A$：状态矩阵（n×n）。
- $B$：输入矩阵（n×m）。
- $C$：输出矩阵（p×n）。
- $D$：直通矩阵（p×m，常为 0）。

## 为什么用状态空间？

- 统一性：适用于 SISO 和 MIMO 系统。
- 易于分析：可以用矩阵代数分析稳定性、可控性、可观性等。
- 数值计算：适合计算机仿真。

## 稳定性分析

系统的稳定性由 $A$ 矩阵的特征值决定：

- 特征值 $\lambda_i$ 是方程 $\det(\lambda I - A) = 0$ 的根。
- 如果所有 $\lambda_i$ 的实部小于 0（即 $\text{Re}(\lambda_i) < 0$），系统稳定。
- 如果有特征值实部大于 0，系统不稳定。

## BIBO 稳定性

BIBO（Bounded-Input Bounded-Output）稳定性关注输入输出关系：

- 如果输入 $u(t)$ 有界（即 $|u(t)| \leq M < \infty$），输出 $y(t)$ 也必须有界。
- 对于 LTI 系统，BIBO 稳定等价于 $A$ 的特征值实部全负，且传递函数的极点都在左半平面。

## 系统解

`状态空间方程的解是`：  
$x(t) = e^{A t} x(0) + \int_0^t e^{A (t-\tau)} B u(\tau) d\tau$  
$y(t) = C e^{A t} x(0) + C \int_0^t e^{A (t-\tau)} B u(\tau) d\tau + D u(t)$

- $e^{A t}$ 是状态转移矩阵。
- 积分项是输入的卷积响应。

## 结合 BIBO 稳定性证明

系统方程  
点吸引子系统：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
$\tau \dot{y} = z$  
$\tau \dot{x} = -\alpha_x x$  
其中 $f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$。

步骤 1：转化为状态空间形式  
定义状态变量：

- $x_1 = y - g$（位置误差）。
- $x_2 = z$（速度）。  
  则：
- $\dot{x_1} = \dot{y} = z / \tau = x_2 / \tau$
- $\dot{x_2} = \dot{z} = \frac{1}{\tau} [\alpha_z (\beta_z (g - y) - z) + f(x)] = \frac{1}{\tau} [-\alpha_z \beta_z x_1 - \alpha_z x_2 + f(x)]$  
  状态空间形式：  
  $\begin{bmatrix} \dot{x_1} \\ \dot{x_2} \end{bmatrix} = \begin{bmatrix} 0 & 1/\tau \\ -\alpha_z \beta_z / \tau & -\alpha_z / \tau \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ 1/\tau \end{bmatrix} f(x)$
- $A = \begin{bmatrix} 0 & 1/\tau \\ -\alpha_z \beta_z / \tau & -\alpha_z / \tau \end{bmatrix}$
- $B = \begin{bmatrix} 0 \\ 1/\tau \end{bmatrix}$
- 输入 $u = f(x)$
- 输出 $y_{\text{out}} = x_1$，则 $C = [1 \, 0]$，$D = 0$。

步骤 2：无输入时的稳定性  
$f(x) = 0$ 时：  
$\dot{x} = A x$  
计算 $A$ 的特征值：  
$\det(s I - A) = s^2 + \frac{\alpha_z}{\tau} s + \frac{\alpha_z \beta_z}{\tau^2} = 0$  
$s = \frac{-\alpha_z \pm \sqrt{\alpha_z^2 - 4 \alpha_z \beta_z}}{2\tau}$  
设 $\alpha_z = 4$，$\beta_z = 1$，$\tau = 1$：  
$s = \frac{-4 \pm \sqrt{16 - 16}}{2} = -2$  
双重负实根 $s = -2$，系统稳定，$x_1$ 和 $x_2$ 指数衰减到 0，即 $y \to g$，$z \to 0$。

步骤 3：加入有界输入 $f(x)$  
重写系统：  
$\tau \dot{z} = \alpha_z \beta_z \left( \left( g + \frac{f(x)}{\alpha_z \beta_z} \right) - y \right) - \alpha_z z$  
定义 $u = g + \frac{f(x)}{\alpha_z \beta_z}$，则：  
$\dot{x} = A x + B (u - g)$

- $f(x)$ 有界：设 $|f(x)| \leq M$，则 $|u - g| \leq \frac{M}{\alpha_z \beta_z}$。
- 系统解：  
  $x(t) = e^{A t} x(0) + \int_0^t e^{A (t-\tau)} B (u(\tau) - g) d\tau$  
  $y_{\text{out}}(t) = C e^{A t} x(0) + C \int_0^t e^{A (t-\tau)} B (u(\tau) - g) d\tau$

步骤 4：BIBO 稳定性

- 自然响应：$e^{A t} x(0)$ 指数衰减到 0。
- 强迫响应：积分项决定 BIBO 性。$e^{A t}$ 的范数 $\|e^{A t}\| \leq K e^{-\lambda t}$，$\lambda = 2$，$K$ 是常数。则：  
  $\left| \int_0^t e^{A (t-\tau)} B (u(\tau) - g) d\tau \right| \leq \int_0^t K e^{-\lambda (t-\tau)} \frac{1}{\tau} \frac{M}{\alpha_z \beta_z} d\tau$  
  令 $s = t - \tau$：  
  $\int_0^t K e^{-\lambda s} \frac{M}{\tau \alpha_z \beta_z} ds = \frac{K M}{\tau \alpha_z \beta_z} \frac{1 - e^{-\lambda t}}{\lambda}$  
  $t \to \infty$ 时：$\leq \frac{K M}{\tau \alpha_z \beta_z} \frac{1}{\lambda}$，有界。

步骤 5：点吸引子特例  
$x \to 0$，$f(x) \to 0$，$u \to g$，强迫响应消失，$y \to g$。

---

# 七、 DMP 模型学习

这一部分的目标是从观察到的运动行为（比如演示数据）中，通过学习调整动态运动基元（DMPs）的参数，特别是非线性力 $f$ 的权重 $w_i$，让系统重现这些行为。作者提出了一种基于局部加权回归（Locally Weighted Regression, LWR）的方法，既简单又高效。

## 基本思路

DMPs 的核心是一个基础动态系统（弹簧阻尼模型），加上一个可学习的非线性力 $f$。通过从演示数据中提取目标 $f$，然后用回归方法拟合，就能让系统生成与演示一致的吸引子行为（点吸引子或极限环吸引子）。这就像教一个机器人模仿你的动作：你先示范一遍，机器人通过分析你的轨迹，调整它的“推力”来复制。

## 点吸引子系统的学习

### 系统回顾

点吸引子系统的方程是：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
$\tau \dot{y} = z$  
$\tau \dot{x} = -\alpha_x x$  
其中 $f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$，$\Psi_i(x) = \exp\left(-\frac{1}{2\sigma_i^2} (x - c_i)^2\right)$ 是高斯基函数，$w_i$ 是待学习的权重。

### 演示数据

假设我们有一组演示数据，包括位置 $y_{\text{demo}}(t)$、速度 $\dot{y}_{\text{demo}}(t)$ 和加速度 $\ddot{y}_{\text{demo}}(t)$，以及初始位置 $y_0$ 和目标位置 $g$。这些数据可以从传感器（如运动捕捉）或仿真中获得。

### 目标：计算 $f_{\text{target}}$

我们希望系统生成的 $y(t)$ 匹配 $y_{\text{demo}}(t)$。从变换系统方程出发：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(x)$  
因为 $\dot{y} = z / \tau$，所以 $z = \tau \dot{y}$，$\dot{z} = \tau \ddot{y}$。代入得：  
$\tau (\tau \ddot{y}) = \alpha_z (\beta_z (g - y) - \tau \dot{y}) + f$  
整理为：  
$f = \tau^2 \ddot{y} - \alpha_z (\beta_z (g - y) - \tau \dot{y})$  
将演示数据代入，定义目标力：  
$f_{\text{target}}(t) = \tau^2 \ddot{y}_{\text{demo}}(t) - \alpha_z (\beta_z (g - y_{\text{demo}}(t)) - \tau \dot{y}_{\text{demo}}(t))$

- 通俗解释：$f_{\text{target}}$ 是演示轨迹需要的“额外推力”，减去弹簧和阻尼的贡献后剩下的部分。

### 学习 $f(x)$ 的权重

现在我们知道每个时刻的目标 $f_{\text{target}}(t)$，需要调整 $f(x)$ 的权重 $w_i$ 让 $f(x(t))$ 尽量接近 $f_{\text{target}}(t)$。因为 $x(t)$ 是规范系统的解（$x(t) = e^{-\alpha_x t / \tau}$），我们可以把 $f_{\text{target}}(t)$ 看作 $x$ 的函数。

### 局部加权回归（LWR）

作者使用 LWR 来拟合 $f(x)$：  
$f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$

- $\Psi_i(x)$ 是基函数，决定了每个权重 $w_i$ 的影响范围。
- $x (g - y_0)$ 是调制项，确保 $f$ 随 $x$ 衰减并与幅度相关。  
  LWR 的目标是最小化加权误差：  
  $J = \sum_t \Psi_i(x(t)) (f_{\text{target}}(t) - w_i x(t) (g - y_0))^2$  
  对 $w_i$ 求导并令其为 0：  
  $w_i = \frac{\sum_t \Psi_i(x(t)) f_{\text{target}}(t) x(t) (g - y_0)}{\sum_t \Psi_i(x(t)) (x(t) (g - y_0))^2}$  
  文章中直接给出了简化的形式（假设离散时间步长一致）：  
  $w_i = \frac{\sum_t \Psi_i(t) f_{\text{target}}(t)}{\sum_t \Psi_i(t)}$
- 通俗解释：每个 $w_i$ 是 $f_{\text{target}}$ 在 $\Psi_i$ 覆盖区域的加权平均，$\Psi_i$ 像一个“放大镜”，只关注 $x$ 靠近 $c_i$ 的部分。

### 离散化处理

实际中，演示数据是离散的（时间步长 $\Delta t$）：

- $y_{\text{demo}}(t_k)$ 是位置序列。
- $\dot{y}_{\text{demo}}(t_k) \approx \frac{y_{\text{demo}}(t_{k+1}) - y_{\text{demo}}(t_k)}{\Delta t}$
- $\ddot{y}_{\text{demo}}(t_k) \approx \frac{\dot{y}_{\text{demo}}(t_{k+1}) - \dot{y}_{\text{demo}}(t_k)}{\Delta t}$  
  计算 $f_{\text{target}}(t_k)$，然后用 LWR 拟合。

## 极限环吸引子系统的学习

### 系统回顾

极限环系统：  
$\tau \dot{z} = \alpha_z (\beta_z (g - y) - z) + f(\phi, r)$  
$\tau \dot{y} = z$  
$\tau \dot{\phi} = 1$  
其中 $f(\phi, r) = \frac{\sum_{i=1}^N \Psi_i w_i}{\sum_{i=1}^N \Psi_i} r$，$\Psi_i = \exp(h_i (\cos(\phi - c_i) - 1))$ 是周期性基函数。

### 目标 $f_{\text{target}}$

类似点吸引子：  
$f_{\text{target}}(t) = \tau^2 \ddot{y}_{\text{demo}}(t) - \alpha_z (\beta_z (g - y_{\text{demo}}(t)) - \tau \dot{y}_{\text{demo}}(t))$  
但这里 $\phi(t) = t / \tau + \phi_0$ 是线性增加的相位，$r$ 是振幅（可以是常数或从数据估计）。

### 学习权重

$f(\phi) = \frac{\sum_{i=1}^N \Psi_i(\phi) w_i}{\sum_{i=1}^N \Psi_i(\phi)} r$  
用 LWR：  
$w_i = \frac{\sum_t \Psi_i(\phi(t)) f_{\text{target}}(t)}{\sum_t \Psi_i(\phi(t)) r^2}$

- $\Psi_i(\phi)$ 是周期性的，覆盖一个周期（如 0 到 $2\pi$）。
- 如果 $r$ 是常数，直接用；如果 $r$ 随时间变，可用 $r(t) = \sqrt{(\tau \dot{y})^2 + (\beta_z (g - y))^2}$ 估计。

## 实际应用中的细节

### 参数选择

- $N$（基函数数量）：点吸引子用 10-50 个，极限环用 5-20 个，取决于轨迹复杂度。
- $\sigma_i$ 或 $h_i$：控制基函数宽度，太窄过拟合，太宽欠拟合。
- $\alpha_z$、$\beta_z$：通常 $\alpha_z = 4$，$\beta_z = 1$（临界阻尼）。

### 鲁棒性

- 数据噪声可能导致 $f_{\text{target}}$ 不平滑，但 LWR 的局部性可以缓解。
- 时间对齐：点吸引子用 $x$ 自动对齐，极限环需手动确定周期。

例子（图 6 和图 7）

- 图 6：从手写字母轨迹学点吸引子，重现了路径。
- 图 7：从循环运动学极限环，复制了振荡行为。

总结  
学习过程是从演示轨迹反推出 $f_{\text{target}}$，然后用 LWR 调整 $w_i$，让 $f$ 匹配目标。点吸引子关注单次运动，极限环关注周期性。这方法简单高效，适合机器人模仿学习。

---

## 最简单示例

为了更清楚，我给一个简单的点吸引子学习推导。  
假设演示数据：

- $y_{\text{demo}}(t) = g (1 - e^{-t})$，$g = 1$，$y_0 = 0$，$\tau = 1$。
- $\dot{y}_{\text{demo}}(t) = e^{-t}$
- $\ddot{y}_{\text{demo}}(t) = -e^{-t}$  
  设 $\alpha_z = 4$，$\beta_z = 1$：  
  $f_{\text{target}}(t) = 1^2 (-e^{-t}) - 4 (1 (1 - (1 - e^{-t})) - 1 \cdot e^{-t})$  
  $= -e^{-t} - 4 (e^{-t} - e^{-t}) = -e^{-t}$  
  规范系统：$\dot{x} = -x$，$x(t) = e^{-t}$，所以 $f_{\text{target}}(x) = -x$。  
  用 $f(x) = \frac{\sum \Psi_i(x) w_i}{\sum \Psi_i(x)} x$ 拟合，设 $N=1$，$\Psi_1(x) = 1$：  
  $w_1 = \frac{\sum (-x) x}{\sum x^2} = -1$  
  $f(x) = -x$，完美匹配。

---

# 八、LWR 具体推导过程

## 局部加权回归背景

LWR 是一种监督学习方法，目标是从数据中拟合一组参数（如权重 $w_i$），使模型输出尽量接近目标值。它的“局部性”体现在每个参数 $w_i$ 主要拟合数据中靠近某个特定区域的部分，通过权重函数（如高斯基函数）实现。文章中用 LWR 学习非线性力 $f(x)$ 的权重 $w_i$，以匹配目标 $f_{\text{target}}(t)$。

## 点吸引子系统的模型

点吸引子系统的非线性力定义为：  
$f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$

- $\Psi_i(x) = \exp\left(-\frac{1}{2\sigma_i^2} (x - c_i)^2\right)$ 是高斯基函数，$c_i$ 是中心，$\sigma_i$ 是宽度。
- $x (g - y_0)$ 是调制项，确保 $f$ 随 $x$ 衰减并与幅度相关。
- $w_i$ 是待学习的权重。  
  目标是从演示数据计算的 $f_{\text{target}}(t)$，通过 $x(t)$ 映射到 $f_{\text{target}}(x)$，然后拟合 $f(x)$。

## 数据假设

给定离散演示数据：时间序列 $t_k$，位置 $y_{\text{demo}}(t_k)$，速度 $\dot{y}_{\text{demo}}(t_k)$，加速度 $\ddot{y}_{\text{demo}}(t_k)$。  
目标力：  
$f_{\text{target}}(t_k) = \tau^2 \ddot{y}_{\text{demo}}(t_k) - \alpha_z (\beta_z (g - y_{\text{demo}}(t_k)) - \tau \dot{y}_{\text{demo}}(t_k))$  
规范系统：$\tau \dot{x} = -\alpha_x x$，解为 $x(t) = x_0 e^{-\alpha_x t / \tau}$，所以每个 $t_k$ 对应一个 $x(t_k)$。

## LWR 目标函数

LWR 的核心是定义一个误差函数，对每个权重 $w_i$ 单独优化。假设我们有 $K$ 个数据点 $(t_k, f_{\text{target}}(t_k), x(t_k))$，目标是最小化加权平方误差：  
$J_i = \sum_{k=1}^K \Psi_i(x(t_k)) (f_{\text{target}}(t_k) - f_i(x(t_k)))^2$  
其中 $f_i(x) = w_i x (g - y_0)$ 是第 $i$ 个基函数的贡献，$\Psi_i(x(t_k))$ 是权重函数，表示数据点 $x(t_k)$ 对 $w_i$ 的影响。

- 通俗解释：$\Psi_i$ 像一个“放大镜”，当 $x(t_k)$ 靠近 $c_i$ 时放大误差，远离时忽略。

## 完整 $f(x)$ 的误差

实际中，$f(x) = \frac{\sum_{i=1}^N \Psi_i(x) w_i}{\sum_{i=1}^N \Psi_i(x)} x (g - y_0)$ 是所有基函数的加权和。但 LWR 假设每个 $w_i$ 独立拟合局部数据，近似为：  
$f(x(t_k)) \approx w_i x(t_k) (g - y_0)$ （当 $x(t_k)$ 靠近 $c_i$ 时，$\Psi_i$ 占主导）。  
总误差为：  
$J = \sum_{k=1}^K \sum_{i=1}^N \Psi_i(x(t_k)) (f_{\text{target}}(t_k) - w_i x(t_k) (g - y_0))^2$  
但文章简化了计算，对每个 $w_i$ 单独优化。

## 推导 $w_i$

以 $J_i$ 为目标，优化单个 $w_i$：  
$J_i = \sum_{k=1}^K \Psi_i(x_k) (f_{\text{target}}(t_k) - w_i x_k (g - y_0))^2$  
（简记 $x_k = x(t_k)$）。

1. 展开误差：  
   $J_i = \sum_{k=1}^K \Psi_i(x_k) (f_{\text{target}}^2(t_k) - 2 w_i x_k (g - y_0) f_{\text{target}}(t_k) + w_i^2 x_k^2 (g - y_0)^2)$
2. 对 $w_i$ 求导：  
   $\frac{\partial J_i}{\partial w_i} = \sum_{k=1}^K \Psi_i(x_k) \left( -2 x_k (g - y_0) f_{\text{target}}(t_k) + 2 w_i x_k^2 (g - y_0)^2 \right)$  
   $= -2 (g - y_0) \sum_{k=1}^K \Psi_i(x_k) x_k f_{\text{target}}(t_k) + 2 w_i (g - y_0)^2 \sum_{k=1}^K \Psi_i(x_k) x_k^2$
3. 令导数为 0：  
   $-2 (g - y_0) \sum_{k=1}^K \Psi_i(x_k) x_k f_{\text{target}}(t_k) + 2 w_i (g - y_0)^2 \sum_{k=1}^K \Psi_i(x_k) x_k^2 = 0$  
   $w_i (g - y_0) \sum_{k=1}^K \Psi_i(x_k) x_k^2 = \sum_{k=1}^K \Psi_i(x_k) x_k f_{\text{target}}(t_k)$  
   $w_i = \frac{\sum_{k=1}^K \Psi_i(x_k) x_k f_{\text{target}}(t_k)}{\sum_{k=1}^K \Psi_i(x_k) x_k^2 (g - y_0)}$

- 通俗解释：分子是 $f_{\text{target}}$ 和 $x$ 的加权协方差，分母是 $x^2$ 的加权方差，$w_i$ 是局部区域的“斜率”。

## 文章中的简化形式

文章给出的公式是：  
$w_i = \frac{\sum_t \Psi_i(t) f_{\text{target}}(t)}{\sum_t \Psi_i(t)}$  
这与推导结果不同。原因可能是：

- 文章假设 $x (g - y_0)$ 被分离出来，拟合的是 $\frac{f_{\text{target}}(t)}{x(t) (g - y_0)}$ 的加权平均：  
  定义 $s(t) = \frac{f_{\text{target}}(t)}{x(t) (g - y_0)}$，目标是 $f(x) / (x (g - y_0)) = \frac{\sum \Psi_i(x) w_i}{\sum \Psi_i(x)}$。  
  误差变为：  
  $J_i = \sum_{k=1}^K \Psi_i(x_k) (s(t_k) - w_i)^2$  
  求导：  
  $\frac{\partial J_i}{\partial w_i} = \sum_{k=1}^K \Psi_i(x_k) (-2 s(t_k) + 2 w_i) = 0$  
  $w_i \sum_{k=1}^K \Psi_i(x_k) = \sum_{k=1}^K \Psi_i(x_k) s(t_k)$  
  $w_i = \frac{\sum_{k=1}^K \Psi_i(x_k) s(t_k)}{\sum_{k=1}^K \Psi_i(x_k)} = \frac{\sum_{k=1}^K \Psi_i(x_k) \frac{f_{\text{target}}(t_k)}{x_k (g - y_0)}}{\sum_{k=1}^K \Psi_i(x_k)}$
- 但文章直接用 $\Psi_i(t)$，可能是简化符号或假设 $x (g - y_0)$ 在实现中单独处理。

## 校正推导

正确的 LWR 应为：  
$w_i = \frac{\sum_{k=1}^K \Psi_i(x_k) f_{\text{target}}(t_k)}{\sum_{k=1}^K \Psi_i(x_k) x_k (g - y_0)}$ （若 $f_{\text{target}}$ 直接拟合）。  
或：  
$w_i = \frac{\sum_{k=1}^K \Psi_i(x_k) \frac{f_{\text{target}}(t_k)}{x_k (g - y_0)}}{\sum_{k=1}^K \Psi_i(x_k)}$ （若拟合归一化后的 $f$）。  
文章公式可能是代码实现中的近似，实际应根据 $f$ 的定义调整。

## 极限环的 LWR

极限环中 $f(\phi, r) = \frac{\sum_{i=1}^N \Psi_i(\phi) w_i}{\sum_{i=1}^N \Psi_i(\phi)} r$，推导类似：  
$J_i = \sum_{k=1}^K \Psi_i(\phi_k) (f_{\text{target}}(t_k) - w_i r(t_k))^2$  
$w_i = \frac{\sum_{k=1}^K \Psi_i(\phi_k) f_{\text{target}}(t_k)}{\sum_{k=1}^K \Psi_i(\phi_k) r^2(t_k)}$

- 若 $r$ 为常数，分母简化。

---
