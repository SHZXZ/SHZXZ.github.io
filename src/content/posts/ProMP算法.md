---
title: promp算法
date: 2025-05-01
lastMod: 2025-07-21T12:26:05.311945Z
summary: 介绍一下promp算法
category: MP类算法
tags: [技能学习, MP类算法, 模仿学习]
---

# ProMP 算法

## Abstract

运动原语（MPs）是一种成熟的方法，用于表示模块化且可重用的机器人运动生成器。许多最先进的机器人学习成果都基于运动原语，因其能够紧凑地表示机器人运动固有的连续性和高维特性。机器人学习的一个主要目标是将多个运动原语作为构建模块，在模块化控制架构中组合，以解决复杂任务。为此，`运动原语的表示必须支持运动之间的融合、适应变化的任务变量，以及并行激活多个运动原语。`我们提出了一种运动原语的概率表述形式，通过维护轨迹上的概率分布来实现运动建模。我们的概率方法允许推导新的操作，这些操作对于在一个框架内实现上述所有属性至关重要。为了将这种轨迹分布用于机器人运动控制，我们解析地推导了一个`随机反馈控制器`，该控制器能够重现给定的轨迹分布。我们在多个模拟和真实机器人场景中评估并比较了我们的方法与现有方法的表现。

总结：  
这篇论文提出了一个新的概率运动原语（ProMPs）框架，用概率分布来表示机器人动作，解决了动作融合、适应性和并行激活的问题。他们还设计了一个控制器，让机器人按这个分布执行动作，并在实验中验证了效果。

## Introduction

运动原语（MPs）在机器人学中被广泛用于表示和学习基本运动，例如击打、抓取等[1, 2, 3]。运动原语表述是机器人控制策略的紧凑参数化形式。通过调制其参数，可以实现模仿学习、强化学习以及适应不同场景。运动原语已被用于解决许多复杂任务，包括“杯中球”[4]、抛球[5, 6]、翻煎饼[7]和系绳球[8]。运动原语的目标是允许通过模块化控制架构，从基本运动中组合出复杂的机器人技能。因此，我们需要一种运动原语架构，支持运动原语的`并行激活和平滑融合`，以组合出依次[9]和同时[10]激活的原语所形成的复杂运动。此外，适应新任务或新情境需要`调制运动原语，以适应更改后的目标位置、目标速度或通过点`[3]。另外，运动的执行速度需要可调，以改变例如击球运动的速度。由于我们希望从数据中学习运动，另一个关键要求是运动原语的参数应`易于通过演示学习以及通过试错进行强化学习。`理想情况下，同一架构应适用于基于冲程和周期性的运动，并且能够在确定性和随机环境中表示最优行为。虽然许多现有运动原语架构实现了上述一个或多个属性[1, 11, 10, 2, 12, 13, 14, 15]，<font color="#ff0000">但尚无一种方法在一个框架内展示所有这些属性。</font>例如，[13]也提供了运动原语的概率解释，通过将运动原语表示为学习到的<font color="#ff0000">图模型</font>。然而，该方法高度<font color="#ff0000">依赖所使用规划器</font>的质量，且运动<font color="#ff0000">无法进行时间缩放。</font>Rozo 等人[12, 16]使用了原语的组合，但其运动原语的控制策略基于<font color="#ff0000">启发式方法</font>，且原语组合对最终运动的<font color="#ff0000">影响尚不清楚</font>。在本文中，我们引入了概率运动原语（ProMPs）的概念，作为表示和学习运动原语的通用概率框架。这种概率运动原语是轨迹上的分布。使用分布使我们能够通过概率理论中的操作来实现上述属性。例如，通过条件化目标位置或速度，可以实现运动到新目标的调制。类似地，通过两个独立轨迹概率分布的乘积，可以实现两个基本行为的稳定并行激活。此外，轨迹分布还可以编码运动的方差，因此概率运动原语通常能直接编码随机系统中的最优行为[17]。最后，概率框架允许我们建模不同自由度轨迹之间的协方差，可用于耦合机器人的关节。此类轨迹分布的属性至今尚未被充分用于表示和学习运动原语。`这种方法缺失的主要原因是难以从轨迹分布中提取控制机器人的策略。`我们展示了如何完成这一步骤，并推导了一个精确重现给定轨迹分布的控制策略。<font color="#ff0000">据我们所知，我们提出了第一个利用概率理论操作能力的原则性运动原语方法。</font>虽然概率运动原语的表示引入了许多新颖组件，但它结合了先前已知运动原语表示的许多优势[18, 10]，例如用于运动时间调整的相位变量（支持运动的时间缩放），以及表示节奏和基于冲程运动的能力。然而，由于概率运动原语纳入了演示的`方差`，其表示的更高灵活性和有利属性需要以多次演示为代价来学习原语，而不像过去的方法[18, 3]那样可以从单次演示中克隆运动。

总结：  
这部分引言介绍了运动原语（MPs）的背景和应用，指出其在机器人学习中的重要性和现有方法的局限性。作者提出了概率运动原语（ProMPs），用概率分布表示轨迹，支持动作融合、调制、并行激活等功能，并解决了从分布中提取控制策略的难题。ProMPs 结合了现有方法的优点，但需要更多演示数据来学习。

## **2.1 概率轨迹表示**

我们将单次运动执行建模为一条轨迹 $\tau = \{q_t\}_{t=0 \ldots T}$，由时间上的关节角度 $q_t$ 定义。在我们的框架中，一个运动原语描述了执行运动的多种方式，这自然引出了轨迹上的概率分布。

### **编码运动的时变方差**

我们的运动原语表示通过建模轨迹的时变方差，能够捕捉具有高变异性的多次演示。表示方差信息至关重要，因为它反映了运动执行中单个时间点的重要性，并且通常是表示随机系统中最优行为的必要条件[17]。  
我们使用权重向量 $w$ 紧凑地表示一条轨迹。给定权重向量 $w$，观测到轨迹 $\tau$ 的概率由线性基函数模型给出：

$$
y_t = \begin{bmatrix} q_t \\ \dot{q}_t \end{bmatrix} = \Phi_t^T w + \epsilon_y, \quad p(\tau \mid w) = \prod_t \mathcal{N}\left(y_t \mid \Phi_t^T w, \Sigma_y\right),
$$

其中 $\Phi_t = \left[\phi_t, \dot{\phi}_t\right]$ 定义了 $n \times 2$ 维时间依赖基矩阵，用于表示关节位置 $q_t$ 和速度 $\dot{q}_t$，$n$ 表示基函数数量，$\epsilon_y \sim \mathcal{N}(0, \Sigma_y)$ 是零均值独立同分布高斯噪声。通过用参数向量 $w$ 对基函数 $\Psi_t$ 加权，我们可以表示一条轨迹的均值。  
为了捕捉轨迹的方差，我们引入了权重向量 $w$ 上的分布 $p(w; \theta)$，其参数为 $\theta$。轨迹分布 $p(\tau; \theta)$ 现在可以通过边缘化权重向量 $w$ 计算，即 $p(\tau; \theta) = \int p(\tau \mid w) p(w; \theta) \mathrm{d}w$。分布 $p(\tau; \theta)$ 定义了一个层次贝叶斯模型（HBM），其参数由观测噪声方差 $\Sigma_y$ 和 $p(w; \theta)$ 的参数 $\theta$ 给出。

#### 解释说明

我们把机器人的动作看成一条轨迹 $\tau$，就像记录了手臂关节随时间变化的角度 $q_t$。在我们的方法里，一个运动原语（MP）不是固定一条轨迹，而是描述了很多可能的轨迹，所以用概率分布来表示。

**为啥要考虑方差？**  
我们不光记录动作的平均路径，还记录动作的“抖动范围”（方差），因为同一个动作可能每次都不完全一样（比如挥拍可能快一点或慢一点）。这种方差信息很重要，它能告诉你哪些时间点对动作更关键（比如击球那一刻得准），还能让机器人在不确定的环境中表现更好。

**怎么表示轨迹？**  
我们用一个权重向量 $w$ 来压缩一条轨迹。轨迹的状态 $y_t$（包括位置 $q_t$ 和速度 $\dot{q}_t$）是用一堆基函数（像积木块）拼出来的，$w$ 决定怎么拼，再加一点随机噪声 $\epsilon_y$。轨迹的概率 $p(\tau \mid w)$ 告诉你这条轨迹在给定 $w$ 下的可能性。

**怎么表示方差？**  
为了表示轨迹的多样性，我们让 $w$ 也服从一个分布 $p(w; \theta)$，而不是固定一个值。最后，轨迹的分布 $p(\tau; \theta)$ 是把所有可能的 $w$ 都考虑进去，算出来的。这就是一个层次贝叶斯模型，参数是噪声 $\Sigma_y$ 和 $p(w; \theta)$ 的 $\theta$。

#### 公式推导

1. **轨迹表示**：  
   轨迹 $\tau$ 是时间序列 $\{q_t\}$，每个时间点 $t$ 的状态 $y_t$ 包含位置和速度：
   $$
   y_t = \begin{bmatrix} q_t \\ \dot{q}_t \end{bmatrix} = \Phi_t^T w + \epsilon_y
   $$

- $\Phi_t = \left[\phi_t, \dot{\phi}_t\right]$ 是基函数矩阵，$\phi_t$ 和 $\dot{\phi}_t$ 分别是位置和速度的基函数，维度是 $n \times 2$（$n$ 是基函数数量）。
- $w$ 是权重向量，决定基函数怎么组合。
- $\epsilon_y \sim \mathcal{N}(0, \Sigma_y)$ 是高斯噪声，表示随机抖动。

2. **概率分布**：  
   给定 $w$，$y_t$ 服从高斯分布：
   $$
   p(y_t \mid w) = \mathcal{N}(y_t \mid \Phi_t^T w, \Sigma_y)
   $$

整条轨迹 $\tau = \{y_t\}$ 的概率是所有时间点的联合概率（假设独立）：

$$
p(\tau \mid w) = \prod_t p(y_t \mid w) = \prod_t \mathcal{N}\left(y_t \mid \Phi_t^T w, \Sigma_y\right)
$$

3. **引入 $w$ 的分布**：  
   为了表示轨迹的多样性，我们让 $w$ 服从分布 $p(w; \theta)$，通常是高斯分布：
   $$
   p(w; \theta) = \mathcal{N}(w \mid \mu_w, \Sigma_w)
   $$

这里的 $\theta = \{\mu_w, \Sigma_w\}$ 是分布参数。

4. **边缘化计算轨迹分布**：  
   轨迹分布 $p(\tau; \theta)$ 是对 $w$ 积分得到的：
   $$
   p(\tau; \theta) = \int p(\tau \mid w) p(w; \theta) \mathrm{d}w
   $$

- **意义**：这表示考虑所有可能的 $w$，计算轨迹 $\tau$ 的概率。
- **结果**：因为 $p(\tau \mid w)$ 和 $p(w; \theta)$ 都是高斯分布，积分后 $p(\tau; \theta)$ 也是高斯的（具体计算复杂，通常用数值方法）。

#### 概率论知识回忆

> [!note]+ 边缘化
> “边缘化”（Marginalization）是概率论中的一个术语，指的是通过对某些变量积分（或求和），得到其他变量的概率分布。

> [!note]+ 层次贝叶斯模型
> 层次贝叶斯模型（Hierarchical Bayesian Model, HBM）是一种概率模型，通过引入多层次的随机变量和参数来建模复杂系统。
>
> - **层次结构**：模型中的变量和参数分层定义，每一层依赖于上一层的参数，层层嵌套。
> - **贝叶斯方法**：使用贝叶斯推理，通过先验分布和似然函数计算后验分布，处理不确定性。

#### 总结

这一节介绍了概率运动原语（ProMPs）的核心：用概率分布表示轨迹，捕捉动作的时变方差。通过权重向量 $w$ 和基函数表示轨迹均值，再用 $w$ 的分布 $p(w; \theta)$ 表示轨迹的方差，最后通过边缘化得到轨迹分布 $p(\tau; \theta)$，形成一个层次贝叶斯模型。

### **时间调制**

时间调制是实现运动更快或更慢执行的必要手段。我们引入一个相位变量 $z$，以将运动与时间信号解耦，这一方法与先前非概率方法[18]类似。相位 $z(t)$ 可以是任何随时间单调递增的函数。通过改变相位变量的变化速率，我们可以调制运动的速度。不失一般性，我们定义运动开始时相位为 $z_0 = 0$，运动结束时为 $z_T = 1$。基函数 $\phi_t$ 现在直接依赖于相位而非时间，即 $\phi_t = \phi(z_t)$，其对应的导数变为 $\dot{\phi}_t = \phi'(z_t) \dot{z}_t$。

#### 解释说明

**时间调制是干嘛的？**  
时间调制就是让机器人动作可以快点或慢点，比如挥拍击球时，你可能想让挥拍快一点（更用力）或慢一点（更轻柔）。直接改时间 $t$ 会很麻烦，所以我们用一个“相位变量” $z$ 来控制速度。

- **相位 $z$ 是个啥？**  
  相位 $z$ 就像一个“进度条”，告诉你动作进行到哪儿了。它跟时间 $t$ 有关，但可以自由调整。$z$ 从 0 开始（动作刚开始），到 1 结束（动作完成），中间怎么变化可以自己定，只要保证 $z$ 一直递增（进度条不会倒退）。
- **怎么调速度？**  
  通过控制 $z$ 变化的快慢（$\dot{z}_t$），就能让动作加速或减速。比如 $z$ 涨得快，动作就快；涨得慢，动作就慢。
- **基函数咋变？**  
  原来基函数 $\phi_t$ 是直接跟时间 $t$ 绑定的，现在改成跟 $z$ 绑定，变成 $\phi(z_t)$。基函数的导数（变化率）也得跟着变，用链式法则算出来：$\dot{\phi}_t = \phi'(z_t) \dot{z}_t$，意思是基函数的变化速度取决于 $z$ 的变化速度。

#### 推导与细节

让我们一步步分析时间调制的实现：

1. **相位变量 $z$ 的定义**：

- $z(t)$ 是一个单调递增函数，满足：
  - 运动开始时（$t=0$）：$z_0 = z(0) = 0$。
  - 运动结束时（$t=T$）：$z_T = z(T) = 1$。
- 例如，可以简单定义 $z(t) = t/T$（线性相位），也可以用非线性函数，比如 $z(t) = 1 - e^{-at}$（指数增长）。
- 相位变化率 $\dot{z}_t = \frac{\mathrm{d}z}{\mathrm{d}t}$ 决定了 $z$ 随时间的变化速度。

2. **基函数的依赖调整**：

- 原来基函数 $\phi_t$ 直接依赖时间 $t$，现在改成依赖相位 $z$：

  $$
  \phi_t = \phi(z_t)
  $$

- **导数计算**：

  - 原来的导数是 $\dot{\phi}_t = \frac{\mathrm{d}\phi_t}{\mathrm{d}t}$。
  - 现在 $\phi_t = \phi(z_t)$，用链式法则：

    $$
    \dot{\phi}_t = \frac{\mathrm{d}\phi(z_t)}{\mathrm{d}t} = \frac{\mathrm{d}\phi(z_t)}{\mathrm{d}z_t} \cdot \frac{\mathrm{d}z_t}{\mathrm{d}t} = \phi'(z_t) \dot{z}_t
    $$

  - 其中 $\phi'(z_t) = \frac{\mathrm{d}\phi}{\mathrm{d}z}$ 是基函数对相位的导数，$\dot{z}_t$ 是相位对时间的导数。

3. **调制速度的实现**：

- 轨迹表示为 $y_t = \Phi_t^T w + \epsilon_y$，其中 $\Phi_t = [\phi_t, \dot{\phi}_t]$。
- 引入相位后：
  - $\phi_t = \phi(z_t)$。
  - $\dot{\phi}_t = \phi'(z_t) \dot{z}_t$。
- 如果改变 $\dot{z}_t$：
  - $\dot{z}_t$ 变大（相位变化快），$\dot{\phi}_t$ 变大，动作加速。
  - $\dot{z}_t$ 变小（相位变化慢），$\dot{\phi}_t$ 变小，动作减速。
- **例子**：
  - 假设 $z(t) = t/T$（线性），则 $\dot{z}_t = 1/T$。
  - 如果想让动作快一倍，可以改成 $z(t) = 2t/T$（$t$ 从 0 到 $T/2$ 时 $z$ 从 0 到 1），此时 $\dot{z}_t = 2/T$，$\dot{\phi}_t$ 变为原来的两倍，动作速度加倍。

**在 ProMPs 中的意义**：

- 相位变量 $z$ 解耦了时间和动作，让速度调整更灵活。
- 概率框架下，轨迹分布 $p(\tau; \theta)$ 仍然成立，只是基函数 $\Phi_t$ 现在依赖 $z_t$，速度调制不影响分布的概率性质。

#### 总结

时间调制通过引入相位变量 $z$ 实现动作速度的调整。$z(t)$ 单调递增，从 0 到 1，基函数从 $\phi_t$ 改为 $\phi(z_t)$，导数变为 $\dot{\phi}_t = \phi'(z_t) \dot{z}_t$。通过改变 $\dot{z}_t$，可以控制动作快慢，适用于 ProMPs 的概率框架。

### **节奏性与基于冲程的运动**

基函数的选择取决于运动的类型，运动可以是节奏性（rhythmic）或基于冲程（stroke-based）的。对于基于冲程的运动，我们使用高斯基函数 $b_{G_i}$；对于节奏性运动，我们使用 Von-Mises 基函数 $b_{VM_i}$，以在相位变量 $z$ 上建模周期性，即：

$$
b_{G_i}(z) = \exp\left(-\frac{(z_t - c_i)^2}{2h}\right), \quad b_{VM_i}(z) = \exp\left(\frac{\cos(2\pi(z_t - c_i))}{h}\right),
$$

其中 $h$ 定义基函数的宽度，$c_i$ 是第 $i$ 个基函数的中心。我们对`基函数进行归一化`，定义为 $\phi_i(z_t) = b_i(z) / \sum_j b_j(z)$。

#### 解释说明

**啥是节奏性和基于冲程的运动？**

- 节奏性运动（rhythmic）：像钟摆一样，反复循环的动作，比如机器人不停地挥手打招呼。
- 基于冲程的运动（stroke-based）：一次性完成的动作，比如挥拍击球，动作有明确的起点和终点。  
  **为啥要用不同的基函数？**  
  基函数是用来“拼”出动作轨迹的，像积木块。不同类型的动作需要不同形状的积木：
- **高斯基函数 $b_{G_i}$**：适合一次性动作，形状像一个“钟形曲线”，集中在某个点附近（中心 $c_i$），用来表示动作的局部特征，比如击球时的快速挥动。
- **Von-Mises 基函数 $b_{VM_i}$**：适合循环动作，形状是周期性的（像正弦波），能表示动作的重复性，比如挥手时的周期性摆动。  
  **参数啥意思？**
- $h$ 控制积木块的“宽度”：$h$ 大，积木块更宽，覆盖范围大；$h$ 小，积木块更窄，覆盖范围小。
- $c_i$ 是积木块的“中心”，决定它在动作进度条 $z$ 上的位置。  
  **为啥要归一化？**  
  归一化是为了让所有积木块的“总高度”加起来是 1，这样拼出来的轨迹不会太大或太小，保持稳定。归一化后的基函数 $\phi_i(z_t)$ 就像是“比例”，告诉你每个积木块占多少份。

#### 公式推导与细节

让我们一步步分析基函数的定义和归一化：

1. **高斯基函数 $b_{G_i}(z)$**：

- 定义：

  $$
  b_{G_i}(z) = \exp\left(-\frac{(z_t - c_i)^2}{2h}\right)
  $$

- **特点**：
  - 这是一个高斯函数，形状像一个钟形曲线。
  - $z_t$ 是相位变量（从 0 到 1），$c_i$ 是中心（例如 $c_i = 0.5$ 表示动作中间）。
  - $h$ 控制宽度，类似高斯分布的方差（但没有 $2\pi$ 项，不是概率密度）。
  - 当 $z_t = c_i$ 时，$b_{G_i}(z) = 1$（最大值）；$z_t$ 远离 $c_i$ 时，$b_{G_i}(z)$ 接近 0。
- **用途**：适合基于冲程的动作，因为动作通常在某些关键点（比如击球点）有高峰，其他地方平滑衰减。

2. **Von-Mises 基函数 $b_{VM_i}(z)$**：

- 定义：

  $$
  b_{VM_i}(z) = \exp\left(\frac{\cos(2\pi(z_t - c_i))}{h}\right)
  $$

- **特点**：
  - Von-Mises 分布是圆形数据的概率分布，这里用来建模周期性。
  - $\cos(2\pi(z_t - c_i))$ 是一个周期函数，周期为 1（因为 $z_t$ 从 0 到 1）。
  - 当 $z_t - c_i = 0$（或整数），$\cos = 1$，$b_{VM_i}(z)$ 最大；当 $z_t - c_i = 0.5$（或半整数），$\cos = -1$，$b_{VM_i}(z)$ 最小。
  - $h$ 控制“集中度”：$h$ 小，$\cos$ 的影响被放大，基函数更“尖锐”；$h$ 大，基函数更平滑。
- **用途**：适合节奏性动作，因为动作是周期性重复的，Von-Mises 基函数能捕捉这种规律。

3. **归一化**：

- 归一化后的基函数：

  $$
  \phi_i(z_t) = \frac{b_i(z)}{\sum_j b_j(z)}
  $$

- **原因**：
  - $b_i(z)$ 是原始基函数（高斯或 Von-Mises），但它们的总和 $\sum_j b_j(z)$ 可能不是 1。
  - 归一化确保 $\sum_i \phi_i(z_t) = 1$，这样基函数的加权和（轨迹 $y_t = \Phi_t^T w$）不会因基函数幅值过大或过小而失控。

> [!example]+ 归一化例子
> 假设我们处理一个基于冲程的运动，使用高斯基函数，有 3 个基函数（$i=1,2,3$），参数为：
>
> - $h = 0.01$（宽度）。
> - 中心：$c_1 = 0.3$，$c_2 = 0.5$，$c_3 = 0.7$。
> - 当前相位：$z_t = 0.5$。
>
> **计算基函数值**：  
> 高斯基函数定义为：
>
> $$
> b_{G_i}(z) = \exp\left(-\frac{(z_t - c_i)^2}{2h}\right)
> $$
>
> - $b_1(z) = \exp\left(-\frac{(0.5 - 0.3)^2}{2 \cdot 0.01}\right) = \exp\left(-\frac{0.04}{0.02}\right) = \exp(-2) \approx 0.1353$
> - $b_2(z) = \exp\left(-\frac{(0.5 - 0.5)^2}{2 \cdot 0.01}\right) = \exp(0) = 1.0$
> - $b_3(z) = \exp\left(-\frac{(0.5 - 0.7)^2}{2 \cdot 0.01}\right) = \exp\left(-\frac{0.04}{0.02}\right) = \exp(-2) \approx 0.1353$
>
> **归一化**：
>
> - 总和：$\sum_j b_j(z) = 0.1353 + 1.0 + 0.1353 = 1.2706$
> - 归一化后的基函数：
>   $$
>   \phi_1(z_t) = \frac{b_1(z)}{\sum_j b_j(z)} = \frac{0.1353}{1.2706} \approx 0.1065
>   $$
>
> $$
> \phi_2(z_t) = \frac{b_2(z)}{\sum_j b_j(z)} = \frac{1.0}{1.2706} \approx 0.7870
> $$
>
> $$
> \phi_3(z_t) = \frac{b_3(z)}{\sum_j b_j(z)} = \frac{0.1353}{1.2706} \approx 0.1065
> $$
>
> - 验证：$\phi_1 + \phi_2 + \phi_3 = 0.1065 + 0.7870 + 0.1065 = 1.0$，归一化成功。

**在 ProMPs 中的意义**：

- 基函数的选择（高斯或 Von-Mises）让 ProMPs 能同时处理一次性动作（冲程）和循环动作（节奏性），增强了框架的通用性。
- 相位变量 $z$ 结合基函数 $\phi(z_t)$，使得动作可以灵活调整速度（见时间调制部分）。
- 归一化后的基函数 $\phi_i(z_t)$ 确保轨迹表示 $y_t = \Phi_t^T w$ 的稳定性，适合概率建模。

#### 总结

ProMPs 通过选择不同基函数支持节奏性和基于冲程的运动：高斯基函数 $b_{G_i}$ 用于冲程动作，Von-Mises 基函数 $b_{VM_i}$ 用于节奏动作，$h$ 和 $c_i$ 控制宽度和中心。归一化后的基函数 $\phi_i(z_t)$ 确保总和为 1，保持轨迹表示的稳定性。

### **关节间耦合编码**

到目前为止，我们假设每个自由度是独立建模的。然而，对于许多任务，我们需要协调关节的运动。一种常见的实现协调的方法是通过相位变量 $z_t$ 耦合轨迹分布的均值[18]。然而，通常还希望编码更高阶的耦合信息，例如时间点 $t$ 处关节之间的协方差。因此，我们将模型扩展到多维。对于每个维度 $i$，我们维护一个参数向量 $w_i$，并定义组合权重向量 $w$ 为 $w = \left[w_1^T, \ldots, w_n^T\right]^T$。基矩阵 $\Phi_t$ 现在扩展为一个块对角矩阵，包含每个维度的基函数及其导数。观测向量 $y_t$ 包含所有关节的角度和速度。时间 $t$ 处观测 $y$ 的概率由下式给出：

$$
p(y_t \mid w) = \mathcal{N}\left( \begin{bmatrix} y_{1,t} \\ \vdots \\ y_{d,t} \end{bmatrix} \Bigg| \begin{bmatrix} \Phi_t^T & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & \Phi_t^T \end{bmatrix} w, \Sigma_y \right) = \mathcal{N}(y_t \mid \Psi_t w, \Sigma_y)
$$

其中 $y_{i,t} = \left[q_{i,t}, \dot{q}_{i,t}\right]^T$ 表示第 $i$ 个关节的角度和速度。我们现在对组合参数向量 $w$ 维护一个分布 $p(w; \theta)$。利用这一分布，我们还可以捕捉关节之间的协方差。

#### 解释说明

**为啥要考虑关节间的耦合？**  
之前我们假设机器人的每个关节（比如手臂的三个关节）是独立动的，各干各的。但实际中，关节得“ teamwork”：比如挥拍击球时，肩、肘、腕得一起配合，不能乱动。

- **怎么耦合？**
  - 之前用相位 $z_t$ 让所有关节的动作“节奏一致”（均值耦合），就像大家跟着同一个节拍跳舞。
  - 但光有节拍不够，还得知道关节之间怎么“配合”（协方差），比如肩动得多，肘是不是也得动得多。
- **咋解决？**
  - 把每个关节的参数 $w_i$ 拼成一个大向量 $w$，这样就能统一管理。
  - 基函数矩阵 $\Phi_t$ 变成一个大矩阵（块对角形式），每个关节用自己的基函数，但组合在一起。
  - 状态 $y_t$ 也变成一个大向量，包含所有关节的位置和速度。
  - 最后，$w$ 的分布 $p(w; \theta)$ 能捕捉关节间的关系（协方差），比如“肩动 1 度，肘可能动 0.5 度”。

#### 公式推导与细节

让我们一步步分析关节耦合的实现：

1. **多维扩展**：

- 之前：每个关节（自由度）独立建模，$y_t = \left[q_t, \dot{q}_t\right]^T$ 只表示一个关节的状态，$w$ 是单个关节的权重向量。
- 现在：假设有 $d$ 个关节，$y_t$ 扩展为所有关节的状态：

  $$
  y_t = \begin{bmatrix} y_{1,t} \\ \vdots \\ y_{d,t} \end{bmatrix}, \quad y_{i,t} = \left[q_{i,t}, \dot{q}_{i,t}\right]^T
  $$

- 权重向量 $w$ 也扩展为所有关节的权重组合：

  $$
  w = \begin{bmatrix} w_1^T \\ \vdots \\ w_d^T \end{bmatrix}
  $$

  - 其中 $w_i$ 是第 $i$ 个关节的权重向量（维度取决于基函数数量）。

2. **基矩阵 $\Phi_t$ 的扩展**：

- 之前：$\Phi_t = \left[\phi_t, \dot{\phi}_t\right]$ 是一个关节的基函数矩阵。
- 现在：$\Phi_t$ 变成一个块对角矩阵，包含 $d$ 个关节的基函数：

  $$
  \Psi_t = \begin{bmatrix} \Phi_t^T & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & \Phi_t^T \end{bmatrix}
  $$

- **解释**：
  - $\Psi_t$ 的维度是 $(2d) \times (n \cdot d)$，其中 $2d$ 是 $y_t$ 的维度（$d$ 个关节，每个关节有位置和速度），$n$ 是每个关节的基函数数量。
  - 块对角结构意味着每个关节用自己的基函数 $\Phi_t$，但整体组合成一个大矩阵。
  - 这里假设所有关节用相同的基函数形式（比如都用高斯基函数），但实际中可以不同。

3. **概率分布**：

- 给定 $w$，$y_t$ 的概率分布为：

  $$
  p(y_t \mid w) = \mathcal{N}(y_t \mid \Psi_t w, \Sigma_y)
  $$

- **均值**：$\Psi_t w$ 是所有关节状态的预测值：

  - 第 $i$ 个关节的预测值是 $\Phi_t^T w_i$。
  - 整体为：
    $$
    \Psi_t w = \begin{bmatrix} \Phi_t^T w_1 \\ \vdots \\ \Phi_t^T w_d \end{bmatrix}
    $$

- **协方差** $\Sigma_y$：
  - $\Sigma_y$ 是一个 $2d \times 2d$ 的矩阵，描述所有关节状态之间的协方差。
  - 例如，$\Sigma_y$ 的 $(i,j)$ 块表示第 $i$ 个关节和第 $j$ 个关节之间的协方差。

4. **权重分布 $p(w; \theta)$**：

- 组合权重 $w$ 的分布：

  $$
  p(w; \theta) = \mathcal{N}(w \mid \mu_w, \Sigma_w)
  $$

- **协方差捕捉**：
  - $\Sigma_w$ 是 $w$ 的协方差矩阵，维度是 $(n \cdot d) \times (n \cdot d)$。
  - 因为 $w = \left[w_1^T, \ldots, w_d^T\right]^T$，$\Sigma_w$ 的分块形式包含了 $w_i$ 和 $w_j$ 之间的协方差，从而间接捕捉了关节间的关系。
  - 例如，$\Sigma_w$ 的 $(i,j)$ 块表示第 $i$ 个关节和第 $j$ 个关节的权重之间的协方差，这种关系会通过 $y_t = \Psi_t w + \epsilon_y$ 传递到关节状态 $y_t$。

**在 ProMPs 中的意义**：

- 通过扩展到多维，ProMPs 能建模多个关节的协调运动。
- 相位变量 $z_t$ 提供均值上的耦合（所有关节按同一节奏），而 $p(w; \theta)$ 的协方差 $\Sigma_w$ 提供更高阶的耦合（关节间的相关性）。
- 这种方法让 ProMPs 更适合复杂任务，比如需要多个关节协同的挥拍动作。

#### 总结

ProMPs 通过扩展模型到多维，实现了关节间的耦合编码。组合权重 $w$ 和块对角基矩阵 $\Psi_t$ 统一表示所有关节的状态，$p(w; \theta)$ 的协方 variance $\Sigma_w$ 捕捉关节间的协方差，支持协调运动建模。

### **从演示中学习**

运动原语（MP）表示的一个关键要求是，单个原语的参数能够从演示中轻松获取。为了便于参数估计，我们假设参数 $w$ 上的分布为高斯分布，即 $p(w; \theta) = \mathcal{N}(w \mid \mu_w, \Sigma_w)$。因此，时间步 $t$ 处状态的分布 $p(y_t \mid \theta)$ 由下式给出：

$$
p(y_t; \theta) = \int \mathcal{N}(y_t \mid \Psi_t^T w, \Sigma_y) \mathcal{N}(w \mid \mu_w, \Sigma_w) \mathrm{d}w = \mathcal{N}(y_t \mid \Psi_t^T \mu_w, \Psi_t^T \Sigma_w \Psi_t + \Sigma_y),
$$

从而，我们可以轻松计算任意时间点 $t$ 的均值和方差。由于 ProMP 表示执行基本运动的多种方式，我们需要多个演示来学习 $p(w; \theta)$。参数 $\theta = \{\mu_w, \Sigma_w\}$ 可以通过最大似然估计从多个演示中学习，例如，使用针对高斯分布的层次贝叶斯模型（HBMs）的期望最大化（EM）算法[19]。

#### 解释说明

**从演示中学习是干嘛的？**  
机器人要学会动作（比如挥拍），得看人类怎么做（演示）。ProMPs 的目标是从这些演示里学出参数，让机器人能模仿动作，还能有点变化（多样性）。

- **咋学？**
  - 假设权重 $w$ 服从高斯分布 $p(w; \theta)$，$\theta$ 包含均值 $\mu_w$ 和协方差 $\Sigma_w$。
  - 有了 $w$ 的分布，我们就能算出动作状态 $y_t$（关节位置和速度）的分布：均值是 $\Psi_t^T \mu_w$，协方差是 $\Psi_t^T \Sigma_w \Psi_t + \Sigma_y$。
  - 看了一堆演示（多次挥拍），用这些数据去调 $\mu_w$ 和 $\Sigma_w$，让模型尽可能符合演示。
- **咋调参数？**
  - 用最大似然估计（MLE）：找一组 $\mu_w$ 和 $\Sigma_w$，让演示数据的概率最大。
  - 因为算起来复杂，用期望最大化（EM）算法来一步步优化。

#### 公式推导与细节

让我们一步步推导状态分布 $p(y_t; \theta)$ 和参数学习的过程：

1. **状态分布 $p(y_t; \theta)$ 的推导**：

- 已知：
  - $p(y_t \mid w) = \mathcal{N}(y_t \mid \Psi_t^T w, \Sigma_y)$
  - $p(w; \theta) = \mathcal{N}(w \mid \mu_w, \Sigma_w)$
- 目标：计算 $p(y_t; \theta) = \int p(y_t \mid w) p(w; \theta) \mathrm{d}w$。
- **均值**：

  - $y_t$ 的期望：
    $$
    \mathbb{E}[y_t] = \mathbb{E}[\Psi_t^T w + \epsilon_y] = \Psi_t^T \mathbb{E}[w] = \Psi_t^T \mu_w
    $$

- **协方差**：

  - $y_t = \Psi_t^T w + \epsilon_y$，$w$ 和 $\epsilon_y$ 独立。
  - 协方差为：
    $$
    \text{Cov}(y_t) = \text{Cov}(\Psi_t^T w) + \text{Cov}(\epsilon_y) = \Psi_t^T \text{Cov}(w) \Psi_t + \Sigma_y = \Psi_t^T \Sigma_w \Psi_t + \Sigma_y
    $$

- **结果**：
  - 因为 $p(y_t \mid w)$ 和 $p(w; \theta)$ 都是高斯分布，它们的卷积（积分）仍然是高斯分布：
    $$
    p(y_t; \theta) = \mathcal{N}(y_t \mid \Psi_t^T \mu_w, \Psi_t^T \Sigma_w \Psi_t + \Sigma_y)
    $$

2. **参数学习（最大似然估计）**：

- **数据**：给定 $N$ 次演示 $\{\tau^{(i)}\}_{i=1}^N$，每条演示 $\tau^{(i)} = \{y_t^{(i)}\}_{t=0}^T$。
- **似然函数**：

  - 假设各时间步独立，单条演示的似然为：

    $$
    p(\tau^{(i)}; \theta) = \prod_{t=0}^T p(y_t^{(i)}; \theta)
    $$

  - 总似然（所有演示）：
    $$
    p(\{\tau^{(i)}\}; \theta) = \prod_{i=1}^N p(\tau^{(i)}; \theta) = \prod_{i=1}^N \prod_{t=0}^T p(y_t^{(i)}; \theta)
    $$

- **目标**：最大化似然，估计 $\theta = \{\mu_w, \Sigma_w\}$：

  $$
  \theta^* = \arg\max_{\theta} \prod_{i=1}^N \prod_{t=0}^T \mathcal{N}(y_t^{(i)} \mid \Psi_t^T \mu_w, \Psi_t^T \Sigma_w \Psi_t + \Sigma_y)
  $$

- **对数似然**：

  - 取对数简化计算：

    $$
    \log p(\{\tau^{(i)}\}; \theta) = \sum_{i=1}^N \sum_{t=0}^T \log \mathcal{N}(y_t^{(i)} \mid \Psi_t^T \mu_w, \Psi_t^T \Sigma_w \Psi_t + \Sigma_y)
    $$

  - 展开高斯分布的对数形式：

    $$
    \log \mathcal{N}(y_t^{(i)} \mid \mu, \Sigma) = -\frac{1}{2}(y_t^{(i)} - \mu)^T \Sigma^{-1} (y_t^{(i)} - \mu) - \frac{1}{2} \log |\Sigma| - \frac{k}{2} \log 2\pi
    $$

  - 其中 $\mu = \Psi_t^T \mu_w$，$\Sigma = \Psi_t^T \Sigma_w \Psi_t + \Sigma_y$，$k$ 是 $y_t$ 的维度。

3. **期望最大化（EM）算法**：

- **问题**：直接优化对数似然很复杂，因为 $w$ 是隐藏变量，似然中涉及积分（边缘化 $w$）。
- **EM 算法**：
  - **E 步**：计算给定当前参数 $\theta^{\text{old}}$ 的后验 $p(w \mid \tau^{(i)}, \theta^{\text{old}})$，并计算期望对数似然（关于 $w$ 的期望）。
  - **M 步**：更新参数 $\theta$，最大化期望对数似然，得到 $\theta^{\text{new}}$。
  - 迭代直到收敛。
- **在 ProMPs 中**：
  - 因为 $p(w; \theta)$ 和 $p(y_t \mid w)$ 都是高斯分布，EM 算法可以通过解析形式更新 $\mu_w$ 和 $\Sigma_w$（具体更新公式较复杂，通常涉及卡尔曼滤波或直接优化）。
  - 参考文献[19]可能提供了具体的 EM 算法实现。

**通俗解释**：  
EM 算法就像“边猜边调”。你先猜一组 $\mu_w$ 和 $\Sigma_w$（初始参数），用它们算出每条演示的“可能权重” $w$（E 步），然后根据这些权重调整 $\mu_w$ 和 $\Sigma_w$（M 步），让模型更符合数据，反复调整直到满意。

**在 ProMPs 中的意义**：

- 从演示中学习让 ProMPs 能模仿人类动作，同时捕捉动作的多样性（通过 $\Sigma_w$）。
- 高斯分布假设简化了计算，EM 算法高效估计参数，适合层次贝叶斯模型。

#### 总结

ProMPs 通过假设 $w$ 服从高斯分布，从演示中学习参数 $\theta = \{\mu_w, \Sigma_w\}$。状态分布 $p(y_t; \theta)$ 是高斯分布，均值和协方差可直接计算。参数学习使用最大似然估计，结合 EM 算法，高效处理隐藏变量 $w$ 的边缘化。

## **2.2 新的概率操作符用于运动原语**

ProMPs 允许使用概率论中的新操作符，例如通过条件化（conditioning）来调制轨迹，以及通过分布乘积来共同激活多个运动原语（MPs）。我们将在通用框架中描述这两种操作符，并随后讨论在我们选择的高斯分布 $p(w; \theta)$ 下的具体实现。

### **通过条件化调制途径点、最终位置或速度**

调制途径点和最终位置是任何运动原语框架的重要属性，以便运动原语能够适应新情境。在我们的概率公式中，这类操作可以通过条件化运动原语以在时间 $t$ 达到某个特定状态 $y_t^*$ 来描述。条件化通过向概率模型添加一个期望观测 $x_t = [y_t^*, \Sigma_y^*]$ 并应用贝叶斯定理来执行，即：

$$
p(w \mid x_t^*) \propto \mathcal{N}(y_t^* \mid \Psi_t^T w, \Sigma_y^*) p(w).
$$

状态向量 $y_t^*$ 表示时间 $t$ 处的期望位置和速度向量，$\Sigma_y^*$ 描述期望观测的精度。我们也可以对 $y_t^*$ 的任何子集进行条件化。例如，通过指定第一个关节的期望关节位置 $q_1$，轨迹分布会自动推断其他关节的最可能位置。  
对于高斯轨迹分布，条件分布 $p(w \mid x_t^*)$ 仍然是高斯分布，其均值和方差为：

$$
\mu_w^{\text{[new]}} = \mu_w + \Sigma_w \Psi_t \left(\Sigma_y^* + \Psi_t^T \Sigma_w \Psi_t\right)^{-1} (y_t^* - \Psi_t^T \mu_w),
$$

$$
\Sigma_w^{\text{[new]}} = \Sigma_w - \Sigma_w \Psi_t \left(\Sigma_y^* + \Psi_t^T \Sigma_w \Psi_t\right)^{-1} \Psi_t^T \Sigma_w.
$$

通过条件化将 ProMP 调整到不同目标状态的过程也在图 1 (a) 中展示。我们可以看到，尽管通过条件化调制了 ProMP，ProMP 仍然保持在原始分布内，因此这种调制也是从原始演示中学习得到的。当前方法（如 DMPs）的调制策略并未展现这种有益效果[18]。

#### 解释说明

**新的概率操作符是干嘛的？**  
ProMPs 给运动原语加了两个新工具：条件化（调轨迹）和分布乘积（组合动作）。这里先讲条件化。

- **条件化是啥？**  
  条件化就像是“给机器人加个要求”：你告诉它在某个时间 $t$ 要达到某个状态（比如挥拍时某个点的位置和速度），它会调整自己的动作来满足这个要求。
- **咋调？**
  - 你设定一个目标状态 $y_t^*$（比如“时间 $t$ 时第一个关节位置是 $q_1$”），还给个“误差范围” $\Sigma_y^*$（表示目标的精度）。
  - 用贝叶斯定理，算出新的权重分布 $p(w \mid x_t^*)$，让动作尽量满足这个目标。
- **高斯分布咋算？**
  - 因为 $w$ 和 $y_t$ 都是高斯分布，条件化后的 $w$ 还是高斯分布，均值和方差有公式（上面给的）。
  - 新均值 $\mu_w^{\text{[new]}}$ 是“原始均值 + 调整项”，调整项根据目标 $y_t^*$ 和误差 $\Sigma_y^*$ 算出来。
  - 新方差 $\Sigma_w^{\text{[new]}}$ 比原来小，因为加了条件后不确定性减少了。
- **有啥好处？**
  - 你可以只指定一部分目标（比如第一个关节的位置），ProMP 会自动推算其他关节的最优位置（因为关节间有耦合）。
  - 调出来的动作还在原始分布里（从演示学来的范围），不像其他方法（比如 DMPs）可能会偏离。

#### 公式推导与细节

让我们一步步推导条件化分布 $p(w \mid x_t^*)$ 的均值和方差：

1. **条件化定义**：

- 已知：
  - 先验：$p(w) = \mathcal{N}(w \mid \mu_w, \Sigma_w)$
  - 似然：$p(y_t^* \mid w) = \mathcal{N}(y_t^* \mid \Psi_t^T w, \Sigma_y^*)$
- 目标：计算后验 $p(w \mid x_t^*) = p(w \mid y_t^*)$。
- 根据贝叶斯定理：
  $$
  p(w \mid y_t^*) \propto p(y_t^* \mid w) p(w)
  $$

2. **高斯分布的条件化**：

- 因为 $p(y_t^* \mid w)$ 和 $p(w)$ 都是高斯分布，后验 $p(w \mid y_t^*)$ 也是高斯分布，形式为：

  $$
  p(w \mid y_t^*) = \mathcal{N}(w \mid \mu_w^{\text{[new]}}, \Sigma_w^{\text{[new]}})
  $$

- **均值和方差公式**：
  - 考虑一般形式：假设 $x \sim \mathcal{N}(\mu_x, \Sigma_x)$，$y \sim \mathcal{N}(Ax, \Sigma_y)$，则 $p(x \mid y)$ 是高斯分布。
  - 在 ProMPs 中：
    - $x = w$，$\mu_x = \mu_w$，$\Sigma_x = \Sigma_w$。
    - $y = y_t^*$，$A = \Psi_t^T$，$\Sigma_y = \Sigma_y^*$。
  - 条件高斯分布的均值和方差公式为：
    $$
    \mu_w^{\text{[new]}} = \mu_w + \Sigma_w A^T (A \Sigma_w A^T + \Sigma_y)^{-1} (y - A \mu_w)
    $$

$$
\Sigma_w^{\text{[new]}} = \Sigma_w - \Sigma_w A^T (A \Sigma_w A^T + \Sigma_y)^{-1} A \Sigma_w
$$

- 代入 $A = \Psi_t^T$，$y = y_t^*$，$\Sigma_y = \Sigma_y^*$：
  $$
  \mu_w^{\text{[new]}} = \mu_w + \Sigma_w \Psi_t \left(\Psi_t^T \Sigma_w \Psi_t + \Sigma_y^*\right)^{-1} (y_t^* - \Psi_t^T \mu_w)
  $$

$$
\Sigma_w^{\text{[new]}} = \Sigma_w - \Sigma_w \Psi_t \left(\Psi_t^T \Sigma_w \Psi_t + \Sigma_y^*\right)^{-1} \Psi_t^T \Sigma_w
$$

**在 ProMPs 中的意义**：

- 条件化让 ProMPs 能灵活调整轨迹，比如指定某个时间点的关节位置，其他关节会自动调整，保持协调。
- 条件化后的轨迹仍在原始分布内（从演示中学到的范围），比传统方法（如 DMPs）更自然。

#### 总结

ProMPs 通过条件化实现途径点和最终位置的调制，用贝叶斯定理计算后验 $p(w \mid y_t^*)$。在高斯分布下，后验均值和方差有解析形式，支持灵活且自然的轨迹调制。

### **运动原语的组合与混合**

另一个有益的概率操作是持续组合和混合不同的运动原语（MPs）以形成单一运动。假设我们维护一组 $i$ 个不同的原语，希望将它们组合。我们可以通过分布的乘积来共同激活它们，即 $p_{\text{new}}(\tau) \propto \prod_i p_i(\tau)^{\alpha[i]}$，其中 $\alpha[i] \in [0, 1]$ 表示第 $i$ 个原语的激活因子。这种乘积捕捉了活跃原语的重叠区域，即所有原语具有高概率质量的轨迹空间部分。  
然而，我们还希望能够调制原语的激活，例如，持续混合运动执行，从一个原语平滑过渡到下一个原语。因此，我们将轨迹分解为单个时间步，并使用时变激活函数 $\alpha[i]_t$，即：

$$
p^*(\tau) \propto \prod_t \prod_i p_i(y_t)^{\alpha[i]_t}, \quad p_i(y_t) = \int p_i(y_t \mid w[i]) p_i(w[i]) \mathrm{d}w[i].
$$

对于高斯分布 $p_i(y_t) = \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)$，结果分布 $p^*(y_t)$ 仍然是高斯分布，其方差和均值为：

$$
\Sigma^*_t = \left( \sum_i \left( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \right)^{-1}, \quad \mu^*_t = (\Sigma^*_t)^{-1} \left( \sum_i \left( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \mu[i]_t \right).
$$

这两项及其导数是获取随机反馈控制器所需的，最终用于控制机器人。我们在图 1 (b) 中展示了两个 ProMPs 的共同激活，在图 1 (c) 中展示了两个 ProMPs 的混合。

#### 解释说明

**组合和混合是干嘛的？**  
ProMPs 能把多个动作（运动原语）“拼”在一起，或者“平滑切换”，让机器人动作更灵活。

- **组合（共同激活）**：
  - 想象你有两个动作：一个是“挥拍击球”，一个是“抬手打招呼”。你想让机器人同时做这两个动作（比如一边挥拍一边抬手）。
  - 方法是把两个动作的概率分布相乘：$p_{\text{new}}(\tau) \propto \prod_i p_i(\tau)^{\alpha[i]}$，$\alpha[i]$ 控制每个动作的“参与度”（比如挥拍多点，打招呼少点）。
  - 结果是找到两个动作的“重叠部分”，比如“手臂既能挥拍又能打招呼”的轨迹。
- **混合（平滑过渡）**：
  - 有时候你想让动作“切换”，比如先挥拍击球，然后变成打招呼。
  - 方法是让参与度 $\alpha[i]_t$ 随时间变化：一开始 $\alpha[1]_t$ 大（挥拍为主），后来 $\alpha[2]_t$ 大（打招呼为主）。
  - 结果是动作平滑过渡，先像挥拍，后来像打招呼。
- **高斯分布咋算？**
  - 每个动作的分布是高斯分布（均值 $\mu[i]_t$，方差 $\Sigma[i]_t$）。
  - 组合后的分布还是高斯分布，均值和方差有公式：均值是“加权平均”，方差是“加权调和平均”。
- **有啥用？**
  - 这些分布最后用来生成控制信号，告诉机器人怎么动。图 1 (b) 和 1 (c) 展示了效果：组合能同时满足两个动作的要求，混合能平滑切换。

#### 公式推导与细节

让我们一步步推导组合和混合的分布：

1. **共同激活（Combination）**：

- 分布乘积：

  $$
  p_{\text{new}}(\tau) \propto \prod_i p_i(\tau)^{\alpha[i]}
  $$

- **分解到时间步**：

  - 轨迹 $\tau = \{y_t\}$，假设各时间步独立：

    $$
    p_i(\tau) = \prod_t p_i(y_t)
    $$

  - 因此：
    $$
    p_{\text{new}}(\tau) \propto \prod_i \left( \prod_t p_i(y_t) \right)^{\alpha[i]} = \prod_t \prod_i p_i(y_t)^{\alpha[i]}
    $$

2. **混合（Blending）**：

- 时变激活：

  $$
  p^*(\tau) \propto \prod_t \prod_i p_i(y_t)^{\alpha[i]_t}
  $$

- 其中：

  $$
  p_i(y_t) = \int p_i(y_t \mid w[i]) p_i(w[i]) \mathrm{d}w[i] = \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)
  $$

  - $\mu[i]_t = \Psi_t^T \mu_{w[i]}$
  - $\Sigma[i]_t = \Psi_t^T \Sigma_{w[i]} \Psi_t + \Sigma_y$

3. **高斯分布的乘积**：

- 每个 $p_i(y_t)$ 是高斯分布：

  $$
  p_i(y_t) = \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)
  $$

- 考虑时间步 $t$，计算：

  $$
  p^*(y_t) \propto \prod_i p_i(y_t)^{\alpha[i]_t} = \prod_i \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)^{\alpha[i]_t}
  $$

- 高斯分布的指数形式：

  $$
  \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t) \propto \exp\left(-\frac{1}{2} (y_t - \mu[i]_t)^T \Sigma[i]_t^{-1} (y_t - \mu[i]_t)\right)
  $$

- 带指数 $\alpha[i]_t$：

  $$
  \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)^{\alpha[i]_t} \propto \exp\left(-\frac{\alpha[i]_t}{2} (y_t - \mu[i]_t)^T \Sigma[i]_t^{-1} (y_t - \mu[i]_t)\right)
  $$

- 这等价于一个新的高斯分布：

  $$
  \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)^{\alpha[i]_t} \propto \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t)
  $$

- 因此：

  $$
  p^*(y_t) \propto \prod_i \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t)
  $$

- **高斯分布乘积的结果**：

  - 多个高斯分布相乘仍是高斯分布：
    $$
    \prod_i \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t) \propto \mathcal{N}(y_t \mid \mu^*_t, \Sigma^*_t)
    $$

  > [!note]+ 推导过程
  >
  > **步骤 1：理解混合分布**
  >
  > - 混合分布 $p^*(y_t)$ 是多个高斯分布的加权乘积：
  >   $$
  >   p^*(y_t) \propto \prod_i \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)^{\alpha[i]_t}
  >   $$
  > - 已证明：
  >   $$
  >   \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t)^{\alpha[i]_t} \propto \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t)
  >   $$
  > - 因此，混合分布可以写为：
  >   $$
  >   p^*(y_t) \propto \prod_i \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t)
  >   $$
  > - 多个高斯分布的乘积仍为高斯分布，形式为 $\mathcal{N}(y_t \mid \mu_t^*, \Sigma_t^*)$，我们需要求 $\mu_t^*$ 和 $\Sigma_t^*$。
  >
  > **步骤 2：高斯分布的概率密度**
  >
  > - 每个高斯分布的概率密度为：
  >   $$
  >   \mathcal{N}(y_t \mid \mu[i]_t, \Sigma[i]_t / \alpha[i]_t) = \frac{1}{(2\pi)^{k/2} |\Sigma[i]_t / \alpha[i]_t|^{1/2}} \exp\left(-\frac{1}{2} (y_t - \mu[i]_t)^T \left( \frac{\Sigma[i]_t}{\alpha[i]_t} \right)^{-1} (y_t - \mu[i]_t)\right)
  >   $$
  > - 其中 $k$ 是 $y_t$ 的维度，$\left( \frac{\Sigma[i]_t}{\alpha[i]_t} \right)^{-1} = \alpha[i]_t \Sigma[i]_t^{-1}$，$|\Sigma[i]_t / \alpha[i]_t| = |\Sigma[i]_t| / (\alpha[i]_t)^k$。
  > - 指数部分：
  >   $$
  >   \exp\left(-\frac{1}{2} (y_t - \mu[i]_t)^T \left( \alpha[i]_t \Sigma[i]_t^{-1} \right) (y_t - \mu[i]_t)\right)
  >   $$
  > - 混合分布的指数部分：
  >   $$
  >   p^*(y_t) \propto \prod_i \exp\left(-\frac{\alpha[i]_t}{2} (y_t - \mu[i]_t)^T \Sigma[i]_t^{-1} (y_t - \mu[i]_t)\right)
  >   $$
  > - 合并指数：
  >   $$
  >   p^*(y_t) \propto \exp\left(-\frac{1}{2} \sum_i \alpha[i]_t (y_t - \mu[i]_t)^T \Sigma[i]_t^{-1} (y_t - \mu[i]_t)\right)
  >   $$
  >
  > **步骤 3：展开指数项并匹配高斯形式**
  >
  > - 目标分布是高斯分布 $\mathcal{N}(y_t \mid \mu_t^*, \Sigma_t^*)$，其指数部分为：
  >   $$
  >   \exp\left(-\frac{1}{2} (y_t - \mu_t^*)^T (\Sigma_t^*)^{-1} (y_t - \mu_t^*)\right)
  >   $$
  > - 展开混合分布的指数：
  >   $$
  >   \sum_i \alpha[i]_t (y_t - \mu[i]_t)^T \Sigma[i]_t^{-1} (y_t - \mu[i]_t) = \sum_i \alpha[i]_t \left( y_t^T \Sigma[i]_t^{-1} y_t - 2 y_t^T \Sigma[i]_t^{-1} \mu[i]_t + \mu[i]_t^T \Sigma[i]_t^{-1} \mu[i]_t \right)
  >   $$
  > - 合并项：
  > - 二次项：
  >   $$
  >   y_t^T \left( \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \right) y_t
  >   $$
  > - 线性项：
  >   $$
  >   -2 y_t^T \left( \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \mu[i]_t \right)
  >   $$
  > - 常数项：
  >   $$
  >   \sum_i \alpha[i]_t \mu[i]_t^T \Sigma[i]_t^{-1} \mu[i]_t
  >   $$
  > - 展开目标分布的指数：
  >   $$
  >   (y_t - \mu_t^*)^T (\Sigma_t^*)^{-1} (y_t - \mu_t^*) = y_t^T (\Sigma_t^*)^{-1} y_t - 2 y_t^T (\Sigma_t^*)^{-1} \mu_t^* + \mu_t^{*T} (\Sigma_t^*)^{-1} \mu_t^*
  >   $$
  > - 比较二次项：
  > - 混合分布：$y_t^T \left( \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \right) y_t$
  > - 目标分布：$y_t^T (\Sigma_t^*)^{-1} y_t$
  > - 因此：
  >   $$
  >   (\Sigma_t^*)^{-1} = \sum_i \alpha[i]_t \Sigma[i]_t^{-1}
  >   $$
  > - 取逆：
  > - 因为 $\left( \Sigma[i]_t / \alpha[i]_t \right)^{-1} = \alpha[i]_t \Sigma[i]_t^{-1}$，所以：
  >   $$
  >   \sum_i \left( \Sigma[i]_t / \alpha[i]_t \right)^{-1} = \sum_i \alpha[i]_t \Sigma[i]_t^{-1}
  >   $$
  > - 故：
  >   $$
  >   \Sigma_t^* = \left( \sum_i \left( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \right)^{-1}
  >   $$
  >
  > **步骤 4：推导均值 $\mu_t^*$**
  >
  > - 比较线性项：
  > - 混合分布：$-2 y_t^T \left ( \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \mu[i]_t \right)$
  > - 目标分布：$-2 y_t^T (\Sigma_t^*)^{-1} \mu_t^*$
  > - 代入 $(\Sigma_t^*)^{-1}$：
  > - $(\Sigma_t^*)^{-1} = \sum_i \alpha[i]_t \Sigma[i]_t^{-1}$
  > - 因此：
  >   $$
  >   (\Sigma_t^*)^{-1} \mu_t^* = \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \mu[i]_t
  >   $$
  > - 两边左乘 $\Sigma_t^*$：
  > - $\mu_t^* = \Sigma_t^* \left ( \sum_i \alpha[i]_t \Sigma[i]_t^{-1} \mu[i]_t \right)$
  > - 因为 $\alpha[i]_t \Sigma[i]_t^{-1} = \left ( \Sigma[i]_t / \alpha[i]_t \right)^{-1}$，所以：
  >   $$
  >   \mu_t^* = \Sigma_t^* \left ( \sum_i \left ( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \mu[i]_t \right)
  >   $$
  >
  > **2. 举个例子**
  >
  > - **场景**：
  > - 两个原语，$t=0.5$，$\alpha[1]_{0.5} = 0.7$，$\alpha[2]_{0.5} = 0.3$。
  > - $\mu[1]_{0.5} = [1, 0]^T$，$\Sigma[1]_{0.5} = \begin{bmatrix} 0.1 & 0 \\ 0 & 0.1 \end{bmatrix}$
  > - $\mu[2]_{0.5} = [2, 0]^T$，$\Sigma[2]_{0.5} = \begin{bmatrix} 0.2 & 0 \\ 0 & 0.2 \end{bmatrix}$
  > - **方差 $\Sigma_t^*$**：
  > - $\left ( \Sigma[1]_{0.5} / \alpha[1]_{0.5} \right)^{-1} = (0.7 \cdot \begin{bmatrix} 0.1 & 0 \\ 0 & 0.1 \end{bmatrix})^{-1} = \begin{bmatrix} 14.29 & 0 \\ 0 & 14.29 \end{bmatrix}$
  > - $\left ( \Sigma[2]_{0.5} / \alpha[2]_{0.5} \right)^{-1} = (0.3 \cdot \begin{bmatrix} 0.2 & 0 \\ 0 & 0.2 \end{bmatrix})^{-1} = \begin{bmatrix} 16.67 & 0 \\ 0 & 16.67 \end{bmatrix}$
  > - $(\Sigma_t^*)^{-1} = \begin{bmatrix} 14.29 + 16.67 & 0 \\ 0 & 14.29 + 16.67 \end{bmatrix} = \begin{bmatrix} 30.96 & 0 \\ 0 & 30.96 \end{bmatrix}$
  > - $\Sigma_t^* = \begin{bmatrix} 1/30.96 & 0 \\ 0 & 1/30.96 \end{bmatrix} = \begin{bmatrix} 0.0323 & 0 \\ 0 & 0.0323 \end{bmatrix}$
  > - **均值 $\mu_t^*$（正确公式）**：
  > - $\sum_i \left ( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \mu[i]_t = \begin{bmatrix} 14.29 & 0 \\ 0 & 14.29 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 16.67 & 0 \\ 0 & 16.67 \end{bmatrix} \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 14.29 \\ 0 \end{bmatrix} + \begin{bmatrix} 33.34 \\ 0 \end{bmatrix} = \begin{bmatrix} 47.63 \\ 0 \end{bmatrix}$
  > - $\mu_t^* = \begin{bmatrix} 0.0323 & 0 \\ 0 & 0.0323 \end{bmatrix} \begin{bmatrix} 47.63 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.54 \\ 0 \end{bmatrix}$
  > - **均值 $\mu_t^*$（错误公式）**：
  > - $(\Sigma_t^*)^{-1} = \begin{bmatrix} 30.96 & 0 \\ 0 & 30.96 \end{bmatrix}$
  > - $\mu_t^* = (\Sigma_t^*)^{-1} \left ( \sum_i \left ( \Sigma[i]_t / \alpha[i]_t \right)^{-1} \mu[i]_t \right) = \begin{bmatrix} 30.96 & 0 \\ 0 & 30.96 \end{bmatrix} \begin{bmatrix} 47.63 \\ 0 \end{bmatrix} = \begin{bmatrix} 1474.63 \\ 0 \end{bmatrix}$
  > - 结果明显错误，数值过大。

**在 ProMPs 中的意义**：

- 组合让多个动作同时生效，适合需要“多任务”的场景（比如同时挥拍和打招呼）。
- 混合让动作平滑切换，适合“顺序任务”（比如先挥拍后打招呼）。
- 高斯分布的解析形式便于计算控制信号，生成自然协调的动作。

#### 总结

ProMPs 通过分布乘积实现运动原语的组合与混合。共同激活用 $\alpha[i]$ 加权，混合用时变 $\alpha[i]_t$ 平滑过渡。在高斯分布下，结果分布的均值和方差有解析形式，支持灵活的动作生成。

## **2.3 使用轨迹分布进行机器人控制**

### **下一状态的分布**

为了充分利用轨迹分布的特性，需要设计一个控制策略，使机器人能够重现这些分布。为此，我们解析地推导了一个随机反馈控制器，该控制器能够精确地重现给定轨迹分布的均值向量 $\mu_t$ 和方差 $\Sigma_t$（对所有时间 $t$ 成立）。  
我们采用基于模型的方法。首先，通过步长 $dt$ 将系统的连续时间动态近似为线性离散时间系统：

$$
y_{t+dt} = (I + A_t dt) y_t + B_t dt u + c_t dt,
$$

其中系统矩阵 $A_t$、输入矩阵 $B_t$ 和漂移向量 $c_t$ 可通过动态系统的一阶泰勒展开获得。我们假设一个具有时变反馈增益的随机线性反馈控制器生成控制动作：

$$
u = K_t y_t + k_t + \tilde{u}, \quad \tilde{u} \sim \mathcal{N}(\tilde{u} \mid 0, \Sigma_u / dt),
$$

其中矩阵 $K_t$ 表示反馈增益矩阵，$k_t$ 为前馈分量。控制噪声 $\tilde{u}$ 表现为维纳过程，其方差随步长 $dt$ 线性增长。将式 (10) 代入式 (9)，可重写系统的下一状态为：

$$
y_{t+dt} = (I + (A_t + B_t K_t) dt) y_t + B_t dt (k_t + \tilde{u}) + c_t dt = F_t y_t + f_t + B_t dt \tilde{u},
$$

其中 $F_t = (I + (A_t + B_t K_t) dt)$，$f_t = B_t k_t dt + c_t dt$。  
为提高清晰度，文中后续部分将省略大多数矩阵的时间下标。从式 (4) 可知，当前状态 $y_t$ 的分布为高斯分布，均值为 $\mu_t = \Psi_t^T \mu_w$，协方差为 $\Sigma_t = \Psi_t^T \Sigma_w \Psi_t$。由于系统动态被建模为高斯线性模型，我们可以通过前向模型解析地得到下一状态的分布 $p(y_{t+dt})$：

$$
p(y_{t+dt}) = \int \mathcal{N}(y_{t+dt} \mid F y_t + f, \Sigma_s dt) \mathcal{N}(y_t \mid \mu_t, \Sigma_t) dy_t = \mathcal{N}(y_{t+dt} \mid F \mu_t + f, F \Sigma_t F^T + \Sigma_s dt),
$$

其中 $dt \Sigma_s = dt B \Sigma_u B^T$ 表示系统噪声矩阵。式 (12) 两侧均为高斯分布，左侧也可以通过期望的轨迹分布 $p(\tau; \theta)$ 计算。我们通过控制律匹配两侧的均值和方差：

$$
\mu_{t+dt} = F \mu_t + (B k + c) dt, \quad \Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt,
$$

其中 $F$ 由式 (11) 给出，包含时变反馈增益 $K$。利用这两个约束，可以求解时变增益 $K$ 和 $k$

#### 解释说明

- **系统动态**：
  - $y_{t+dt} = (I + A_t dt) y_t + B_t dt u + c_t dt$
- **下一状态分布**：
  - 当前位置 $y_t$ 是个概率分布（高斯分布），加了控制和抖动后，$y_{t+dt}$ 还是高斯分布。
  - 算分布就像“平均位置”和“不确定性”：平均位置是 $F \mu_t + f$，不确定性是 $F \Sigma_t F^T + \Sigma_s dt$。
- **匹配目标**：
  - 目标是让机器人轨迹跟目标分布一样：平均位置得是 $\mu_{t+dt}$，不确定性得是 $\Sigma_{t+dt}$。
  - 公式 $\mu_{t+dt} = F \mu_t + (B k + c) dt$ 和 $\Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt$ 就是“预测 = 目标”，用来解 $K$ 和 $k$。

#### 公式推导与细节

- 系统动态：
  $$
  y_{t+dt} = (I + A_t dt) y_t + B_t dt u + c_t dt
  $$

> [!note]+ 为什么是这种形式？
>
> - **为什么是这种形式？**
> - **离散化近似**：
>   - 机器人系统的真实动态通常是连续时间系统，用微分方程描述，例如：
>     $$
>     \dot{y}_t = A_t y_t + B_t u + c_t
>     $$
>     其中 $y_t$ 是状态（例如关节位置和速度），$u$ 是控制输入，$A_t$ 是系统矩阵，$B_t$ 是输入矩阵，$c_t$ 是漂移向量。
>   - 为了在计算机中实现控制，需要将连续时间系统离散化。文中通过一阶泰勒展开（即欧拉方法）近似：
>     - 连续时间动态 $\dot{y}_t = A_t y_t + B_t u + c_t$ 表示状态的变化率。
>     - 离散化后，状态从 $t$ 到 $t+dt$ 的变化为：
>       $$
>       y_{t+dt} = y_t + \dot{y}_t dt
>       $$
>     - 代入 $\dot{y}_t$：
>       $$
>       y_{t+dt} = y_t + (A_t y_t + B_t u + c_t) dt = (I + A_t dt) y_t + B_t dt u + c_t dt
>       $$
>     - 这就是文中给出的形式。
>   - **一阶泰勒展开**：
>     - 文中提到 $A_t$、$B_t$ 和 $c_t$ 可以通过动态系统的一阶泰勒展开获得。
>     - 如果系统是非线性的（例如 $\dot{y}_t = f(y_t, u)$），`一阶泰勒展开将其线性化`：
>       $$
>       f(y_t, u) \approx f(y_0, u_0) + \frac{\partial f}{\partial y} \bigg|_{y_0, u_0} (y_t - y_0) + \frac{\partial f}{\partial u} \bigg|_{y_0, u_0} (u - u_0)
>       $$
>       其中 $\frac{\partial f}{\partial y} = A_t$，$\frac{\partial f}{\partial u} = B_t$，$f(y_0, u_0) - \frac{\partial f}{\partial y} y_0 - \frac{\partial f}{\partial u} u_0 = c_t$。
> - **线性化的必要性**：
>   - ProMPs 的核心是轨迹分布是高斯分布，系统动态需要是线性的，`高斯分布通过线性变换后仍是高斯分布`。
>   - 非线性系统会导致下一状态分布不再是高斯分布，解析计算变得复杂，无法直接匹配均值和方差。
>   - 可以用更精确的公式（比如考虑加速度、二阶变化），但那样算起来太复杂。ProMPs 为了简单（还能用高斯分布），就选了最简单的“一步一步”近似。

- 控制器：
  $$
  u = K_t y_t + k_t + \tilde{u}, \quad \tilde{u} \sim \mathcal{N}(0, \Sigma_u / dt)
  $$

> [!note]+ 控制器为什么这样设置?
>
> - **线性控制器的选择**：
>   - 控制器是线性的（$u$ 是 $y_t$ 的线性函数），因为系统动态是线性的，线性控制器保证下一状态分布仍是高斯分布。
>   - 高斯分布的性质让均值和方差可以解析计算，便于匹配目标轨迹分布。
>   - 可以用更复杂的控制器，例如基于强化学习的策略，或者非线性反馈控制器，但这些方法通常需要数值优化，失去解析解的优点。

> [!note]+ 维纳过程
> **维纳过程是什么？**
>
> 维纳过程（Wiener Process），也称为布朗运动（Brownian Motion），是随机过程的一种，广泛用于描述连续时间中的随机现象。它在数学、物理、金融和工程领域（如 ProMPs 中的机器人控制）有重要应用。以下是其定义和特性：
>
> - **定义**：  
>   维纳过程 $W_t$ 是一个连续时间随机过程，满足以下条件：
>
> 1.  **初始条件**：$W_0 = 0$（起点为 0）。
> 2.  **独立增量**：对于任意时间 $0 \leq t_1 < t_2 < \cdots < t_n$，增量 $W_{t_2} - W_{t_1}, W_{t_3} - W_{t_2}, \ldots, W_{t_n} - W_{t_{n-1}}$ 相互独立。
> 3.  **高斯增量**：增量 $W_{t+s} - W_t$ 服从高斯分布：
>     $$
>     W_{t+s} - W_t \sim \mathcal{N}(0, s)
>     $$
>     即均值为 0，方差为时间间隔 $s$。
> 4.  **连续性**：$W_t$ 的路径几乎处处连续（即 $W_t$ 作为 $t$ 的函数是连续的，但不可微）。
>
> - **数学表示**：
> - 维纳过程的增量 $W_{t+dt} - W_t$ 在小时间步长 $dt$ 内的分布为：
>   $$
>   W_{t+dt} - W_t \sim \mathcal{N}(0, dt)
>   $$
> - 因此，维纳过程的“速度” $\frac{W_{t+dt} - W_t}{dt}$ 的方差为：
>   $$
>   \text{Var}\left( \frac{W_{t+dt} - W_t}{dt} \right) = \frac{\text{Var}(W_{t+dt} - W_t)}{dt^2} = \frac{dt}{dt^2} = \frac{1}{dt}
>   $$

> [!note]+
>
> - 为了让噪声的方差可控，引入一个常数协方差矩阵 $\Sigma_u$，调整噪声强度：
>   - 令 $\tilde{u} \sim \mathcal{N}(0, \Sigma_u / dt)$，其中 $\Sigma_u$ 是噪声的“基础协方差”。
>   - 这样，$\tilde{u}$ 的方差随 $dt$ 变化，符合维纳过程的性质。

- 代入 $u$：

  - $u = K_t y_t + k_t + \tilde{u}$
  - $B_t dt u = B_t dt (K_t y_t + k_t + \tilde{u})$
  - 因此：
    $$
    y_{t+dt} = (I + A_t dt) y_t + B_t dt (K_t y_t + k_t + \tilde{u}) + c_t dt
    $$

- 合并项：
  - $(I + A_t dt) y_t + B_t dt K_t y_t = (I + A_t dt + B_t K_t dt) y_t = (I + (A_t + B_t K_t) dt) y_t$
  - $B_t dt k_t + c_t dt$ 是常数项
  - $B_t dt \tilde{u}$ 是噪声项
  - 定义：
    $$
    F_t = I + (A_t + B_t K_t) dt, \quad f_t = B_t k_t dt + c_t dt
    $$
  - 因此：
    $$
    y_{t+dt} = F_t y_t + f_t + B_t dt \tilde{u}
    $$

**步骤 2：当前状态分布**

- 当前状态 $y_t$ 的分布已知：
  $$
  y_t \sim \mathcal{N}(\mu_t, \Sigma_t)
  $$
  其中 $\mu_t = \Psi_t^T \mu_w$，$\Sigma_t = \Psi_t^T \Sigma_w \Psi_t$（来自轨迹分布）。

**步骤 3：噪声分布**

- 噪声 $\tilde{u} \sim \mathcal{N}(0, \Sigma_u / dt)$
- 噪声项 $B_t dt \tilde{u}$ 的分布：
  - 均值：
    $$
    \mathbb{E}[B_t dt \tilde{u}] = B_t dt \mathbb{E}[\tilde{u}] = 0
    $$
  - 协方差：
    $$
    \text{Var}(B_t dt \tilde{u}) = B_t dt (\Sigma_u / dt) dt B_t^T = B_t \Sigma_u B_t^T dt
    $$
  - 文中定义 $\Sigma_s = B_t \Sigma_u B_t^T$，所以：
    $$
    B_t dt \tilde{u} \sim \mathcal{N}(0, \Sigma_s dt)
    $$

**步骤 4：下一状态 $y_{t+dt}$ 的分布**

- 表达式：
  $$
  y_{t+dt} = F_t y_t + f_t + B_t dt \tilde{u}
  $$
- $y_t$ 和 $\tilde{u}$ 是独立的随机变量，且都是高斯分布。
- **均值**：

  - $\mathbb{E}[y_{t+dt}] = \mathbb{E}[F_t y_t + f_t + B_t dt \tilde{u}]$
  - $F_t$ 和 $f_t$ 是确定性矩阵/向量：
    $$
    \mathbb{E}[F_t y_t] = F_t \mathbb{E}[y_t] = F_t \mu_t
    $$
    $$
    \mathbb{E}[f_t] = f_t
    $$
    $$
    \mathbb{E}[B_t dt \tilde{u}] = 0
    $$
  - 因此：
    $$
    \mathbb{E}[y_{t+dt}] = F_t \mu_t + f_t
    $$

- **协方差**：

  - $\text{Var}(y_{t+dt}) = \text{Var}(F_t y_t + f_t + B_t dt \tilde{u})$
  - $f_t$ 是常数，$\text{Var}(f_t) = 0$。
  - $y_t$ 和 $\tilde{u}$ 独立：
    $$
    \text{Var}(y_{t+dt}) = \text{Var}(F_t y_t) + \text{Var}(B_t dt \tilde{u})
    $$
  - 第一项：
    $$
    \text{Var}(F_t y_t) = F_t \text{Var}(y_t) F_t^T = F_t \Sigma_t F_t^T
    $$
  - 第二项：
    $$
    \text{Var}(B_t dt \tilde{u}) = \Sigma_s dt
    $$
  - 因此：
    $$
    \text{Var}(y_{t+dt}) = F_t \Sigma_t F_t^T + \Sigma_s dt
    $$

- **分布**：
  - 因为 $y_t$ 和 $\tilde{u}$ 都是高斯分布，$y_{t+dt}$ 是它们的线性组合，所以 $y_{t+dt}$ 也服从高斯分布：
    $$
    y_{t+dt} \sim \mathcal{N}(F_t \mu_t + f_t, F_t \Sigma_t F_t^T + \Sigma_s dt)
    $$
    **步骤 5：匹配均值和方差**
- 目标轨迹分布 $p(\tau; \theta)$ 给出 $\mu_{t+dt}$ 和 $\Sigma_{t+dt}$，我们让预测分布 $p(y_{t+dt})$ 匹配目标分布：
  - 均值：
    $$
    \mu_{t+dt} = F \mu_t + f
    $$
    代入 $f = B k dt + c dt$：
    $$
    \mu_{t+dt} = F \mu_t + (B k + c) dt
    $$
  - 方差：
    $$
    \Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt
    $$

#### 总结

- 系统动态通过线性化和离散化得到，控制器是线性反馈形式，带高斯噪声。
- 下一状态分布 $p(y_{t+dt})$ 通过联合分布和边缘化计算，得到高斯分布。
- 匹配均值和方差的约束 $\mu_{t+dt} = F \mu_t + (B k + c) dt$ 和 $\Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt$，为后续求解 $K$ 和 $k$ 提供了基础。

### **控制器增益的推导**

通过重新整理项，方差约束变为：

$$
\Sigma_{t+dt} - \Sigma_t = \Sigma_s dt + (A + B K) \Sigma_t dt + \Sigma_t (A + B K)^T dt + O(dt^2),
$$

其中 $O(dt^2)$ 表示 $dt$ 的二阶项。将式除以 $dt$ 并取极限 $dt \to 0$，二阶项消失，得到方差的时间导数：

$$
\dot{\Sigma}_t = \lim_{dt \to 0} \frac{\Sigma_{t+dt} - \Sigma_t}{dt} = (A + B K) \Sigma_t + \Sigma_t (A + B K)^T + \Sigma_s.
$$

矩阵 $\dot{\Sigma}_t$ 也可以从轨迹分布中得到：$\dot{\Sigma}_t = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t$，将其代入式 (15)。重新整理后，得到：

$$
M + M^T = B K \Sigma_t + (B K \Sigma_t)^T, \quad \text{其中} \quad M = \dot{\Phi}_t \Sigma_w \Phi_t^T - A \Sigma_t - \Sigma_s / 2.
$$

设定 $M = B K \Sigma_t$ 并求解增益矩阵 $K$：

$$
K = B^\dagger \left( \dot{\Psi}_t^T \Sigma_w \Psi_t - A \Sigma_t - \frac{\Sigma_s}{2} \right) \Sigma_t^{-1},
$$

得到解，其中 $B^\dagger$ 表示控制矩阵 $B$ 的伪逆。

#### 解释说明

- **方差约束**：
  - 方差 $\Sigma_t$ 表示轨迹的不确定性，$\Sigma_{t+dt}$ 是下一步的不确定性。
  - 公式 $\Sigma_{t+dt} - \Sigma_t = (A + B K) \Sigma_t dt + \Sigma_t (A + B K)^T dt + \Sigma_s dt + O(dt^2)$ 就像“新不确定性 = 旧不确定性 + 控制影响 + 噪声”。
  - 除以 $dt$，取极限，得到不确定性变化率 $\dot{\Sigma}_t$。
- **解 $K$**：
  - $K$ 是控制器的“调整力度”，决定机器人如何根据当前状态调整动作。
  - 公式 $K = B^\dagger (\dot{\Psi}_t^T \Sigma_w \Psi_t - A \Sigma_t - \Sigma_s / 2) \Sigma_t^{-1}$ 就像“调整力度 = 目标变化率 - 自然变化 - 噪声影响”。
  - $B^\dagger$ 是为了处理 $B$ 矩阵可能不可逆的情况（类似除法用伪逆代替）。

#### 公式推导与细节

**步骤 1：从方差约束开始（式 14）**

- 方差约束来自前文（式 13）：

  $$
  \Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt
  $$

- 其中：

  - $F = I + (A + B K) dt$（省略时间下标）。
  - $\Sigma_s dt = B \Sigma_u B^T dt$，表示系统噪声协方差。

- 展开 $F \Sigma_t F^T$：

  - $F = I + (A + B K) dt$
  - $F^T = (I + (A + B K) dt)^T = I + (A + B K)^T dt$
  - $F \Sigma_t F^T = (I + (A + B K) dt) \Sigma_t (I + (A + B K)^T dt)$
  - 展开矩阵乘法：
    $$
    F \Sigma_t F^T = (I + (A + B K) dt) \left( \Sigma_t + \Sigma_t (A + B K)^T dt \right)
    $$
    $$
    = (I + (A + B K) dt) \Sigma_t + (I + (A + B K) dt) \Sigma_t (A + B K)^T dt
    $$
    $$
    = \Sigma_t + (A + B K) dt \Sigma_t + \Sigma_t (A + B K)^T dt + (A + B K) dt \Sigma_t (A + B K)^T dt
    $$

- 忽略 $O(dt^2)$ 项（因为 $dt$ 很小，$dt^2$ 项可以忽略）：

  - $(A + B K) dt \Sigma_t (A + B K)^T dt$ 是 $dt^2$ 量级，记为 $O(dt^2)$。
  - 因此：
    $$
    F \Sigma_t F^T \approx \Sigma_t + (A + B K) \Sigma_t dt + \Sigma_t (A + B K)^T dt
    $$

- 代回方差约束：

  $$
  \Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt
  $$

  $$
  \Sigma_{t+dt} = \Sigma_t + (A + B K) \Sigma_t dt + \Sigma_t (A + B K)^T dt + \Sigma_s dt + O(dt^2)
  $$

- 两边减去 $\Sigma_t$：
  $$
  \Sigma_{t+dt} - \Sigma_t = (A + B K) \Sigma_t dt + \Sigma_t (A + B K)^T dt + \Sigma_s dt + O(dt^2)
  $$

**步骤 2：取极限，得到方差的时间导数（式 15）**

- 除以 $dt$：

  $$
  \frac{\Sigma_{t+dt} - \Sigma_t}{dt} = (A + B K) \Sigma_t + \Sigma_t (A + B K)^T + \Sigma_s + \frac{O(dt^2)}{dt}
  $$

- 取极限 $dt \to 0$：
  - $\frac{O(dt^2)}{dt} = O(dt)$，当 $dt \to 0$ 时，$O(dt) \to 0$。
  - 因此：
    $$
    \dot{\Sigma}_t = \lim_{dt \to 0} \frac{\Sigma_{t+dt} - \Sigma_t}{dt} = (A + B K) \Sigma_t + \Sigma_t (A + B K)^T + \Sigma_s
    $$

**步骤 3：从轨迹分布计算 $\dot{\Sigma}_t$**

- 轨迹分布的协方差（前文式 4）：

  $$
  \Sigma_t = \Psi_t^T \Sigma_w \Psi_t
  $$

- 对时间 $t$ 求导：
  - 使用矩阵导数规则：
    $$
    \dot{\Sigma}_t = \frac{d}{dt} (\Psi_t^T \Sigma_w \Psi_t)
    $$
  - $\Psi_t$ 是时变矩阵，$\Sigma_w$ 是常数矩阵：
    $$
    \frac{d}{dt} (\Psi_t^T \Sigma_w \Psi_t) = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t
    $$
  - 因此：
    $$
    \dot{\Sigma}_t = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t
    $$

**步骤 4：代入方差导数方程（式 15）**

- 将 $\dot{\Sigma}_t = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t$ 代入：
  $$
  \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t = (A + B K) \Sigma_t + \Sigma_t (A + B K)^T + \Sigma_s
  $$

**步骤 5：整理方程（式 16）**

- 注意到等式两边都是对称矩阵：

  - 左侧 $\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t$ 是对称的（因为 $(\dot{\Psi}_t^T \Sigma_w \Psi_t)^T = \Psi_t^T \Sigma_w \dot{\Psi}_t$）。
  - 右侧 $(A + B K) \Sigma_t + \Sigma_t (A + B K)^T$ 也是对称的（因为 $((A + B K) \Sigma_t)^T = \Sigma_t (A + B K)^T$）。

- 移项整理：

  - 将 $A \Sigma_t + \Sigma_t A^T$ 和 $\Sigma_s$ 移到左侧：
    $$
    \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t - (A \Sigma_t + \Sigma_t A^T) - \Sigma_s = B K \Sigma_t + \Sigma_t K^T B^T
    $$

- 定义中间矩阵 $M$：

  - 为了让等式形式更简单，定义：

    $$
    M + M^T = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t - (A \Sigma_t + \Sigma_t A^T) - \Sigma_s
    $$

  - 因此：
    $$
    M + M^T = B K \Sigma_t + (B K \Sigma_t)^T
    $$
  - 所以：
    $$
    M  = B K \Sigma_t
    $$

- $M$ 还等于什么呢？：
  - 注意到 $M + M^T$ 是对称的，假设：（`等式左右都是对称矩阵，所以想取一半`）
    $$
    M = \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) - A \Sigma_t - \frac{\Sigma_s}{2}
    $$
  - 验证：
    $$
    M + M^T = \left( \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) - A \Sigma_t - \frac{\Sigma_s}{2} \right) + \left( \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) - A \Sigma_t - \frac{\Sigma_s}{2} \right)^T
    $$
    - 第一项：
      $$
      \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) + \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t)^T = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t
      $$
    - 第二项：
      $$
      -A \Sigma_t - (A \Sigma_t)^T = -A \Sigma_t - \Sigma_t A^T
      $$
    - 第三项：
      $$
      -\frac{\Sigma_s}{2} - \frac{\Sigma_s^T}{2} = -\Sigma_s \quad (\text{因为 } \Sigma_s \text{ 是对称的})
      $$
    - 因此：
      $$
      M + M^T = \dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t - (A \Sigma_t + \Sigma_t A^T) - \Sigma_s
      $$

**步骤 6：解 $K$（式 17）**

- 假设 $M = B K \Sigma_t$：

  - 因为 $M + M^T$ 和 $B K \Sigma_t + (B K \Sigma_t)^T$ 都是对称矩阵，这种假设是为了简化求解。
  - 代入 $M$ 的定义：
    $$
    B K \Sigma_t = \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) - A \Sigma_t - \frac{\Sigma_s}{2}
    $$

- 注意到 $\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t$ 是对称矩阵：

  - $\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t = 2 \dot{\Psi}_t^T \Sigma_w \Psi_t$（因为 $\dot{\Psi}_t^T \Sigma_w \Psi_t$ 是对称的）。
  - 因此：
    $$
    \frac{1}{2} (\dot{\Psi}_t^T \Sigma_w \Psi_t + \Psi_t^T \Sigma_w \dot{\Psi}_t) = \dot{\Psi}_t^T \Sigma_w \Psi_t
    $$

- 代入：
  - $B K \Sigma_t = \dot{\Psi}_t^T \Sigma_w \Psi_t - A \Sigma_t - \frac{\Sigma_s}{2}$
  - 右乘 $\Sigma_t^{-1}$（假设 $\Sigma_t$ 可逆）：
    $$
    B K = \left( \dot{\Psi}_t^T \Sigma_w \Psi_t - A \Sigma_t - \frac{\Sigma_s}{2} \right) \Sigma_t^{-1}
    $$
  - 左乘 $B^\dagger$（$B$ 的伪逆）：
    $$
    K = B^\dagger \left( \dot{\Psi}_t^T \Sigma_w \Psi_t - A \Sigma_t - \frac{\Sigma_s}{2} \right) \Sigma_t^{-1}
    $$

> [!note]+ 伪逆是什么？
>
> 伪逆（Pseudo-Inverse），也称为广义逆（Generalized Inverse），是一个矩阵的推广形式，用于解决非方阵或不可逆矩阵的逆问题。在线性代数中，对于一个矩阵 $B$，其伪逆 $B^\dagger$ 满足某些特定的性质，使得它可以用来求解线性方程 $Bx = y$，即使 $B$ 不是方阵或不可逆。`读作 dagger, latex 命令也是\dagger`
>
> 伪逆的数学定义基于 Moore-Penrose 伪逆，满足以下四个条件：
>
> 1.  $B B^\dagger B = B$
> 2.  $B^\dagger B B^\dagger = B^\dagger$
> 3.  $(B B^\dagger)^T = B B^\dagger$（$B B^\dagger$ 是对称的）
> 4.  $(B^\dagger B)^T = B^\dagger B$（$B^\dagger B$ 是对称的）
>
> **数学推导与计算**  
> 伪逆通常通过奇异值分解（Singular Value Decomposition, SVD）计算：
>
> - 假设矩阵 $B \in \mathbb{R}^{m \times n}$，其 SVD 为：
>   $$
>   B = U \Sigma V^T
>   $$
> - $U \in \mathbb{R}^{m \times m}$ 和 $V \in \mathbb{R}^{n \times n}$ 是正交矩阵。
> - $\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵，非零奇异值在对角线上。
> - 伪逆 $B^\dagger$ 为：
>   $$
>   B^\dagger = V \Sigma^\dagger U^T
>   $$
> - $\Sigma^\dagger$ 是 $\Sigma$ 的伪逆，将非零奇异值取倒数，其他元素为 0。
>
> **伪逆的用途**
>
> - **求解线性方程**：
> - 对于 $Bx = y$，如果 $B$ 是方阵且可逆，解为 $x = B^{-1} y$。
> - 如果 $B$ 不可逆或非方阵，伪逆提供最小二乘解（当 $y$ 不在 $B$ 的列空间时）或最小范数解（当解不唯一时）：
>   $$
>   x = B^\dagger y
>   $$
>   **特殊情况**
>
> 1.  **$B$ 是方阵且可逆**：
>
> - $B^\dagger = B^{-1}$，伪逆退化为普通逆。
>
> 2.  **$B$ 有满列秩（$B^T B$ 可逆）**：
>
> - $B^\dagger = (B^T B)^{-1} B^T$，称为左伪逆。
>
> 3.  **$B$ 有满行秩（$B B^T$ 可逆）**：
>
> - $B^\dagger = B^T (B B^T)^{-1}$，称为右伪逆。

### **前馈控制信号的推导**

类似地，我们通过匹配轨迹分布的均值 $\mu_{t+dt}$ 与前向模型计算的均值来获得前馈控制信号 $k$。在重新整理项、除以 $dt$ 并取极限 $dt \to 0$ 后，我们得到前馈控制向量 $k$ 的连续时间约束：

$$
\dot{\mu}_t = (A + B K) \mu_t + B k + c
$$

我们再次利用轨迹分布 $p(\tau; \theta)$，得到 $\mu_t = \Psi_t^T \mu_w$ 和 $\dot{\mu}_t = \dot{\Psi}_t^T \mu_w$，并将这些代入式 (18) 求解 $k$：

$$
k = B^\dagger \left( \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c \right)
$$

其中 $B^\dagger$ 表示控制矩阵 $B$ 的伪逆。

#### 解释说明

- **均值约束**：
  - 均值 $\mu_t$ 表示轨迹的“平均位置”，$\mu_{t+dt}$ 是下一步的平均位置。
  - 公式 $\mu_{t+dt} = F \mu_t + (B k + c) dt$ 就像“新位置 = 旧位置 + 控制推动 + 漂移”。
  - 除以 $dt$，取极限，得到 $\dot{\mu}_t$，就像“位置变化速度 = 控制效果 + 前馈推动 + 漂移”。
- **解 $k$**：
  - $k$ 是前馈控制信号，像是一个“固定推动力”，让轨迹的平均位置按预期走。
  - $\dot{\mu}_t = \dot{\Psi}_t^T \mu_w$ 是轨迹分布告诉我们的“目标速度”。
  - 我们把 $\dot{\mu}_t = (A + B K) \mu_t + B k + c$ 整理，算出 $B k = \text{目标速度} - \text{控制效果} - \text{漂移}$。
  - 最后用 $B^\dagger$（伪逆，像个“万能除法”）把 $k$ 解出来，因为 $B$ 可能不好直接“除”。
- **类比生活**：
  - 想象你在推车，$\mu_t$ 是车的位置，$\dot{\mu}_t$ 是车的速度。
  - 你想让车按计划（轨迹分布）走，$k$ 就是你额外施加的一个“固定推力”。
  - 公式 $k = B^\dagger (...)$ 就像算出“推力 = 目标速度 - 自动滚动的速度 - 风的干扰”，用伪逆是因为推力可能不直接等于某个值，得“凑个最优解”。

#### 公式推导与细节

**步骤 1：从均值约束开始（式 13）**

- 根据前文（式 13），轨迹分布的均值约束为：
  $$
  \mu_{t+dt} = F \mu_t + (B k + c) dt
  $$
- 其中：

  - $F = I + (A + B K) dt$（省略时间下标）。
  - $\mu_t$ 是轨迹分布在时间 $t$ 的均值，$\mu_{t+dt}$ 是时间 $t+dt$ 的均值。
  - $B k + c$ 是前馈项和漂移项，乘以 $dt$ 表示时间步的影响。

- 展开 $F \mu_t$：
  - $F = I + (A + B K) dt$
  - $F \mu_t = (I + (A + B K) dt) \mu_t = \mu_t + (A + B K) \mu_t dt$
- 因此：

  $$
  \mu_{t+dt} = \mu_t + (A + B K) \mu_t dt + (B k + c) dt
  $$

- 两边减去 $\mu_t$：
  $$
  \mu_{t+dt} - \mu_t = (A + B K) \mu_t dt + (B k + c) dt
  $$

**步骤 2：取极限，得到均值的时间导数（式 18）**

- 除以 $dt$：

  $$
  \frac{\mu_{t+dt} - \mu_t}{dt} = (A + B K) \mu_t + B k + c
  $$

- 取极限 $dt \to 0$：
  - 左侧 $\frac{\mu_{t+dt} - \mu_t}{dt}$ 趋向于 $\dot{\mu}_t$（均值的时间导数）。
  - 右侧没有 $dt$ 的高阶项，直接保留。
- 因此：
  $$
  \dot{\mu}_t = (A + B K) \mu_t + B k + c
  $$

**步骤 3：从轨迹分布计算 $\mu_t$ 和 $\dot{\mu}_t$**

- 根据前文（式 4），轨迹分布的均值为：

  $$
  \mu_t = \Psi_t^T \mu_w
  $$

  - $\Psi_t$ 是时间 $t$ 的基函数矩阵，$\mu_w$ 是权重分布的均值。

- 对时间 $t$ 求导：
  - $\mu_t = \Psi_t^T \mu_w$，$\mu_w$ 是常数向量，$\Psi_t$ 是时变矩阵：
    $$
    \dot{\mu}_t = \frac{d}{dt} (\Psi_t^T \mu_w) = \dot{\Psi}_t^T \mu_w
    $$
  - 这里 $\dot{\Psi}_t = \frac{d}{dt} \Psi_t$ 是基函数矩阵对时间的导数。

**步骤 4：代入均值导数方程，求解 $k$（式 19）**

- 将 $\mu_t = \Psi_t^T \mu_w$ 和 $\dot{\mu}_t = \dot{\Psi}_t^T \mu_w$ 代入式 (18)：

  $$
  \dot{\Psi}_t^T \mu_w = (A + B K) (\Psi_t^T \mu_w) + B k + c
  $$

- 整理：
  - 左侧：$\dot{\Psi}_t^T \mu_w$
  - 右侧：$(A + B K) \Psi_t^T \mu_w + B k + c$
- 移项以孤立 $B k$：
  - 将 $(A + B K) \Psi_t^T \mu_w + c$ 移到左侧：
    $$
    \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c = B k
    $$
- 因此：

  $$
  B k = \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c
  $$

- 求解 $k$：
  - $B$ 是输入矩阵，可能不是方阵或不可逆，无法直接用 $B^{-1}$。
  - 使用伪逆 $B^\dagger$（广义逆）来求解：
    - 左乘 $B^\dagger$：
      $$
      B^\dagger B k = B^\dagger \left( \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c \right)
      $$
    - 因为 $B^\dagger B \approx I$（在最小二乘或最小范数意义下），所以：
      $$
      k = B^\dagger \left( \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c \right)
      $$

#### 总结

- 前馈控制信号 $k$ 的推导从均值约束 $\mu_{t+dt} = F \mu_t + (B k + c) dt$ 开始，通过除以 $dt$ 并取极限，得到 $\dot{\mu}_t = (A + B K) \mu_t + B k + c$。
- 利用轨迹分布的 $\mu_t = \Psi_t^T \mu_w$ 和 $\dot{\mu}_t = \dot{\Psi}_t^T \mu_w$，代入后整理出 $B k$，用伪逆 $B^\dagger$ 解得：
  $$
  k = B^\dagger \left( \dot{\Psi}_t^T \mu_w - (A + B K) \Psi_t^T \mu_w - c \right)
  $$

### **控制噪声的估计**

为了匹配轨迹分布，我们还需要匹配用于生成该分布的控制噪声矩阵 $\Sigma_u$。我们首先通过检查轨迹分布的时间步之间的互相关来计算系统噪声协方差 $\Sigma_s = B \Sigma_u B^T$。为此，我们计算当前状态 $y_t$ 和下一状态 $y_{t+dt}$ 的联合分布 $p(y_t, y_{t+dt})$：

$$
p(y_t, y_{t+dt}) = \mathcal{N} \left( \begin{bmatrix} y_t \\ y_{t+dt} \end{bmatrix} \Big| \begin{bmatrix} \mu_t \\ \mu_{t+dt} \end{bmatrix}, \begin{bmatrix} \Sigma_t & C_t \\ C_t^T & \Sigma_{t+dt} \end{bmatrix} \right)
$$

其中 $C_t = \Psi_t \Sigma_w \Psi_{t+dt}^T$ 是互相关矩阵。我们再次使用系统模型来匹配互相关。通过系统动态，$y_t$ 和 $y_{t+dt}$ 的联合分布为：

$$
p(y_t, y_{t+dt}) = \mathcal{N}(y_t \mid \mu_t, \Sigma_t) \mathcal{N}(y_{t+dt} \mid F y_t + f, \Sigma_u)
$$

得到：

$$
p(y_t, y_{t+dt}) = \mathcal{N} \left( \begin{bmatrix} y_t \\ y_{t+dt} \end{bmatrix} \Big| \begin{bmatrix} \mu_t \\ F \mu_t + f \end{bmatrix}, \begin{bmatrix} \Sigma_t & \Sigma_t F^T \\ F \Sigma_t & F \Sigma_t F^T + \Sigma_s dt \end{bmatrix} \right)
$$

通过匹配式 (20) 和式 (21) 的协方差矩阵，可以得到系统噪声协方差：

$$
\Sigma_s dt = \Sigma_{t+dt} - F \Sigma_t F^T = \Sigma_{t+dt} - F \Sigma_t \Sigma_t^{-1} \Sigma_t F^T = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t
$$

控制噪声的协方差 $\Sigma_u$ 随后由 $\Sigma_u = B^\dagger \Sigma_s B^{\dagger T}$ 给出。从式 (22) 可以看出，随机反馈控制器的协方差不依赖于控制器增益，因此可以在估计控制器增益之前预先计算。

#### 解释说明

- **联合分布**：
  - 联合分布 $p(y_t, y_{t+dt})$ 就像在看“现在的位置”和“下一步的位置”一起的概率分布。
  - 轨迹分布告诉我们这两个位置的“平均值”（$\mu_t, \mu_{t+dt}$）和“不确定性”（$\Sigma_t, \Sigma_{t+dt}$），还有它们之间的“相关性”（$C_t$）。
  - 系统动态也给我们一个联合分布，里面有 $\Sigma_s dt$，表示噪声带来的额外不确定性。
- **匹配协方差**：
  - 我们把轨迹分布的协方差矩阵和系统动态的协方差矩阵对比，找到 $\Sigma_s$。
  - 公式 $\Sigma_s dt = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t$ 就像“下一步的不确定性 - 当前与下一步的相关性影响”。
- **算 $\Sigma_u$**：
  - $\Sigma_s = B \Sigma_u B^T$ 是系统噪声，$\Sigma_u$ 是控制噪声，就像“机器人动作抖动的强度”。
  - 用伪逆 $B^\dagger$ 把 $\Sigma_s$ 转回 $\Sigma_u$，因为 $B$ 可能不好直接“除”。
  - 就像你知道风吹的结果（$\Sigma_s$），想反推风有多大（$\Sigma_u$），伪逆帮你估个最优解。
- **不依赖增益**：
  - $\Sigma_s$ 只看轨迹分布的数据（$\Sigma_{t+dt}, C_t$），跟控制器增益 $K$ 没关系，所以可以先算好 $\Sigma_u$，再去算 $K$ 和 $k$。
  - 就像你先测好风力，再决定怎么推车，不用等推车策略定了才测风。

#### 公式推导与细节

**步骤 1：联合分布的定义（式 20）**

- 目标是计算当前状态 $y_t$ 和下一状态 $y_{t+dt}$ 的联合分布 $p(y_t, y_{t+dt})$，以确定系统噪声协方差 $\Sigma_s$。
- 根据轨迹分布，联合分布为高斯分布：
  $$
  p(y_t, y_{t+dt}) = \mathcal{N} \left( \begin{bmatrix} y_t \\ y_{t+dt} \end{bmatrix} \Big| \begin{bmatrix} \mu_t \\ \mu_{t+dt} \end{bmatrix}, \begin{bmatrix} \Sigma_t & C_t \\ C_t^T & \Sigma_{t+dt} \end{bmatrix} \right)
  $$
- 其中：
  - 均值向量：$\begin{bmatrix} \mu_t \\ \mu_{t+dt} \end{bmatrix}$，$\mu_t$ 和 $\mu_{t+dt}$ 是轨迹分布在时间 $t$ 和 $t+dt$ 的均值。
  - 协方差矩阵：
    - $\Sigma_t$ 是 $y_t$ 的协方差。
    - $\Sigma_{t+dt}$ 是 $y_{t+dt}$ 的协方差。
    - $C_t = \Psi_t \Sigma_w \Psi_{t+dt}^T$ 是 $y_t$ 和 $y_{t+dt}$ 的互相关矩阵，表示两时间步之间的相关性。
- 互相关 $C_t$ 的定义：
  - 根据前文（式 4），$y_t = \Psi_t^T w$，$y_{t+dt} = \Psi_{t+dt}^T w$，$w \sim \mathcal{N}(\mu_w, \Sigma_w)$。
  - 互相关：
    $$
    C_t = \text{Cov}(y_t, y_{t+dt}) = \text{Cov}(\Psi_t^T w, \Psi_{t+dt}^T w) = \Psi_t^T \text{Cov}(w, w) \Psi_{t+dt} = \Psi_t^T \Sigma_w \Psi_{t+dt}
    $$
  - 注意文中写 $C_t = \Psi_t \Sigma_w \Psi_{t+dt}^T$，可能有笔误：
    - 正确形式应为 $C_t = \Psi_t^T \Sigma_w \Psi_{t+dt}$，因为 $\Psi_t^T$ 和 $\Psi_{t+dt}$ 是基函数矩阵的转置，与权重 $w$ 的维度匹配。

**步骤 2：系统动态的联合分布（式 21）**

- 根据系统动态，联合分布可以通过条件分布计算：
  - 当前状态：$y_t \sim \mathcal{N}(\mu_t, \Sigma_t)$
  - 下一状态（根据式 11）：
    $$
    y_{t+dt} = F y_t + f + B \tilde{u} dt, \quad \tilde{u} \sim \mathcal{N}(0, \Sigma_u / dt)
    $$
    - 噪声项 $B \tilde{u} dt$ 的协方差：
      $$
      \text{Var}(B \tilde{u} dt) = B (\Sigma_u / dt) B^T dt = B \Sigma_u B^T dt = \Sigma_s dt
      $$
    - 因此：
      $$
      y_{t+dt} \mid y_t \sim \mathcal{N}(F y_t + f, \Sigma_s dt)
      $$
- 联合分布：
  $$
  p(y_t, y_{t+dt}) = \mathcal{N}(y_t \mid \mu_t, \Sigma_t) \mathcal{N}(y_{t+dt} \mid F y_t + f, \Sigma_s dt)
  $$
- 计算联合高斯分布：
  - 均值向量：
    - $\mathbb{E}[y_t] = \mu_t$
    - $\mathbb{E}[y_{t+dt} \mid y_t] = F y_t + f$
    - 因此：
      $$
      \mathbb{E} \begin{bmatrix} y_t \\ y_{t+dt} \end{bmatrix} = \begin{bmatrix} \mu_t \\ F \mu_t + f \end{bmatrix}
      $$
  - 协方差矩阵：
    - $\text{Var}(y_t) = \Sigma_t$
    - $\text{Var}(y_{t+dt} \mid y_t) = \Sigma_s dt$
    - 互相关 $\text{Cov}(y_t, y_{t+dt})$：
      - $\text{Cov}(y_t, y_{t+dt}) = \text{Cov}(y_t, F y_t + f + B \tilde{u} dt)$
      - $f$ 是常数，$\text{Cov}(y_t, f) = 0$
      - $\tilde{u}$ 与 $y_t$ 独立，$\text{Cov}(y_t, B \tilde{u} dt) = 0$
      - $\text{Cov}(y_t, F y_t) = \text{Cov}(y_t, F y_t) = \Sigma_t F^T$
      - 因此：
        $$
        \text{Cov}(y_t, y_{t+dt}) = \Sigma_t F^T
        $$
    - 同样，$\text{Cov}(y_{t+dt}, y_t) = (\text{Cov}(y_t, y_{t+dt}))^T = F \Sigma_t$
    - $\text{Var}(y_{t+dt}) = \text{Var}(F y_t + f + B \tilde{u} dt) = F \Sigma_t F^T + \Sigma_s dt$
  - 联合协方差矩阵：
    $$
    \begin{bmatrix} \Sigma_t & \Sigma_t F^T \\ F \Sigma_t & F \Sigma_t F^T + \Sigma_s dt \end{bmatrix}
    $$
- 因此：
  $$
  p(y_t, y_{t+dt}) = \mathcal{N} \left( \begin{bmatrix} y_t \\ y_{t+dt} \end{bmatrix} \Big| \begin{bmatrix} \mu_t \\ F \mu_t + f \end{bmatrix}, \begin{bmatrix} \Sigma_t & \Sigma_t F^T \\ F \Sigma_t & F \Sigma_t F^T + \Sigma_s dt \end{bmatrix} \right)
  $$

**步骤 3：匹配协方差矩阵，求 $\Sigma_s$（式 22）**

- 比较式 (20) 和式 (21) 的协方差矩阵：
  - 式 (20)：
    $$
    \begin{bmatrix} \Sigma_t & C_t \\ C_t^T & \Sigma_{t+dt} \end{bmatrix}
    $$
  - 式 (21)：
    $$
    \begin{bmatrix} \Sigma_t & \Sigma_t F^T \\ F \Sigma_t & F \Sigma_t F^T + \Sigma_s dt \end{bmatrix}
    $$
- 匹配对应项：
  - $\Sigma_t = \Sigma_t$（显然成立）。
  - 互相关：
    $$
    C_t = \Sigma_t F^T
    $$
    - 验证：
      - $C_t = \Psi_t^T \Sigma_w \Psi_{t+dt}$ 是轨迹分布的互相关。
      - $\Sigma_t F^T$ 是系统动态的互相关，可能需要进一步验证与 $\Psi_t^T \Sigma_w \Psi_{t+dt}$ 的一致性（后文会讨论）。
  - 下一状态协方差：
    $$
    \Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt
    $$
- 孤立 $\Sigma_s dt$：
  - 从 $\Sigma_{t+dt} = F \Sigma_t F^T + \Sigma_s dt$ 移项：
    $$
    \Sigma_s dt = \Sigma_{t+dt} - F \Sigma_t F^T
    $$
- 进一步用互相关 $C_t$ 表示：

  - 从 $C_t = \Sigma_t F^T$，得 $F = \Sigma_t^{-1} C_t$（假设 $\Sigma_t$ 可逆）。
  - 代入：
    - $F \Sigma_t F^T = (\Sigma_t^{-1} C_t) \Sigma_t (\Sigma_t^{-1} C_t)^T = \Sigma_t^{-1} C_t \Sigma_t \Sigma_t^{-1} C_t^T = C_t^T \Sigma_t^{-1} C_t$
    - 因此：
      $$
      \Sigma_s dt = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t
      $$

- 因此：
  $$
  \Sigma_s dt = \Sigma_{t+dt} - F \Sigma_t F^T = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t
  $$

**步骤 4：计算控制噪声 $\Sigma_u$**

- 已知 $\Sigma_s = B \Sigma_u B^T$，需要求 $\Sigma_u$：
  - $\Sigma_s = \frac{1}{dt} (\Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t)$
  - 方程 $B \Sigma_u B^T = \Sigma_s$ 需要解 $\Sigma_u$。
- 使用伪逆：
  - 左乘 $B^\dagger$，右乘 $B^{\dagger T}$：
    $$
    B^\dagger B \Sigma_u B^T B^{\dagger T} = B^\dagger \Sigma_s B^{\dagger T}
    $$
  - 假设 $\Sigma_u$ 是对称的（控制噪声协方差通常是对称矩阵），可以近似：
    $$
    \Sigma_u \approx B^\dagger \Sigma_s B^{\dagger T}
    $$
  - 严格来说，$B^\dagger B \neq I$（除非 $B$ 满列秩），但在最小二乘意义下，$B^\dagger \Sigma_s B^{\dagger T}$ 提供了 $\Sigma_u$ 的一个估计。
- 文中公式：
  $$
  \Sigma_u = B^\dagger \Sigma_s B^{\dagger T}
  $$

**步骤 5：控制噪声不依赖增益的性质**

- 从式 (22)：
  - $\Sigma_s dt = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t$
  - $\Sigma_{t+dt}$ 和 $C_t = \Psi_t^T \Sigma_w \Psi_{t+dt}$ 都来自轨迹分布，直接由 $\Psi_t, \Psi_{t+dt}, \Sigma_w$ 确定。
  - $F = I + (A + B K) dt$ 出现在 $C_t = \Sigma_t F^T$ 中，但最终 $\Sigma_s$ 的计算不直接依赖 $K$（因为 $F \Sigma_t F^T$ 被替换为 $C_t^T \Sigma_t^{-1} C_t$）。
- 因此，$\Sigma_s$ 和 $\Sigma_u$ 可以预先计算，不依赖于控制器增益 $K$ 和 $k$。

#### 总结

- 控制噪声 $\Sigma_u$ 的估计通过匹配轨迹分布和系统动态的联合分布 $p(y_t, y_{t+dt})$ 的协方差矩阵完成。
- 系统噪声协方差 $\Sigma_s$ 从 $\Sigma_s dt = \Sigma_{t+dt} - C_t^T \Sigma_t^{-1} C_t$ 计算，其中 $C_t = \Psi_t^T \Sigma_w \Psi_{t+dt}$ 是互相关。
- 控制噪声 $\Sigma_u = B^\dagger \Sigma_s B^{\dagger T}$，使用伪逆 $B^\dagger$ 确保解的存在。
- $\Sigma_s$ 和 $\Sigma_u$ 不依赖控制器增益，可以预先计算。
- 推导中纠正了文中笔误（$C_t$ 和条件分布协方差），确保公式一致性。
- 通俗理解：$\Sigma_u$ 是控制动作的“抖动强度”，通过比较轨迹分布和系统动态的不确定性，估算出来，且不受控制器设计影响。
