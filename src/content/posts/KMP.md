---
title: kmp算法
date: 2025-05-25
lastMod: 2025-05-26
summary: 介绍一下kmp算法
category: MP类算法
tags: [技能学习, MP类算法, 模仿学习]
---

# KMP 算法

## 定义示范数据集

定义了示范数据，写成一个集合：$\{{s_{n,h},\xi_{n,h}}\}_{n=1}^N\}_{h=1}^H$。这里 $s_{n,h}\in\mathbb{R}^I$ 是输入（比如时间或位置），$\xi_{n,h}\in\mathbb{R}^O$ 是输出（比如手的速度或位置）。$I$ 是输入的维度（比如时间是 1 维，位置是 3 维），$O$ 是输出的维度，$H$ 是示范次数（比如你示范了 5 次），$N$ 是每条轨迹的长度（比如 100 个点）。这些数据就像一堆“输入-输出对”，记录了每次示范的细节。

## 输入和输出能代表什么？

这很灵活。
如果 $s$ 是机器人位置，$\xi$ 是速度，就成了“自主系统”，轨迹不靠时间，自己决定下一步。比如“手在 A 点，下一秒该往哪动”。
如果 $s$ 是时间，$\xi$ 是位置，就是“时间驱动轨迹”，按秒走。比如“1 秒时手在 A，2 秒时到 B”。这让模型能适应不同任务类型。

## 怎么学这些示范呢？

用概率模型抓住分布，比如高斯混合模型（GMM）、隐马尔可夫模型（HMM），或者简单的单高斯分布。这些模型像“统计工具”，分析多次示范的规律。GMM 把数据分成几类，每类用高斯分布描述；HMM 考虑动作顺序（比如先抬手再放下）；单高斯最简单，把所有数据看成一堆。作者选了 GMM 举例，因为它能同时看输入和输出的“联合概率分布” $P(s,\xi)$。

## GMM 的公式

$\begin{bmatrix}s\\\xi\end{bmatrix}\sim\sum_{c=1}^C\pi_c\mathcal{N}(\mu_c,\Sigma_c)$。
啥意思呢？想象你在画示范的“概率地图”。GMM 把数据分成 $C$ 个“山头”，每座山代表一种模式。$\pi_c$ 是每座山的大小（占比，比如 60% 快搬，40% 慢搬），$\mu_c$ 是山顶的位置（平均值，比如快搬的平均时间和位置），$\Sigma_c$ 是山的形状（方差，描述散布范围）。比如，你示范三次搬箱子，GMM 可能发现两座山：一座是“快搬”，一座是“慢搬”。

具体点说，$\mu_c$ 是个向量
像这样：
$\mu_c=\begin{bmatrix}\mu_{c,s}\\\mu_{c,\xi}\end{bmatrix}$，其中 $\mu_{c,s}$ 是输入的平均值，$\mu_{c,\xi}$ 是输出的平均值。$\Sigma_c$ 是个矩阵：$\Sigma_c=\begin{bmatrix}\Sigma_{c,ss}&\Sigma_{c,s\xi}\\\Sigma_{c,\xi s}&\Sigma_{c,\xi\xi}\end{bmatrix}$，描述输入和输出之间的关系和散布。比如 $\Sigma_{c,s\xi}$ 表示时间和位置的关系（协方差）。这些参数通过示范数据算出来，比如用期望最大化（EM）算法。

## 参数化轨迹

“我们从一个参数化的轨迹开始推导 KMP。”
这个轨迹写成公式：$\xi(s)=\Theta(s)^Tw$。
比如教机器人画一条线，$s$ 是输入（比如时间或位置），$\xi(s)$ 是输出（比如手的坐标）。$\Theta(s)$ 是一个矩阵，像一组“画笔模板”，$w$ 是这些模板的“用力大小”（权重）。机器人用这些模板和权重组合出轨迹。

具体看 $\Theta(s)$，它定义为：$\Theta(s)=\begin{bmatrix}\phi(s)&0&\cdots&0\\0&\phi(s)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\phi(s)\end{bmatrix}$，这里 $\Theta(s)\in\mathbb{R}^{BO\times O}$，$B$ 是模板数量，$O$ 是输出维度（比如 3 维位置）。$\phi(s)\in\mathbb{R}^B$ 是一组基函数（basis functions），就像不同形状的画笔。比如 $s$ 是时间，$\phi(s)$ 可能是“直线”“曲线”之类的小模板。$\Theta(s)$ 把这些模板按输出维度排列，每个维度用同样的 $\phi(s)$，但独立处理。

$w$ 是权重向量，$w\in\mathbb{R}^{BO}$，长度是 $B\times O$，因为每个维度都要配一组权重。比如 $O=3$（x、y、z 坐标），$B=5$（5 个模板），$w$ 就有 15 个元素。作者假设 $w$ 服从正态分布：$w\sim\mathcal{N}(\mu_w,\Sigma_w)$，其中 $\mu_w$ 是平均值，$\Sigma_w$ 是协方差，都是未知的，得学出来。

有了这个假设，轨迹 $\xi(s)$ 也变成概率分布：$\xi(s)\sim\mathcal{N}(\Theta(s)^T\mu_w,\Theta(s)^T\Sigma_w\Theta(s))$

$\xi(s)$ 不再是固定值，而是一个随机值，平均是 $\Theta(s)^T\mu_w$（模板和权重的平均组合），方差是 $\Theta(s)^T\Sigma_w\Theta(s)$（不确定性）。

目标是什么呢？“我们要模仿一个参考轨迹 $\{\hat{\xi}_n\}_{n=1}^N$。”这个参考轨迹是从 2.1 节的示范（比如 GMM 学出来的）来的。我们希望公式 (4) 的分布尽量接近参考轨迹的分布。怎么做？用 KL 散度（Kullback-Leibler divergence）来衡量两者的差距，然后最小化它。KL 散度像个“相似度尺子”，值越小越像。

推导分几步：先在 2.2.1 节用 KL 散度定目标，然后在 2.2.2 和 2.2.3 节分别求解 $\mu_w$ 和 $\Sigma_w$ 的最优解，最后用“核技巧”（kernel trick）把模型变成 KMP。`核技巧是个聪明办法，能避开复杂的基函数，直接用数据算结果。`

### 小结

1. **参数化轨迹**

   - 公式：$\xi(s)=\Theta(s)^Tw$。
   - $s$ 是输入，$\xi(s)$ 是输出，$\Theta(s)$ 是模板矩阵，$w$ 是权重。
   - 比如 $s=时间$，$\xi(s)=位置$，$\Theta(s)$ 决定形状，$w$ 决定大小。

2. **$\Theta(s)$ 的结构**

   - $\Theta(s)=\begin{bmatrix}\phi(s)&0&\cdots&0\\0&\phi(s)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\phi(s)\end{bmatrix}$。
   - $\phi(s)$ 是 $B$ 维基函数，比如 $[\sin(s),\cos(s),\ldots]$。
   - 每个输出维度（$O$ 个）用同样的 $\phi(s)$，但独立组合。

3. **权重的分布**

   - $w\sim\mathcal{N}(\mu_w,\Sigma_w)$。
   - $w$ 是随机变量，$\mu_w$ 是平均权重，$\Sigma_w$ 是权重的不确定性。
   - 比如 $w$ 决定“用力多大”，它不是固定值，而是带点随机。

4. **轨迹的分布**

   - $\xi(s)\sim\mathcal{N}(\Theta(s)^T\mu_w,\Theta(s)^T\Sigma_w\Theta(s))$。
   - 均值 $\Theta(s)^T\mu_w$ 是轨迹的“主线”，方差 $\Theta(s)^T\Sigma_w\Theta(s)$ 是“抖动范围”。
   - 这模仿了示范的随机性。

5. **优化目标**
   - 让 $\xi(s)$ 的分布接近参考轨迹 $\{\hat{\xi}_n\}$ 的分布。
   - 用 KL 散度最小化差距，KL 散度是概率分布的“距离”。

## **2.2.1 节：基于信息论的模仿学习**

用信息论（KL 散度）优化 KMP 的轨迹分布，让它尽量接近参考轨迹。作者把问题拆成两个子问题：优化均值和协方差，为后面推导 KMP 铺路。

作者说：“KL 散度能衡量两个概率分布的差距，我们用它来优化参数化轨迹，让它匹配参考轨迹。”啥是 KL 散度？想象你在比较两张“概率地图”，一张是你设计的（参数化轨迹），一张是示范给的（参考轨迹）。KL 散度像个“尺子”，告诉你两张图差多少，差距越小越像。从信息论角度看，最小化 KL 散度就像“少丢信息”，保证模仿时不失真。

### kL 散度

> [!note] KL 散度
> KL 散度（Kullback-Leibler 散度），也叫相对熵，是一种衡量两个概率分布之间差异的方法。在信息论和统计学中，它常用来量化一个分布相对于另一个分布的“信息损失”。
>
> 具体来说，对于两个概率分布 $P(x)$ 和 $Q(x)$，KL 散度的定义是：
>
> $D_{KL}(P||Q)=\sum_{x}P(x)\log\left(\frac{P(x)}{Q(x)}\right)$
>
> 如果是连续分布，则用积分形式：
>
> $D_{KL}(P||Q)=\int P(x)\log\left(\frac{P(x)}{Q(x)}\right)dx$
>
> 几个关键点：
>
> 1.  非负性：KL 散度总是大于等于 0。当 $P=Q$ 时，$D_{KL}(P||Q)=0$，表示两个分布完全相同。
> 2.  不对称性：$D_{KL}(P||Q)\neq D_{KL}(Q||P)$，所以它不是严格意义上的“距离”，而是一种散度。
> 3.  直观理解：它可以看作是用 $Q$ 去近似 $P$ 时额外需要的“信息量”。
>
> 应用场景：
>
> - 机器学习：比如在变分推断中，用 KL 散度最小化近似分布和真实分布的差异。
> - 信息论：衡量编码效率或数据压缩的理论基础。
> - 统计检验：比较模型分布和真实数据的契合度。

> [!example] 举例
> 假设有两个正态分布：
>
> - $P(x)$ 是均值为 $\mu_1=0$，标准差为 $\sigma_1=1$ 的正态分布，即 $P(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$
> - $Q(x)$ 是均值为 $\mu_2=1$，标准差为 $\sigma_2=1$ 的正态分布，即 $Q(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{(x-1)^2}{2}}$
>
> 我们要计算 $D_{KL}(P||Q)$。
>
> 公式是：
>
> $D_{KL}(P||Q)=\int_{-\infty}^{\infty}P(x)\log\left(\frac{P(x)}{Q(x)}\right)dx$
>
> 首先计算 $\frac{P(x)}{Q(x)}$：
>
> $\frac{P(x)}{Q(x)}=\frac{\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}}{\frac{1}{\sqrt{2\pi}}e^{-\frac{(x-1)^2}{2}}}=e^{-\frac{x^2}{2}+\frac{(x-1)^2}{2}}$
>
> 化简指数部分：
>
> $-\frac{x^2}{2}+\frac{(x-1)^2}{2}=-\frac{x^2}{2}+\frac{x^2-2x+1}{2}=\frac{x^2-2x+1-x^2}{2}=\frac{-2x+1}{2}$
>
> 所以：
>
> $\frac{P(x)}{Q(x)}=e^{\frac{-2x+1}{2}}=e^{-\frac{2x-1}{2}}$
>
> 对数形式：
>
> $\log\left(\frac{P(x)}{Q(x)}\right)=\log\left(e^{-\frac{2x-1}{2}}\right)=-\frac{2x-1}{2}$
>
> 于是：
>
> $D_{KL}(P||Q)=\int_{-\infty}^{\infty}P(x)\left(-\frac{2x-1}{2}\right)dx$
>
> 拆开积分：
>
> $D_{KL}(P||Q)=-\frac{1}{2}\int_{-\infty}^{\infty}P(x)(2x-1)dx=-\frac{1}{2}\left(2\int_{-\infty}^{\infty}xP(x)dx-\int_{-\infty}^{\infty}P(x)dx\right)$
>
> 计算每一项：
>
> 1.  $\int_{-\infty}^{\infty}P(x)dx=1$（概率密度函数的性质）
> 2.  $\int_{-\infty}^{\infty}xP(x)dx=E_P[x]=\mu_1=0$（$P(x)$ 的均值）
>
> 代入：
>
> $D_{KL}(P||Q)=-\frac{1}{2}(2\times 0-1)=-\frac{1}{2}\times(-1)=\frac{1}{2}$
>
> 结果是 $D_{KL}(P||Q)=\frac{1}{2}$。这表明两个均值相差 1、标准差相同的正态分布之间的 KL 散度为 $\frac{1}{2}$。

> [!note] 正态分布的 KL 散度的解析解
> 对于正态分布，KL 散度有解析解。当 $P\sim N(\mu_1,\sigma_1^2)$，$Q\sim N(\mu_2,\sigma_2^2)$ 时：
>
> $D_{KL}(P||Q)=\frac{1}{2}\left[\frac{(\mu_1-\mu_2)^2}{\sigma_2^2}+\frac{\sigma_1^2}{\sigma_2^2}-1-\log\left(\frac{\sigma_1^2}{\sigma_2^2}\right)\right]$
>
> 代入 $\mu_1=0$，$\mu_2=1$，$\sigma_1=\sigma_2=1$，验证：
>
> $D_{KL}(P||Q)=\frac{1}{2}\left[\frac{(0-1)^2}{1}+\frac{1}{1}-1-\log\left(\frac{1}{1}\right)\right]=\frac{1}{2}[1+1-1-0]=\frac{1}{2}$

> [!note] KL 散度的解析解
> 对于任意两个概率分布，KL 散度 $D_{KL}(P||Q)$ 不一定都有解析解。是否能得到解析解取决于 $P(x)$ 和 $Q(x)$ 的具体形式以及 $\log\left(\frac{P(x)}{Q(x)}\right)$ 是否能方便地积分或求和。
>
> 1.  **有解析解的情况**：
>
> - 正态分布：如上一个例子，当 $P$ 和 $Q$ 都是正态分布时，KL 散度有明确的公式：
>
>   $D_{KL}(P||Q)=\frac{1}{2}\left[\frac{(\mu_1-\mu_2)^2}{\sigma_2^2}+\frac{\sigma_1^2}{\sigma_2^2}-1-\log\left(\frac{\sigma_1^2}{\sigma_2^2}\right)\right]$
>
> - 指数分布：如果 $P(x)=\lambda_1 e^{-\lambda_1 x}$，$Q(x)=\lambda_2 e^{-\lambda_2 x}$（$x\geq 0$），则：
>
>   $D_{KL}(P||Q)=\log\left(\frac{\lambda_1}{\lambda_2}\right)+\frac{\lambda_2}{\lambda_1}-1$
>
> - 离散均匀分布：如果 $P$ 和 $Q$ 是有限离散均匀分布，也可以直接求和得到解析解。
>
> 2.  **无解析解的情况**：
>
> - 当 $P$ 和 $Q$ 是复杂分布（如混合高斯分布、非标准分布）时，$\frac{P(x)}{Q(x)}$ 的形式可能非常复杂，导致积分或求和无法解析求解。
> - 例如，$P$ 是高斯混合模型，$Q$ 是单一高斯分布，KL 散度通常需要数值方法近似计算。
>
> 因此，KL 散度的解析解并非通用的，依赖于分布对的数学性质。对于无法解析的情况，可以用蒙特卡洛方法或变分近似来估计。

> [!note] 散度家族
> KL 散度属于“散度”（divergence）家族的一部分，衡量分布间差异的其他方法有很多，以下是一些常见的类似概念：
>
> 1.  **Jensen-Shannon 散度（JS 散度）**
>
> - 定义：$D_{JS}(P||Q)=\frac{1}{2}D_{KL}(P||M)+\frac{1}{2}D_{KL}(Q||M)$，其中 $M=\frac{P+Q}{2}$ 是平均分布。
> - 特点：对称，且有界（对于离散分布，取值在 $[0,\log 2]$ 之间）。比 KL 散度更像“距离”，但仍不是严格的度量。
> - 应用：常用于机器学习中需要对称性或稳定性的场景。
>
> 1.  **Wasserstein 距离（地球移动距离）**
>
> - 定义：$W(P,Q)=\inf_{\gamma\in\Gamma(P,Q)}\int|x-y|d\gamma(x,y)$，其中 $\Gamma(P,Q)$ 是 $P$ 和 $Q$ 的联合分布集合。
> - 特点：是真正的距离度量（满足对称性和三角不等式），考虑分布的几何结构。
> - 应用：生成模型（如 GANs）中常用，因为它能捕捉分布的空间关系。
>
> 3.  **总变差距离（Total Variation Distance）**
>
> - 定义：$TV(P,Q)=\frac{1}{2}\int|P(x)-Q(x)|dx$（连续）或 $\frac{1}{2}\sum_x|P(x)-Q(x)|$（离散）。
> - 特点：对称，有界（取值在 $[0,1]$），简单直观。
> - 应用：统计检验和概率分布比较。
>
> 1.  **Hellinger 距离**
>
> - 定义：$H(P,Q)=\frac{1}{\sqrt{2}}\sqrt{\int(\sqrt{P(x)}-\sqrt{Q(x)})^2dx}$。
> - 特点：对称，有界（取值在 $[0,1]$），对分布的平方根敏感。
> - 应用：常用于信息检索和概率分布比较。
>
> 1.  **Rényi 散度**
>
> - 定义：$D_{\alpha}(P||Q)=\frac{1}{\alpha-1}\log\left(\int P(x)^{\alpha}Q(x)^{1-\alpha}dx\right)$，其中 $\alpha>0$ 且 $\alpha\neq 1$。
> - 特点：KL 散度是其极限形式（$\alpha\to 1$），通过调整 $\alpha$ 可以控制对分布尾部的敏感度。
> - 应用：信息论和统计物理。
>
> 6.  **f-散度**
>
> - 定义：$D_f(P||Q)=\int Q(x)f\left(\frac{P(x)}{Q(x)}\right)dx$，其中 $f$ 是凸函数。
> - 特点：KL 散度是 $f(t)=t\log t$ 的特例，JS 散度和总变差距离也属于此类。
> - 应用：提供统一的散度框架，灵活性高。

`用 KL 散度把模仿学习变成优化问题。`
目标是让参数化轨迹 $\xi(s)$ 的分布匹配参考轨迹，拆成均值和协方差两步解。

目标是最小化一个函数：$J_{ini}(\mu_w,\Sigma_w)=\sum_{n=1}^ND_{KL}(P_p(\xi|s_n)||P_r(\xi|s_n))$
这里 $P_p(\xi|s_n)$ 是参数化轨迹的分布，$P_r(\xi|s_n)$ 是参考轨迹的分布，$N$ 是轨迹点数（比如 100 个点）。$D_{KL}$ 是 KL 散度，衡量每个点 $s_n$ 处的分布差距，总和就是 $J_{ini}$。

$P_p(\xi|s_n)=\mathcal{N}(\xi|\Theta(s_n)^T\mu_w,\Theta(s_n)^T\Sigma_w\Theta(s_n))$

$P_r(\xi|s_n)=\mathcal{N}(\xi|\hat{\mu}_n,\hat{\Sigma}_n)$$\hat{\mu}_n$ 是参考轨迹的均值（比如示范的平均位置），$\hat{\Sigma}_n$ 是协方差（示范的抖动范围）。

KL 散度的定义是：$D_{KL}(P_p(\xi|s_n)||P_r(\xi|s_n))=\int P_p(\xi|s_n)\log\frac{P_p(\xi|s_n)}{P_r(\xi|s_n)}d\xi$
这像在算“信息差”，但直接算积分太麻烦。
因为两个分布都是高斯分布，可以用现成的公式简化。

对于两个高斯分布 $\mathcal{N}(\mu_1,\Sigma_1)$ 和 $\mathcal{N}(\mu_2,\Sigma_2)$，KL 散度是：$D_{KL}=\frac{1}{2}(\log|\Sigma_2|-\log|\Sigma_1|-d+\text{Tr}(\Sigma_2^{-1}\Sigma_1)+(\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2))$。
这里 $d$ 是维度（$O$），$|\cdot|$ 是行列式，$\text{Tr}(\cdot)$ 是矩阵迹（对角线和）。

代入 $P_p$ 和 $P_r$：$\mu_1=\Theta(s_n)^T\mu_w$，$\Sigma_1=\Theta(s_n)^T\Sigma_w\Theta(s_n)$，$\mu_2=\hat{\mu}_n$，$\Sigma_2=\hat{\Sigma}_n$。于是 $J_{ini}$ 变成： $$J_{ini}(\mu_w,\Sigma_w)=\sum_{n=1}^N\frac{1}{2}(\log|\hat{\Sigma}_n|-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|-O+\text{Tr}(\hat{\Sigma}_n^{-1}\Theta(s_n)^T\Sigma_w\Theta(s_n))+(\Theta(s_n)^T\mu_w-\hat{\mu}_n)^T\hat{\Sigma}_n^{-1}(\Theta(s_n)^T\mu_w-\hat{\mu}_n))$$

这个公式看着复杂，但能拆开。去掉常数项 $\frac{1}{2}$、$\log|\hat{\Sigma}_n|$ 和 $O$（因为不影响优化），分成两部分：

- **均值子问题**：$J_{ini}(\mu_w)=\sum_{n=1}^N(\Theta(s_n)^T\mu_w-\hat{\mu}_n)^T\hat{\Sigma}_n^{-1}(\Theta(s_n)^T\mu_w-\hat{\mu}_n)$
- 这是“预测均值”和“参考均值”的加权差距，$\hat{\Sigma}_n^{-1}$ 是权重（不确定性小的点更重要）。
- **协方差子问题**：$J_{ini}(\Sigma_w)=\sum_{n=1}^N(-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|+\text{Tr}(\hat{\Sigma}_n^{-1}\Theta(s_n)^T\Sigma_w\Theta(s_n)))$
- 这是让预测协方差接近参考协方差。

### 小结

1. **KL 散度的作用**

   - 衡量 $P_p(\xi|s_n)$ 和 $P_r(\xi|s_n)$ 的差距，最小化它让预测轨迹像示范轨迹。
   - 信息论角度：少丢信息，模仿更准。

2. **目标函数**

   - $J_{ini}(\mu_w,\Sigma_w)=\sum_{n=1}^ND_{KL}(P_p(\xi|s_n)||P_r(\xi|s_n))$。
   - 对 $N$ 个点求和，每个点算一次分布差距。

3. **两个分布**

   - $P_p(\xi|s_n)=\mathcal{N}(\Theta(s_n)^T\mu_w,\Theta(s_n)^T\Sigma_w\Theta(s_n))$：预测轨迹。
   - $P_r(\xi|s_n)=\mathcal{N}(\hat{\mu}_n,\hat{\Sigma}_n)$：参考轨迹。

4. **KL 散度公式**

   - $D_{KL}=\int P_p\log\frac{P_p}{P_r}d\xi$，高斯分布有解析解。
   - 代入后得 (9)，包含均值差和协方差差。

5. **拆分成子问题**
   - 均值：$J_{ini}(\mu_w)$，优化 $\mu_w$ 让预测均值接近 $\hat{\mu}_n$。
   - 协方差：$J_{ini}(\Sigma_w)$，优化 $\Sigma_w$ 让预测抖动接近 $\hat{\Sigma}_n$。

## **2.2.2 节：KMP 的均值预测**。

在上一节的均值子问题基础上加了个惩罚项，推导出最优解，然后用核方法简化计算，最终得到 KMP 的均值预测公式。

“我们不像核岭回归（KRR）那样直接优化，而是加了个惩罚项 $||\mu_w||^2$，避免过拟合。”
啥是过拟合？想象你教机器人画线，示范是“差不多直”，但它学得太死，连小抖动都模仿，结果新输入时画歪了。惩罚项就像“别太认真”，让模型简单点。
于是
新的均值子问题写成：$J(\mu_w)=\sum_{n=1}^N(\Theta(s_n)^T\mu_w-\hat{\mu}_n)^T\hat{\Sigma}_n^{-1}(\Theta(s_n)^T\mu_w-\hat{\mu}_n)+\lambda\mu_w^T\mu_w$
这里 $\lambda>0$ 是惩罚力度。

这个 $J(\mu_w)$ 像“加权最小二乘”，但多了 $\lambda\mu_w^T\mu_w$。第一部分是预测均值 $\Theta(s_n)^T\mu_w$ 和参考均值 $\hat{\mu}_n$ 的差距，$\hat{\Sigma}_n^{-1}$ 是权重（抖动小的点更重要）；第二部分是惩罚，让 $\mu_w$ 别太大。跟 KRR 比，KRR 假设 $\hat{\Sigma}_n^{-1}=I_O$（单位矩阵），没用示范的协方差。而这里 $\hat{\Sigma}_n$ 反映了示范的随机性，比如抖动大的点可以偏离多点，抖动小的得贴近。

用了 KRR 的“对偶变换”，推导出最优解：$\mu_w^*=\Phi(\Phi^T\Phi+\lambda\Sigma)^{-1}\mu$
这里：$\Phi=[\Theta(s_1)\Theta(s_2)\cdots\Theta(s_N)]$，是个大矩阵，把所有点的模板拼起来；$\Sigma=\text{blockdiag}(\hat{\Sigma}_1,\hat{\Sigma}_2,\ldots,\hat{\Sigma}_N)$，是对角块矩阵，装所有协方差；
$\mu=[\hat{\mu}_1^T\hat{\mu}_2^T\cdots\hat{\mu}_N^T]^T$，是所有参考均值叠起来。

> [!abstract]
> 对偶变换是凸优化中的核心工具，通过拉格朗日函数将原始问题转化为对偶问题，提供了计算和理论上的双重优势。弱对偶性总是成立，而强对偶性依赖于凸性和约束资格条件（如Slater条件）。当强对偶性成立时，最优解就会满足 KKT 条件。在机器学习中（如SVM），对偶变换不仅是优化手段，还带来了核方法等强大工具。

作者`直接`给出了对偶变换的结果，下面`补充`一下具体过程

> [!note] 对偶变换过程
>
> 我们的任务是找到 $\boldsymbol{\mu}_w$，使 $J(\boldsymbol{\mu}_w)$ 最小化。下面一步步推导：
>
> 1.  展开损失函数
>
> 先把损失函数写得更清楚：
>
> $$ J(\boldsymbol{\mu}_w)=\sum_{n=1}^N(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)^\top\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)+\lambda\boldsymbol{\mu}\_w^\top\boldsymbol{\mu}\_w $$
>
> - 第一项是所有数据点 $n=1$ 到 $N$ 的误差平方和，用协方差逆加权。
> - 第二项是 $\boldsymbol{\mu}_w$ 的 L 2 范数的 квадрат，乘以 $\lambda$。
>
> 为了求最小值，我们需要对 $\boldsymbol{\mu}_w$ 求导，然后令导数等于零。
>
> 2.  对 $\boldsymbol{\mu}_w$ 求导
>
> 我们分开计算两部分的导数：
>
> - 第一项的导数：  
>   定义误差向量 $\mathbf{e}_n=\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w-\hat{\boldsymbol{\mu}}_n$。  
>   那么第一项是 $\sum_{n=1}^N\mathbf{e}_n^\top\hat{\boldsymbol{\Sigma}}_n^{-1}\mathbf{e}_n$。  
>   对 $\boldsymbol{\mu}_w$ 求导，矩阵微分的规则告诉我们：  
>   $$ \frac{\partial}{\partial\boldsymbol{\mu}_w}(\mathbf{e}\_n^\top\hat{\boldsymbol{\Sigma}}\_n^{-1}\mathbf{e}\_n)=2\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n) $$  
>   （推导细节：用链式法则，$\mathbf{e}_n$ 对 $\boldsymbol{\mu}_w$ 的导数是 $\boldsymbol{\Theta}(\mathbf{s}_n)$，$\hat{\boldsymbol{\Sigma}}_n^{-1}$ 是对称矩阵，所以结果乘以 2。）  
>   对所有 $N$ 个数据点求和：  
>   $$ \frac{\partial}{\partial\boldsymbol{\mu}\_w}\sum_{n=1}^N\mathbf{e}_n^\top\hat{\boldsymbol{\Sigma}}\_n^{-1}\mathbf{e}\_n=2\sum_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n) $$
> - 第二项的导数：  
>   $$ \frac{\partial}{\partial\boldsymbol{\mu}\_w}(\lambda\boldsymbol{\mu}\_w^\top\boldsymbol{\mu}\_w)=2\lambda\boldsymbol{\mu}\_w $$  
>   （因为 $\boldsymbol{\mu}_w^\top\boldsymbol{\mu}_w$ 是标量，对向量 $\boldsymbol{\mu}_w$ 求导，结果是 $2\boldsymbol{\mu}_w$，再乘以 $\lambda$。）
>
> 总导数是两部分之和：  
> $$ \frac{\partial J}{\partial\boldsymbol{\mu}_w}=2\sum_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)+2\lambda\boldsymbol{\mu}\_w $$
>
> 3.  令导数等于零
>
> 为了找到极值点，令导数为零：  
> $$ 2\sum\_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)+2\lambda\boldsymbol{\mu}\_w=0 $$
>
> 两边除以 2：  
> $$ \sum\_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)+\lambda\boldsymbol{\mu}\_w=0 $$
>
> 4.  整理矩阵形式
>
> 把求和展开：  
> $$ \sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\hat{\boldsymbol{\mu}}\_n+\lambda\boldsymbol{\mu}\_w=0 $$
>
> 把含 $\boldsymbol{\mu}_w$ 的项移到一边：  
> $$ \sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w+\lambda\boldsymbol{\mu}\_w=\sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\hat{\boldsymbol{\mu}}\_n $$
>
> 提取 $\boldsymbol{\mu}_w$：  
> $$ \left(\sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\boldsymbol{\Theta}(\mathbf{s}\_n)^\top+\lambda\mathbf{I}\right)\boldsymbol{\mu}\_w=\sum*{n=1}^N\boldsymbol{\Theta}(\mathbf{s}\_n)\hat{\boldsymbol{\Sigma}}\_n^{-1}\hat{\boldsymbol{\mu}}\_n $$  
> （这里 $\mathbf{I}$ 是单位矩阵，维度与 $\boldsymbol{\mu}_w$ 一致，即 $B\mathcal{O}\times B\mathcal{O}$。）
>
> 5.  定义矩阵符号
>
> 文章中定义了：
>
> - $\boldsymbol{\Phi}=[\boldsymbol{\Theta}(\mathbf{s}_1),\boldsymbol{\Theta}(\mathbf{s}_2),\ldots,\boldsymbol{\Theta}(\mathbf{s}_N)]$，形状是 $B\mathcal{O}\times N\mathcal{O}$。
> - $\boldsymbol{\Sigma}$ 是一个对角块矩阵，形状是 $N\mathcal{O}\times N\mathcal{O}$，其逆为 $\boldsymbol{\Sigma}^{-1}$，由 $\hat{\boldsymbol{\Sigma}}_1^{-1},\hat{\boldsymbol{\Sigma}}_2^{-1},\ldots,\hat{\boldsymbol{\Sigma}}_N^{-1}$ 组成。
> - $\boldsymbol{\mu}=[\hat{\boldsymbol{\mu}}_1^\top,\hat{\boldsymbol{\mu}}_2^\top,\ldots,\hat{\boldsymbol{\mu}}_N^\top]^\top$，维度是 $N\mathcal{O}$。  
>   （$\mathcal{O}$ 是输出维度，$N$ 是数据点数。）
>
> 用这些符号重写：
>
> - 左边第一项：$\sum_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}_n)\hat{\boldsymbol{\Sigma}}_n^{-1}\boldsymbol{\Theta}(\mathbf{s}_n)^\top=\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Phi}^\top$。  
>   （证明：$\boldsymbol{\Sigma}^{-1}$ 是对角块形式，矩阵乘法正好对应求和形式。）
> - 右边：$\sum_{n=1}^N\boldsymbol{\Theta}(\mathbf{s}_n)\hat{\boldsymbol{\Sigma}}_n^{-1}\hat{\boldsymbol{\mu}}_n=\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$。  
>   （因为 $\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$ 将 $\hat{\boldsymbol{\mu}}_n$ 按 $\hat{\boldsymbol{\Sigma}}_n^{-1}$ 加权，再与 $\boldsymbol{\Phi}$ 相乘。）
>
> 所以方程变成：  
> $$ (\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Phi}^\top+\lambda\mathbf{I})\boldsymbol{\mu}\_w=\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} $$
>
> 6.  解出 $\boldsymbol{\mu}_w^*$
>
> 两边左乘 $(\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Phi}^\top+\lambda\mathbf{I})^{-1}$：  
> $$ \boldsymbol{\mu}\_w=(\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Phi}^\top+\lambda\mathbf{I})^{-1}\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} $$
>
> 这已经是解了，但作者给出的形式是：  
> $$ \boldsymbol{\mu}\_w^\*=\boldsymbol{\Phi}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}+\lambda\boldsymbol{\Sigma})^{-1}\boldsymbol{\mu} $$
>
> 这两种形式不同，原因是对偶变换的应用。直接解是“原始形式”，而作者用的是“对偶形式”。我们需要从对偶角度重新推导。
>
> 7.  对偶变换推导
>
> KRR 的精髓是对偶变换，将问题从高维参数空间（$\boldsymbol{\mu}_w$ 的 $B\mathcal{O}$ 维）转为数据空间（$N\mathcal{O}$ 维）。我们回到损失函数：  
> $$ J(\boldsymbol{\mu}_w)=\sum_{n=1}^N(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)^\top\hat{\boldsymbol{\Sigma}}\_n^{-1}(\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w-\hat{\boldsymbol{\mu}}\_n)+\lambda\boldsymbol{\mu}\_w^\top\boldsymbol{\mu}\_w $$
>
> 用矩阵形式：  
> $$ J(\boldsymbol{\mu}\_w)=(\boldsymbol{\Phi}^\top\boldsymbol{\mu}\_w-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{\Phi}^\top\boldsymbol{\mu}\_w-\boldsymbol{\mu})+\lambda\boldsymbol{\mu}\_w^\top\boldsymbol{\mu}\_w $$
>
> 假设 $\boldsymbol{\mu}_w$ 可以表示为 $\boldsymbol{\Phi}$ 的线性组合（对偶假设）：  
> $$ \boldsymbol{\mu}\_w=\boldsymbol{\Phi}\boldsymbol{\alpha} $$  
> 其中 $\boldsymbol{\alpha}$ 是对偶变量，维度是 $N\mathcal{O}$。代入：  
> $$ J(\boldsymbol{\alpha})=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu})+\lambda\boldsymbol{\alpha}^\top\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha} $$
>
> 对 $\boldsymbol{\alpha}$ 求导：  
> $$ \frac{\partial J}{\partial\boldsymbol{\alpha}}=2\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu})+2\lambda\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}=0 $$
>
> 整理：  
> $$ \boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu})+\lambda\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}=0 $$
>
> 两边左乘 $(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}$（假设可逆，或用伪逆）：  
> $$ \boldsymbol{\Sigma}^{-1}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu})+\lambda\boldsymbol{\alpha}=0 $$
>
> $$ \boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}-\boldsymbol{\mu}+\lambda\boldsymbol{\Sigma}\boldsymbol{\alpha}=0 $$
>
> $$ \boldsymbol{\Phi}^\top\boldsymbol{\Phi}\boldsymbol{\alpha}+\lambda\boldsymbol{\Sigma}\boldsymbol{\alpha}=\boldsymbol{\mu} $$
>
> $$ (\boldsymbol{\Phi}^\top\boldsymbol{\Phi}+\lambda\boldsymbol{\Sigma})\boldsymbol{\alpha}=\boldsymbol{\mu} $$
>
> $$ \boldsymbol{\alpha}=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}+\lambda\boldsymbol{\Sigma})^{-1}\boldsymbol{\mu} $$
>
> 所以：  
> $$ \boldsymbol{\mu}\_w^\*=\boldsymbol{\Phi}\boldsymbol{\alpha}=\boldsymbol{\Phi}(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}+\lambda\boldsymbol{\Sigma})^{-1}\boldsymbol{\mu} $$

有了 $\mu_w^*$，给个新输入 $s^*$（比如新时间），预测均值是：$\mathbb{E}(\xi(s^*))=\Theta(s^*)^T\mu_w^*=\Theta(s^*)^T\Phi(\Phi^T\Phi+\lambda\Sigma)^{-1}\mu$
这能算，但有个问题：$\Theta(s)$ 靠基函数 $\phi(s)$，高维输入（比如位置+角度）时，设计 $\phi(s)$ 很麻烦。

于是，作者提出 `“核化”` 它，避免显式定义基函数。
定义基函数的内积：$\phi(s_i)^T\phi(s_j)=k(s_i,s_j)$，$k(\cdot,\cdot)$ 是核函数（比如高斯核）。
根据 $\Theta(s)$ 的结构 (3)，$\Theta(s_i)^T\Theta(s_j)=\begin{bmatrix}k(s_i,s_j)&0&\cdots&0\\0&k(s_i,s_j)&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&k(s_i,s_j)\end{bmatrix}$
这可以写成：$k(s_i,s_j)=\Theta(s_i)^T\Theta(s_j)=k(s_i,s_j)I_O$，$I_O$ 是 $O$ 维单位矩阵。

再定义两个矩阵：$K=\begin{bmatrix}k(s_1,s_1)&k(s_1,s_2)&\cdots&k(s_1,s_N)\\k(s_2,s_1)&k(s_2,s_2)&\cdots&k(s_2,s_N)\\\vdots&\vdots&\ddots&\vdots\\k(s_N,s_1)&k(s_N,s_2)&\cdots&k(s_N,s_N)\end{bmatrix}$，是所有点对的核矩阵；

$k^*=[k(s^*,s_1)k(s^*,s_2)\cdots k(s^*,s_N)]$，是新点和训练点的核向量。

于是，(15) 变成：$\mathbb{E}(\xi(s^*))=k^*(K+\lambda\Sigma)^{-1}\mu$
这不用 $\Theta(s)$，直接用核函数算，简单多了。

### 小结

1. **加惩罚项**

   - $J(\mu_w)=\sum_{n=1}^N(\Theta(s_n)^T\mu_w-\hat{\mu}_n)^T\hat{\Sigma}_n^{-1}(\Theta(s_n)^T\mu_w-\hat{\mu}_n)+\lambda\mu_w^T\mu_w$。
   - 第一部分是误差，第二部分防过拟合。

2. **跟 KRR 的区别**

   - KRR 没用 $\hat{\Sigma}_n$，这里用它加权，抖动大的点宽松，小的点严格。

3. **最优解**

   - $\mu_w^*=\Phi(\Phi^T\Phi+\lambda\Sigma)^{-1}\mu$。
   - $\Phi$ 是模板矩阵，$\Sigma$ 是协方差，$\mu$ 是参考均值。

4. **预测**

   - $\mathbb{E}(\xi(s^*))=\Theta(s^*)^T\Phi(\Phi^T\Phi+\lambda\Sigma)^{-1}\mu$。
   - 新点的均值靠 $\Theta(s^*)$ 算。

5. **核化**
   - 定义 $k(s_i,s_j)=\phi(s_i)^T\phi(s_j)$，$\Theta(s_i)^T\Theta(s_j)=k(s_i,s_j)I_O$。
   - $K$ 是核矩阵，$k^*$ 是核向量，预测变成 $\mathbb{E}(\xi(s^*))=k^*(K+\lambda\Sigma)^{-1}\mu$。

## **2.2.3 节：KMP 的协方差预测**。

“跟均值优化一样，我们在协方差子问题 (11) 加了个惩罚项，限制 $\Theta(s_n)^T\Sigma_w\Theta(s_n)$ 别太大。”
为啥要限制？如果协方差太大，轨迹抖动范围就离谱，预测不靠谱。惩罚项可以用 $\Sigma_w$ 的最大特征值（Rayleigh 商的性质），但为了简单，他们选了个更宽松的 $\text{Tr}(\Sigma_w)$（矩阵迹，对角线和），因为 $\Sigma_w$ 是正定矩阵，迹比最大特征值大。

于是

新目标函数是：$J(\Sigma_w)=\sum_{n=1}^N(-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|+\text{Tr}(\hat{\Sigma}_n^{-1}\Theta(s_n)^T\Sigma_w\Theta(s_n)))+\lambda\text{Tr}(\Sigma_w)$

怎么解？对 $\Sigma_w$ 求导并令其为 0。导数用到了矩阵求导公式（注释 †）：

$\frac{\partial|\text{AXB}|}{\partial\text{X}}=|\text{AXB}|(\text{X}^T)^{-1}$ 和 $\frac{\partial}{\partial\text{X}}\text{Tr}(\text{AXB})=\text{A}^T\text{B}^T$。

> [!note] 第一项求导过程
>
> #### **对第一块求导：$\sum_{n=1}^N-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|$**
>
> 这块有“行列式”（determinant，用 $|\cdot|$ 表示）。行列式是个数字，比如 $2\times2$ 矩阵 $\begin{bmatrix}a&b\\c&d\end{bmatrix}$ 的行列式是 $ad-bc$。这里是 $\Theta(s_n)^T\Sigma_w\Theta(s_n)$ 的行列式，记作 $A_n=\Theta(s_n)^T\Sigma_w\Theta(s_n)$，它是 $O\times O$ 的矩阵（$O$ 是输出维度，比如位置是 3 维）。
>
> - **为啥有负号和对数？**  
>   $-\log|A_n|$ 是想让 $|A_n|$ 尽量大（因为负对数越小，$|A_n|$ 越大），但不能无限大，后面的惩罚项会限制它。
> - **导数咋算？**  
>   文章注释 † 给了公式：$\frac{\partial|\text{AXB}|}{\partial\text{X}}=|\text{AXB}|(\text{X}^T)^{-1}$。这里 $\text{A}=\Theta(s_n)^T$，$\text{X}=\Sigma_w$，$\text{B}=\Theta(s_n)$，所以：
>
> $\frac{\partial|\Theta(s_n)^T\Sigma_w\Theta(s_n)|}{\partial\Sigma_w}=|\Theta(s_n)^T\Sigma_w\Theta(s_n)|(\Sigma_w^T)^{-1}$
>
> 因为 $\Sigma_w$ 是对称矩阵（协方差矩阵的性质），$\Sigma_w^T=\Sigma_w$，所以：
>
> $\frac{\partial|A_n|}{\partial\Sigma_w}=|A_n|(\Sigma_w)^{-1}$
>
> 现在外面还有 $-\log$，用链式法则：
>
> $\frac{\partial(-\log|A_n|)}{\partial\Sigma_w}=\frac{\partial(-\log|A_n|)}{\partial|A_n|}\cdot\frac{\partial|A_n|}{\partial\Sigma_w}$
>
> 对数的导数是：$\frac{\partial(-\log x)}{\partial x}=-\frac{1}{x}$，所以：
>
> $\frac{\partial(-\log|A_n|)}{\partial|A_n|}=-\frac{1}{|A_n|}$
>
> 代入：
>
> $\frac{\partial(-\log|A_n|)}{\partial\Sigma_w}=-\frac{1}{|A_n|}\cdot|A_n|\Sigma_w^{-1}=-\Sigma_w^{-1}$
>
> 对 $N$ 个点求和：
>
> $\frac{\partial}{\partial\Sigma_w}\sum_{n=1}^N-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|=\sum_{n=1}^N-\Sigma_w^{-1}$
>
> **通俗理解**：这一项像在说“别让 $\Sigma_w$ 太小”，因为 $\Sigma_w^{-1}$ 是倒数，$\Sigma_w$ 小了它就大了，导数变负，推动 $\Sigma_w$ 变大。

计算后得：
$\sum_{n=1}^N(-\Sigma_w^{-1}+\Theta(s_n)\hat{\Sigma}_n^{-1}\Theta(s_n)^T)+\lambda\text{I}=0$

整理成紧凑形式，用 2.2.2 节的 $\Phi$ 和 $\Sigma$（公式 (14)）

解出：$\Sigma_w^*=N(\Phi\Sigma^{-1}\Phi^T+\lambda\text{I})^{-1}$

这像加权最小二乘的协方差，但多了 $N$ 和正则项 $\lambda\text{I}$。

新点 $s^*$ 的协方差是：$D(\xi(s^*))=\Theta(s^*)^T\Sigma_w^*\Theta(s^*)$。代入 (24)，得：$D(\xi(s^*))=N\Theta(s^*)^T(\Phi\Sigma^{-1}\Phi^T+\lambda\text{I})^{-1}\Theta(s^*)$。

用 Woodbury 恒等式（注释 ‡）：$(\text{A}+\text{CB}\text{C}^T)^{-1}=\text{A}^{-1}-\text{A}^{-1}\text{C}(\text{B}^{-1}+\text{C}^T\text{A}^{-1}\text{C})^{-1}\text{C}^T\text{A}^{-1}$

化简成：$D(\xi(s^*))=\frac{N}{\lambda}\Theta(s^*)^T(\text{I}-\Phi(\Phi^T\Phi+\lambda\Sigma)^{-1}\Phi^T)\Theta(s^*)$，编号是 (25)。

再用核化方法。

回忆 2.2.2 节的核定义：$\Theta(s_i)^T\Theta(s_j)=k(s_i,s_j)I_O$ (18)，$K$ 是核矩阵 (19)，$k^*$ 是核向量 (20)。

于是，$\Theta(s^*)^T\Theta(s^*)=k(s^*,s^*)I_O$，$\Phi^T\Theta(s^*)=[k(s_1,s^*)I_O\cdots k(s_N,s^*)I_O]^T=k^{*T}$（注意 $k^*$ 是行向量，矩阵乘法时转置）。

代入 (25)，得：$D(\xi(s^*))=\frac{N}{\lambda}(k(s^*,s^*)-k^*(K+\lambda\Sigma)^{-1}k^{*T})$， (26)。

这不用显式基函数，直接用核函数算。

跟 GPR 和 CrKR 比，(26) 有啥不同？

有两点
一是 $(K+\lambda\Sigma)^{-1}$ 用示范的 $\Sigma$（公式 (14)），GPR 用单位矩阵，CrKR 用对角矩阵；
二是 KMP 预测完整的协方差矩阵，能捕捉输出维度间的关系（比如 x 和 y 的相关性），GPR 和 CrKR 只给对角线协方差。

作者把 $\{s_n,\hat{\mu}_n,\hat{\Sigma}_n\}_{n=1}^N$ 叫“参考数据库” $D$，把均值和协方差预测总结成算法 1：

**算法 1：核化运动基元**

1. **初始化**
   - 定义核函数 $k(\cdot,\cdot)$，设 $\lambda$。
2. **从示范中学习（见 2.1 节）**
   - 收集示范 $\{s_{n,h},\xi_{n,h}\}_{n=1}^N\}_{h=1}^H$。
   - 提取参考数据库 $\{s_n,\hat{\mu}_n,\hat{\Sigma}_n\}_{n=1}^N$。
3. **用 KMP 预测（见 2.2 节）**
   - 输入：新点 $s^*$。
   - 计算 $\Sigma$、$\mu$（公式 (14)），$K$（公式 (19)），$k^*$（公式 (20)）。
   - 输出：均值 $\mathbb{E}(\xi(s^*))=k^*(K+\lambda\Sigma)^{-1}\mu$，协方差 $D(\xi(s^*))=\frac{N}{\lambda}(k(s^*,s^*)-k^*(K+\lambda\Sigma)^{-1}k^{*T})$。

### 小结

1. **加惩罚项**

   - $J(\Sigma_w)=\sum_{n=1}^N(-\log|\Theta(s_n)^T\Sigma_w\Theta(s_n)|+\text{Tr}(\hat{\Sigma}_n^{-1}\Theta(s_n)^T\Sigma_w\Theta(s_n)))+\lambda\text{Tr}(\Sigma_w)$。
   - $\lambda\text{Tr}(\Sigma_w)$ 限制协方差别太大。

2. **求导解最优解**

   - $\frac{\partial J}{\partial\Sigma_w}=\sum_{n=1}^N(-\Sigma_w^{-1}+\Theta(s_n)\hat{\Sigma}_n^{-1}\Theta(s_n)^T)+\lambda\text{I}=0$。
   - $\Sigma_w^*=N(\Phi\Sigma^{-1}\Phi^T+\lambda\text{I})^{-1}$。

3. **新点协方差**

   - $D(\xi(s^*))=N\Theta(s^*)^T(\Phi\Sigma^{-1}\Phi^T+\lambda\text{I})^{-1}\Theta(s^*)$。
   - 用 Woodbury 化简，再核化成：$D(\xi(s^*))=\frac{N}{\lambda}(k(s^*,s^*)-k^*(K+\lambda\Sigma)^{-1}k^{*T})$。

4. **跟 GPR/CrKR 对比**

   - KMP 用 $\Sigma$ 加权，预测完整协方差；GPR/CrKR 简化权重，协方差对角。

5. **算法总结**
   - 输入示范，输出新点的均值和协方差。

## 第 3.1 节：使用 KMP 进行轨迹调制

> [!note]+ 3.1 节的目的
> 通过扩展参考数据库，调整轨迹使其通过新途径点或终点，并通过更新规则解决冲突。推导过程表明，公式 31 与原始损失函数形式相同，可以直接套用 KMP 的优化方法求解最优轨迹。

问题定义

我们需要调整轨迹，使其通过 $M$ 个新的期望点，这些点定义为 $\{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\xi}}_m \}_{m=1}^M$，每个点关联一个条件概率分布：

$$ \bar{\boldsymbol{\xi}}\_m | \bar{\mathbf{s}}\_m \sim \mathcal{N}(\bar{\boldsymbol{\mu}}\_m, \bar{\boldsymbol{\Sigma}}\_m) $$

这些条件分布可以根据任务需求设计：

- 如果机器人需要高精度通过某个途径点，可以设置较小的协方差 $\bar{\boldsymbol{\Sigma}}_m$。
- 如果允许较大的跟踪误差，可以设置较大的协方差 $\bar{\boldsymbol{\Sigma}}_m$。

目标是同时考虑原始参考轨迹分布和新的期望点，重新定义损失函数。原始损失函数（公式 5）是：

$$ J(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}\_w) = \sum_{n=1}^N D\_{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}\_n) \| P_r(\boldsymbol{\xi}|\mathbf{s}\_n)) $$

其中：

- $P_p(\boldsymbol{\xi}|\mathbf{s}_n) = \mathcal{N}(\boldsymbol{\xi} | \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w, \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n))$ 是预测分布。
- $P_r(\boldsymbol{\xi}|\mathbf{s}_n) = \mathcal{N}(\boldsymbol{\xi} | \hat{\boldsymbol{\mu}}_n, \hat{\boldsymbol{\Sigma}}_n)$ 是参考分布。

现在，我们引入新的期望点，重新定义损失函数（公式 27）为：

$$ J*{\text{Uini}}(\boldsymbol{\mu}\_w, \boldsymbol{\Sigma}\_w) = \sum*{n=1}^N D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}\_n) \| P_r(\boldsymbol{\xi}|\mathbf{s}\_n)) + \sum*{m=1}^M D\_{\text{KL}}(P_p(\boldsymbol{\xi}|\bar{\mathbf{s}}\_m) \| P_d(\boldsymbol{\xi}|\bar{\mathbf{s}}\_m)) $$

其中：

- $P_p(\boldsymbol{\xi}|\bar{\mathbf{s}}_m) = \mathcal{N}(\boldsymbol{\xi} | \boldsymbol{\Theta}(\bar{\mathbf{s}}_m)^\top\boldsymbol{\mu}_w, \boldsymbol{\Theta}(\bar{\mathbf{s}}_m)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\bar{\mathbf{s}}_m))$ （公式 28）
- $P_d(\boldsymbol{\xi}|\bar{\mathbf{s}}_m) = \mathcal{N}(\boldsymbol{\xi} | \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m)$ （公式 29）

方法：扩展参考数据库

为了统一处理原始参考数据库 $D = \{ \mathbf{s}_n, \hat{\boldsymbol{\mu}}_n, \hat{\boldsymbol{\Sigma}}_n \}_{n=1}^N$ 和期望数据库 $\bar{D} = \{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m \}_{m=1}^M$，我们将两者拼接，生成一个扩展的参考数据库 $\{ \mathbf{s}_{\text{U}i}, \boldsymbol{\mu}_{\text{U}i}, \boldsymbol{\Sigma}_{\text{U}i} \}_{i=1}^{N+M}$，定义如下（公式 30）：

$$ \begin{cases} \mathbf{s}_{\text{U}i} = \mathbf{s}\_i, \quad \boldsymbol{\mu}_{\text{U}i} = \hat{\boldsymbol{\mu}}_i, \quad \boldsymbol{\Sigma}_{\text{U}i} = \hat{\boldsymbol{\Sigma}}_i, & \text{if } 1 \leq i \leq N \\ \mathbf{s}_{\text{U}i} = \bar{\mathbf{s}}_{i-N}, \quad \boldsymbol{\mu}_{\text{U}i} = \bar{\boldsymbol{\mu}}_{i-N}, \quad \boldsymbol{\Sigma}_{\text{U}i} = \bar{\boldsymbol{\Sigma}}\_{i-N}, & \text{if } N < i \leq N+M \end{cases} $$

使用扩展数据库，损失函数（公式 27）可以重写为（公式 31）：

$$ J*{\text{Uini}}(\boldsymbol{\mu}\_w, \boldsymbol{\Sigma}\_w) = \sum*{i=1}^{N+M} D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) \| P*u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i})) $$

其中：

- $P_p(\boldsymbol{\xi}|\mathbf{s}_{\text{U}i}) = \mathcal{N}(\boldsymbol{\xi} | \boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top\boldsymbol{\mu}_w, \boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_{\text{U}i}))$ （公式 32）
- $P_u(\boldsymbol{\xi}|\mathbf{s}_{\text{U}i}) = \mathcal{N}(\boldsymbol{\xi} | \boldsymbol{\mu}_{\text{U}i}, \boldsymbol{\Sigma}_{\text{U}i})$ （公式 33）

从公式 27 到公式 31 的推导

我们来一步步推导如何从公式 27 得到公式 31。

1. **公式 27 的展开**：  
   公式 27 包含两部分：

   - 第一部分是原始参考数据库的 KL 散度：$\sum_{n=1}^N D_{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}_n) \| P_r(\boldsymbol{\xi}|\mathbf{s}_n))$。
   - 第二部分是新期望点的 KL 散度：$\sum_{m=1}^M D_{\text{KL}}(P_p(\boldsymbol{\xi}|\bar{\mathbf{s}}_m) \| P_d(\boldsymbol{\xi}|\bar{\mathbf{s}}_m))$。

   根据公式 30 的定义：

   - 当 $i$ 从 1 到 $N$ 时，$\mathbf{s}_{\text{U}i} = \mathbf{s}_i$，$\boldsymbol{\mu}_{\text{U}i} = \hat{\boldsymbol{\mu}}_i$，$\boldsymbol{\Sigma}_{\text{U}i} = \hat{\boldsymbol{\Sigma}}_i$，因此：
     $$ P*p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) = P*p(\boldsymbol{\xi}|\mathbf{s}\_i), \quad P_u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) = P_r(\boldsymbol{\xi}|\mathbf{s}\_i) $$
     对应第一部分的求和。
   - 当 $i$ 从 $N+1$ 到 $N+M$ 时，$\mathbf{s}_{\text{U}i} = \bar{\mathbf{s}}_{i-N}$，$\boldsymbol{\mu}_{\text{U}i} = \bar{\boldsymbol{\mu}}_{i-N}$，$\boldsymbol{\Sigma}_{\text{U}i} = \bar{\boldsymbol{\Sigma}}_{i-N}$，因此：
     $$ P*p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) = P*p(\boldsymbol{\xi}|\bar{\mathbf{s}}*{i-N}), \quad P*u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) = P*d(\boldsymbol{\xi}|\bar{\mathbf{s}}*{i-N}) $$
     对应第二部分的求和。

2. **合并求和**：  
   将两部分求和合并为一个统一的求和形式：

   - 第一部分求和对应 $i$ 从 1 到 $N$。
   - 第二部分求和对应 $i$ 从 $N+1$ 到 $N+M$，只需将下标 $m = i-N$ 转换回 $m$。

   因此，公式 27 可以写为：

   $$ J*{\text{Uini}}(\boldsymbol{\mu}\_w, \boldsymbol{\Sigma}\_w) = \sum*{i=1}^N D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) \| P*u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i})) + \sum*{i=N+1}^{N+M} D*{\text{KL}}(P*p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) \| P*u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i})) $$

   合并后即为：

   $$ J*{\text{Uini}}(\boldsymbol{\mu}\_w, \boldsymbol{\Sigma}\_w) = \sum*{i=1}^{N+M} D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i}) \| P*u(\boldsymbol{\xi}|\mathbf{s}*{\text{U}i})) $$

   这就是公式 31。

求解最优解

公式 31 的形式与原始损失函数（公式 5）相同，因此我们可以直接套用第 2.2.2 节的推导方法求解最优解 $\boldsymbol{\mu}_w^*$ 和 $\boldsymbol{\Sigma}_w^*$。

1. **定义扩展矩阵**：  
   根据扩展数据库，定义：

   - $\boldsymbol{\Phi}_{\text{U}} = [\boldsymbol{\Theta}(\mathbf{s}_{\text{U}1}), \boldsymbol{\Theta}(\mathbf{s}_{\text{U}2}), \ldots, \boldsymbol{\Theta}(\mathbf{s}_{\text{U},N+M})]$，形状为 $B\mathcal{O} \times (N+M)\mathcal{O}$。
   - $\boldsymbol{\Sigma}_{\text{U}}$ 是一个对角块矩阵，形状为 $(N+M)\mathcal{O} \times (N+M)\mathcal{O}$，其逆 $\boldsymbol{\Sigma}_{\text{U}}^{-1}$ 由 $\boldsymbol{\Sigma}_{\text{U}i}^{-1}$ 组成。
   - $\boldsymbol{\mu}_{\text{U}} = [\boldsymbol{\mu}_{\text{U}1}^\top, \boldsymbol{\mu}_{\text{U}2}^\top, \ldots, \boldsymbol{\mu}_{\text{U},N+M}^\top]^\top$，维度为 $(N+M)\mathcal{O}$。

2. **损失函数的矩阵形式**：  
   公式 31 的 KL 散度可以展开（参考第 2.2.2 节的推导）。假设 $\boldsymbol{\Sigma}_w$ 已知（通常通过其他方法估计），我们只优化 $\boldsymbol{\mu}_w$，损失函数简化为：

   $$ J*{\text{Uini}}(\boldsymbol{\mu}\_w) = \sum*{i=1}^{N+M} (\boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{U}i})^\top \boldsymbol{\Sigma}_{\text{U}i}^{-1} (\boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top\boldsymbol{\mu}_w - \boldsymbol{\mu}_{\text{U}i}) + \lambda \boldsymbol{\mu}\_w^\top\boldsymbol{\mu}\_w $$

3. **求导并解出 $\boldsymbol{\mu}_w^*$**：  
   对 $\boldsymbol{\mu}_w$ 求导并令导数为零（类似第 2.2.2 节的步骤）：

   $$ \sum*{i=1}^{N+M} \boldsymbol{\Theta}(\mathbf{s}*{\text{U}i}) \boldsymbol{\Sigma}_{\text{U}i}^{-1} (\boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top\boldsymbol{\mu}_w - \boldsymbol{\mu}_{\text{U}i}) + \lambda \boldsymbol{\mu}\_w = 0 $$

   整理后：

   $$ \left( \sum*{i=1}^{N+M} \boldsymbol{\Theta}(\mathbf{s}*{\text{U}i}) \boldsymbol{\Sigma}_{\text{U}i}^{-1} \boldsymbol{\Theta}(\mathbf{s}_{\text{U}i})^\top + \lambda \mathbf{I} \right) \boldsymbol{\mu}_w = \sum_{i=1}^{N+M} \boldsymbol{\Theta}(\mathbf{s}_{\text{U}i}) \boldsymbol{\Sigma}_{\text{U}i}^{-1} \boldsymbol{\mu}\_{\text{U}i} $$

   使用矩阵形式：

   $$ \left( \boldsymbol{\Phi}_{\text{U}} \boldsymbol{\Sigma}_{\text{U}}^{-1} \boldsymbol{\Phi}_{\text{U}}^\top + \lambda \mathbf{I} \right) \boldsymbol{\mu}\_w = \boldsymbol{\Phi}_{\text{U}} \boldsymbol{\Sigma}_{\text{U}}^{-1} \boldsymbol{\mu}_{\text{U}} $$

   解出 $\boldsymbol{\mu}_w^*$：

   $$ \boldsymbol{\mu}_w^\* = \left( \boldsymbol{\Phi}_{\text{U}} \boldsymbol{\Sigma}_{\text{U}}^{-1} \boldsymbol{\Phi}_{\text{U}}^\top + \lambda \mathbf{I} \right)^{-1} \boldsymbol{\Phi}_{\text{U}} \boldsymbol{\Sigma}_{\text{U}}^{-1} \boldsymbol{\mu}\_{\text{U}} $$

   使用对偶形式（参考第 2.2.2 节）：

   $$ \boldsymbol{\mu}_w^\* = \boldsymbol{\Phi}_{\text{U}} (\boldsymbol{\Phi}_{\text{U}}^\top \boldsymbol{\Phi}_{\text{U}} + \lambda \boldsymbol{\Sigma}_{\text{U}})^{-1} \boldsymbol{\mu}_{\text{U}} $$

4. **核化预测**：  
   对于新查询点 $\mathbf{s}^*$，预测均值为：

   $$ \mathbb{E}[\boldsymbol{\xi}(\mathbf{s}^*)] = \boldsymbol{\Theta}(\mathbf{s}^_)^\top \boldsymbol{\mu}\_w^_ = \boldsymbol{\Theta}(\mathbf{s}^\*)^\top \boldsymbol{\Phi}_{\text{U}} (\boldsymbol{\Phi}_{\text{U}}^\top \boldsymbol{\Phi}_{\text{U}} + \lambda \boldsymbol{\Sigma}_{\text{U}})^{-1} \boldsymbol{\mu}\_{\text{U}} $$

   引入核函数：

   - $\boldsymbol{\Phi}_{\text{U}}^\top \boldsymbol{\Phi}_{\text{U}} = \mathbf{K}_{\text{U}}$（核矩阵）。
   - $\boldsymbol{\Theta}(\mathbf{s}^*)^\top \boldsymbol{\Phi}_{\text{U}} = \mathbf{k}_{\text{U}}^*$。

   最终预测公式为：

   $$ \mathbb{E}[\boldsymbol{\xi}(\mathbf{s}^*)] = \mathbf{k}_{\text{U}}^\* (\mathbf{K}_{\text{U}} + \lambda \boldsymbol{\Sigma}_{\text{U}})^{-1} \boldsymbol{\mu}_{\text{U}} $$

冲突处理

扩展数据库可能会导致冲突。例如，如果某个新输入 $\bar{\mathbf{s}}_m = \mathbf{s}_n$，但 $\bar{\boldsymbol{\mu}}_m$ 和 $\hat{\boldsymbol{\mu}}_n$ 相距较远，而 $\bar{\boldsymbol{\Sigma}}_m$ 和 $\hat{\boldsymbol{\Sigma}}_n$ 几乎相同，那么公式 31 的最优解只能在 $\bar{\boldsymbol{\mu}}_m$ 和 $\hat{\boldsymbol{\mu}}_n$ 之间折中。

在轨迹调制中，通常希望新期望点具有最高优先级。因此，作者提出了一种更新参考数据库的方法，减少冲突，同时保留大部分原始数据点。具体更新步骤如下：

- 对于期望数据库中的每个数据点 $\{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m \}$，将其输入 $\bar{\mathbf{s}}_m$ 与参考数据库的输入 $\{ \mathbf{s}_n \}_{n=1}^N$ 比较，找到最近的数据点 $\{ \mathbf{s}_r, \hat{\boldsymbol{\mu}}_r, \hat{\boldsymbol{\Sigma}}_r \}$，满足：

  $$ d(\bar{\mathbf{s}}\_m, \mathbf{s}\_r) \leq d(\bar{\mathbf{s}}\_m, \mathbf{s}\_n), \quad \forall n \in \{1, 2, \ldots, N\} $$

  其中 $d(\cdot)$ 是任意距离度量（如 2 范数）。

- 如果最近距离 $d(\bar{\mathbf{s}}_m, \mathbf{s}_r)$ 小于预定义阈值 $\zeta > 0$，则用 $\{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m \}$ 替换 $\{ \mathbf{s}_r, \hat{\boldsymbol{\mu}}_r, \hat{\boldsymbol{\Sigma}}_r \}$。
- 否则，将 $\{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m \}$ 插入参考数据库。

更新规则（公式 34）为：

$$ \begin{cases} D \leftarrow \{ D / \{ \mathbf{s}\_r, \hat{\boldsymbol{\mu}}\_r, \hat{\boldsymbol{\Sigma}}\_r \} \} \cup \{ \bar{\mathbf{s}}\_m, \bar{\boldsymbol{\mu}}\_m, \bar{\boldsymbol{\Sigma}}\_m \}, & \text{if } d(\bar{\mathbf{s}}\_m, \mathbf{s}\_r) < \zeta \\ D \leftarrow D \cup \{ \bar{\mathbf{s}}\_m, \bar{\boldsymbol{\mu}}\_m, \bar{\boldsymbol{\Sigma}}\_m \}, & \text{otherwise} \end{cases} $$

其中 $r = \arg\min_n d(\bar{\mathbf{s}}_m, \mathbf{s}_n)$，$n \in \{1, 2, \ldots, N\}$，符号 $/$ 和 $\cup$ 分别表示排除和并集操作。

轨迹调制就像在导航中调整路线：你有一条原始路线（参考轨迹），但路上突然出现障碍（新途径点），你需要调整路线经过这些点。KMP 的方法是把新点和原始路线“合并”成一个新的路线图（扩展数据库），然后重新规划路径。如果新点和原始路线有冲突（比如两个点位置很近但目标不同），就优先选择新点（更新规则），确保机器人能按新要求走。

## 3.2 节：使用 KMP 进行轨迹叠加

> [!note]+ 3.2 节的目的
> 第 3.2 节通过加权 KL 散度混合多条轨迹，推导了加权均值和协方差子问题，并将其转化为等价形式，最终使用 KMP 求解混合轨迹。

问题定义

给定 $L$ 条参考轨迹分布，每条轨迹关联一组输入和优先级，记为 $\{ \{ \mathbf{s}_n, \hat{\boldsymbol{\xi}}_{n,l}, \gamma_{n,l} \}_{n=1}^N \}_{l=1}^L$，其中：

- $\hat{\boldsymbol{\xi}}_{n,l} | \mathbf{s}_n \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_{n,l}, \hat{\boldsymbol{\Sigma}}_{n,l})$ 表示第 $l$ 条轨迹在输入 $\mathbf{s}_n$ 处的分布。
- $\gamma_{n,l} \in (0, 1)$ 是优先级，满足 $\sum_{l=1}^L \gamma_{n,l} = 1$。

目标是混合这 $L$ 条轨迹，生成一条新的轨迹，考虑每条轨迹的优先级。作者提出了一种加权损失函数（公式 35）：

$$ J*{\text{Sini}}(\boldsymbol{\mu}\_w, \boldsymbol{\Sigma}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}\_n) \| P*{s,l}(\boldsymbol{\xi}|\mathbf{s}\_n)) $$

其中：

- $P_p(\boldsymbol{\xi}|\mathbf{s}_n) = \mathcal{N}(\boldsymbol{\xi} | \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w, \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n))$ 是预测分布。
- $P_{s,l}(\boldsymbol{\xi}|\mathbf{s}_n) = \mathcal{N}(\boldsymbol{\xi} | \hat{\boldsymbol{\mu}}_{n,l}, \hat{\boldsymbol{\Sigma}}_{n,l})$ 是第 $l$ 条参考轨迹的分布（公式 36）。

分解损失函数：从公式 35 到公式 37 和 38

我们首先推导如何从公式 35 分解出公式 37（加权均值子问题）和公式 38（加权协方差子问题）。

1. **KL 散度的展开**：  
   KL 散度 $D_{\text{KL}}(P_p \| P_{s,l})$ 衡量预测分布和参考分布的差异。对于两个正态分布 $P_p \sim \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$ 和 $P_{s,l} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_{n,l}, \hat{\boldsymbol{\Sigma}}_{n,l})$，KL 散度公式为：

   $$ D*{\text{KL}}(P_p \| P*{s,l}) = \frac{1}{2} \left( \log \frac{|\hat{\boldsymbol{\Sigma}}_{n,l}|}{|\boldsymbol{\Sigma}\_p|} - d + \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}\_p - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\mu}\_p - \hat{\boldsymbol{\mu}}_{n,l}) \right) $$

   其中 $d$ 是输出维度（即 $\mathcal{O}$）。代入：

   - $\boldsymbol{\mu}_p = \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w$，$\boldsymbol{\Sigma}_p = \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n)$。
   - 参考分布均值和协方差为 $\hat{\boldsymbol{\mu}}_{n,l}$ 和 $\hat{\boldsymbol{\Sigma}}_{n,l}$。

   因此：

   $$ D*{\text{KL}}(P_p(\boldsymbol{\xi}|\mathbf{s}\_n) \| P*{s,l}(\boldsymbol{\xi}|\mathbf{s}_n)) = \frac{1}{2} \left( \log \frac{|\hat{\boldsymbol{\Sigma}}_{n,l}|}{|\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)|} - \mathcal{O} + \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) + (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l}) \right) $$

2. **分离 $\boldsymbol{\mu}_w$ 和 $\boldsymbol{\Sigma}_w$**：  
   上述 KL 散度可以分为两部分：

   - 与 $\boldsymbol{\mu}_w$ 相关的项：$(\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}_w - \hat{\boldsymbol{\mu}}_{n,l})$。
   - 与 $\boldsymbol{\Sigma}_w$ 相关的项：$\log \frac{|\hat{\boldsymbol{\Sigma}}_{n,l}|}{|\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n)|} - \mathcal{O} + \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n))$。

   代入公式 35，并忽略与优化无关的常数项（例如 $\log |\hat{\boldsymbol{\Sigma}}_{n,l}|$ 和 $-\mathcal{O}$），损失函数可以分解为：

   - **加权均值子问题（公式 37）**：

     $$ J*{\text{Sini}}(\boldsymbol{\mu}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l}) $$

   - **加权协方差子问题（公式 38）**：

     $$ J*{\text{Sini}}(\boldsymbol{\Sigma}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} \left( -\log |\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)| + \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) \right) $$

从公式 37 和 38 到公式 39 和 40 的推导

接下来，我们推导如何从公式 37 和 38 得到等价的公式 39 和 40。

1. **加权均值子问题（从公式 37 到公式 39）**：  
   公式 37 为：

   $$ J*{\text{Sini}}(\boldsymbol{\mu}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l}) $$

   展开每一项：

   $$ (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l})^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \hat{\boldsymbol{\mu}}_{n,l}) = \boldsymbol{\mu}_w^\top \boldsymbol{\Theta}(\mathbf{s}\_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w - 2 \hat{\boldsymbol{\mu}}_{n,l}^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top \boldsymbol{\mu}\_w + \hat{\boldsymbol{\mu}}_{n,l}^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \hat{\boldsymbol{\mu}}_{n,l} $$

   代入并忽略与 $\boldsymbol{\mu}_w$ 无关的常数项（即 $\hat{\boldsymbol{\mu}}_{n,l}^\top \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \hat{\boldsymbol{\mu}}_{n,l}$），公式 37 可以写为：

   $$ J*{\text{Sini}}(\boldsymbol{\mu}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} \left( \boldsymbol{\mu}_w^\top \boldsymbol{\Theta}(\mathbf{s}\_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w - 2 \hat{\boldsymbol{\mu}}_{n,l}^\top \hat{\boldsymbol{\Sigma}}\_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top \boldsymbol{\mu}\_w \right) $$

   对 $\boldsymbol{\mu}_w$ 求导：

   $$ \frac{\partial J*{\text{Sini}}}{\partial \boldsymbol{\mu}\_w} = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} \left( 2 \boldsymbol{\Theta}(\mathbf{s}_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w - 2 \boldsymbol{\Theta}(\mathbf{s}\_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \hat{\boldsymbol{\mu}}\_{n,l} \right) $$

   令导数为零：

   $$ \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} \boldsymbol{\Theta}(\mathbf{s}\_n) \hat{\boldsymbol{\Sigma}}*{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w = \sum_{n=1}^N \sum*{l=1}^L \gamma*{n,l} \boldsymbol{\Theta}(\mathbf{s}_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \hat{\boldsymbol{\mu}}\_{n,l} $$

   为了将问题转化为等价形式，定义加权均值和协方差：

   - $\boldsymbol{\Sigma}_{\text{S}n}^{-1} = \sum_{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1}$ （公式 41）。
   - $\boldsymbol{\mu}_{\text{S}n} = \boldsymbol{\Sigma}_{\text{S}n} \sum_{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1} \hat{\boldsymbol{\mu}}_{n,l}$ （公式 42）。

   这些定义来源于 $L$ 个高斯分布 $\mathcal{N}(\hat{\boldsymbol{\mu}}_{n,l}, \hat{\boldsymbol{\Sigma}}_{n,l}/\gamma_{n,l})$ 的乘积（公式 43）：

   $$ \mathcal{N}(\boldsymbol{\mu}_{\text{S}n}, \boldsymbol{\Sigma}_{\text{S}n}) \propto \prod*{l=1}^L \mathcal{N}(\hat{\boldsymbol{\mu}}*{n,l}, \hat{\boldsymbol{\Sigma}}_{n,l}/\gamma_{n,l}) $$

   现在，我们构造一个等价损失函数（公式 39）：

   $$ \tilde{J}_{\text{Sini}}(\boldsymbol{\mu}\_w) = \sum_{n=1}^N (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}n})^\top \boldsymbol{\Sigma}_{\text{S}n}^{-1} (\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}n}) $$

   展开并求导：

   $$ \frac{\partial \tilde{J}_{\text{Sini}}}{\partial \boldsymbol{\mu}\_w} = \sum_{n=1}^N 2 \boldsymbol{\Theta}(\mathbf{s}_n) \boldsymbol{\Sigma}_{\text{S}n}^{-1} (\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}n}) $$

   令导数为零：

   $$ \sum*{n=1}^N \boldsymbol{\Theta}(\mathbf{s}\_n) \boldsymbol{\Sigma}*{\text{S}n}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w = \sum_{n=1}^N \boldsymbol{\Theta}(\mathbf{s}_n) \boldsymbol{\Sigma}_{\text{S}n}^{-1} \boldsymbol{\mu}\_{\text{S}n} $$

   代入 $\boldsymbol{\Sigma}_{\text{S}n}^{-1}$ 和 $\boldsymbol{\mu}_{\text{S}n}$ 的定义，右边为：

   $$ \sum*{n=1}^N \boldsymbol{\Theta}(\mathbf{s}\_n) \boldsymbol{\Sigma}*{\text{S}n}^{-1} \boldsymbol{\mu}_{\text{S}n} = \sum_{n=1}^N \boldsymbol{\Theta}(\mathbf{s}_n) \left( \sum_{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1} \right) \left( \boldsymbol{\Sigma}_{\text{S}n} \sum_{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1} \hat{\boldsymbol{\mu}}\_{n,l} \right) $$

   注意到 $\boldsymbol{\Sigma}_{\text{S}n}^{-1} \boldsymbol{\Sigma}_{\text{S}n} = \mathbf{I}$，因此：

   $$ \sum*{n=1}^N \boldsymbol{\Theta}(\mathbf{s}\_n) \sum*{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1} \hat{\boldsymbol{\mu}}_{n,l} = \sum_{n=1}^N \sum*{l=1}^L \gamma*{n,l} \boldsymbol{\Theta}(\mathbf{s}_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \hat{\boldsymbol{\mu}}\_{n,l} $$

   左边为：

   $$ \sum*{n=1}^N \boldsymbol{\Theta}(\mathbf{s}\_n) \boldsymbol{\Sigma}*{\text{S}n}^{-1} \boldsymbol{\Theta}(\mathbf{s}_n)^\top \boldsymbol{\mu}\_w = \sum_{n=1}^N \sum*{l=1}^L \gamma*{n,l} \boldsymbol{\Theta}(\mathbf{s}_n) \hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top \boldsymbol{\mu}\_w $$

   这与公式 37 的导数结果一致，因此公式 39 是公式 37 的等价形式。

2. **加权协方差子问题（从公式 38 到公式 40）**：  
   公式 38 为：

   $$ J*{\text{Sini}}(\boldsymbol{\Sigma}\_w) = \sum*{n=1}^N \sum*{l=1}^L \gamma*{n,l} \left( -\log |\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)| + \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) \right) $$

   注意到 $\boldsymbol{\Sigma}_{\text{S}n}^{-1} = \sum_{l=1}^L \left( \frac{\hat{\boldsymbol{\Sigma}}_{n,l}}{\gamma_{n,l}} \right)^{-1}$，我们构造等价损失函数（公式 40）：

   $$ \tilde{J}_{\text{Sini}}(\boldsymbol{\Sigma}\_w) = \sum_{n=1}^N \left( -\log |\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)| + \text{Tr}(\boldsymbol{\Sigma}_{\text{S}n}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) \right) $$

   由于 $\text{Tr}$ 是线性运算：

   $$ \sum*{l=1}^L \gamma*{n,l} \text{Tr}(\hat{\boldsymbol{\Sigma}}_{n,l}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) = \text{Tr}\left( \left( \sum_{l=1}^L \gamma*{n,l} \hat{\boldsymbol{\Sigma}}*{n,l}^{-1} \right) \boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n) \right) = \text{Tr}(\boldsymbol{\Sigma}_{\text{S}n}^{-1} \boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)) $$

   第一项 $-\log |\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}_w\boldsymbol{\Theta}(\mathbf{s}_n)|$ 不依赖 $l$，因此：

   $$ \sum*{l=1}^L \gamma*{n,l} (-\log |\boldsymbol{\Theta}(\mathbf{s}_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)|) = -\log |\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)| \sum_{l=1}^L \gamma\_{n,l} = -\log |\boldsymbol{\Theta}(\mathbf{s}\_n)^\top\boldsymbol{\Sigma}\_w\boldsymbol{\Theta}(\mathbf{s}\_n)| $$

   因此，公式 38 和公式 40 等价。

求解最优解

公式 39 和 40 与第 2.2.2 节的子问题形式相同，因此可以直接套用 KMP 的优化方法。

1. **加权均值子问题（公式 39）**：  
   定义：

   - $\boldsymbol{\Phi} = [\boldsymbol{\Theta}(\mathbf{s}_1), \ldots, \boldsymbol{\Theta}(\mathbf{s}_N)]$，形状为 $B\mathcal{O} \times N\mathcal{O}$。
   - $\boldsymbol{\Sigma}_{\text{S}}^{-1}$ 是对角块矩阵，由 $\boldsymbol{\Sigma}_{\text{S}n}^{-1}$ 组成，形状为 $N\mathcal{O} \times N\mathcal{O}$。
   - $\boldsymbol{\mu}_{\text{S}} = [\boldsymbol{\mu}_{\text{S}1}^\top, \ldots, \boldsymbol{\mu}_{\text{S}N}^\top]^\top$，维度为 $N\mathcal{O}$。

   公式 39 的矩阵形式为：

   $$ \tilde{J}_{\text{Sini}}(\boldsymbol{\mu}\_w) = (\boldsymbol{\Phi}^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}})^\top \boldsymbol{\Sigma}_{\text{S}}^{-1} (\boldsymbol{\Phi}^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}}) $$

   求导并令导数为零：

   $$ \boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} (\boldsymbol{\Phi}^\top\boldsymbol{\mu}\_w - \boldsymbol{\mu}_{\text{S}}) = 0 $$

   $$ \boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\Phi}^\top \boldsymbol{\mu}\_w = \boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\mu}\_{\text{S}} $$

   加上正则化项 $\lambda \boldsymbol{\mu}_w^\top\boldsymbol{\mu}_w$（如第 2.2.2 节）：

   $$ (\boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\Phi}^\top + \lambda \mathbf{I}) \boldsymbol{\mu}\_w = \boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\mu}\_{\text{S}} $$

   解出 $\boldsymbol{\mu}_w^*$：

   $$ \boldsymbol{\mu}_w^\* = (\boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\Phi}^\top + \lambda \mathbf{I})^{-1} \boldsymbol{\Phi} \boldsymbol{\Sigma}_{\text{S}}^{-1} \boldsymbol{\mu}_{\text{S}} $$

   使用对偶形式：

   $$ \boldsymbol{\mu}_w^\* = \boldsymbol{\Phi} (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \boldsymbol{\Sigma}_{\text{S}})^{-1} \boldsymbol{\mu}\_{\text{S}} $$

2. **加权协方差子问题（公式 40）**：  
   公式 40 与第 2.2.2 节的协方差优化问题相同，可以通过迭代优化或解析方法求解（具体方法在文中未详细展开，通常需要数值优化）。

3. **核化预测**：  
   对于新查询点 $\mathbf{s}^*$，预测均值为：

   $$ \mathbb{E}[\boldsymbol{\xi}(\mathbf{s}^*)] = \mathbf{k}^\* (\mathbf{K} + \lambda \boldsymbol{\Sigma}_{\text{S}})^{-1} \boldsymbol{\mu}_{\text{S}} $$

   其中 $\mathbf{K} = \boldsymbol{\Phi}^\top \boldsymbol{\Phi}$，$\mathbf{k}^*$ 是核向量。

方法总结

轨迹叠加的步骤如下：

1. 通过公式 43 计算混合参考数据库 $\{ \mathbf{s}_n, \boldsymbol{\mu}_{\text{S}n}, \boldsymbol{\Sigma}_{\text{S}n} \}_{n=1}^N$。
2. 使用算法 1（Algorithm 1）预测任意查询点的混合轨迹点。

轨迹叠加就像在做决策：你有几条路可以走（候选轨迹），每条路有不同的“推荐指数”（优先级）。KMP 帮你把这些路“混合”成一条新路，优先考虑推荐指数高的路，但也尽量保留其他路的特点。公式 43 就像把所有路的“建议”加权平均，生成一条综合路线。

## 3.3 节：使用 KMP 进行局部运动学习

> [!note]+ 3.3 节的目的
> 3.3 节通过局部坐标系和仿射变换扩展 KMP，增强了其外推能力。推导过程表明，公式 46 是公式 45 的最优解，表示高斯分布乘积的期望。

问题背景

之前我们讨论的轨迹都是在全局坐标系（基坐标系 $\{O\}$）中表示的。然而，为了增强 KMP 在任务空间中的外推能力，可以将人类示范编码到局部坐标系中，提取局部运动模式，从而应用于更广泛的任务实例。局部坐标系的定义通常取决于具体任务。例如，在运输任务中，机器人需要将物体从起始位置（可能变化）移动到不同的目标位置，可以分别在起始和目标位置定义两个局部坐标系。

方法概述

作者提出了局部 KMP（Local-KMP）方法，通过以下步骤实现：

1. 定义 $P$ 个局部坐标系 $\{ \mathbf{A}^{(p)}, \mathbf{b}^{(p)} \}_{p=1}^P$，其中 $\mathbf{A}^{(p)}$ 是旋转矩阵，$\mathbf{b}^{(p)}$ 是平移向量，表示局部坐标系 $\{p\}$ 相对于基坐标系 $\{O\}$ 的变换。
2. 将人类示范投影到每个局部坐标系中，生成局部参考数据库。
3. 在每个局部坐标系中应用 KMP 学习局部轨迹分布。
4. 对于新查询点，通过局部 KMP 预测局部轨迹点，并将其转换回基坐标系。

局部坐标系变换

定义 $P$ 个局部坐标系 $\{ \mathbf{A}^{(p)}, \mathbf{b}^{(p)} \}_{p=1}^P$。对于每组人类示范 $\{ \{ \mathbf{s}_{n,h}, \boldsymbol{\xi}_{n,h} \}_{n=1}^N \}_{h=1}^H$，将其投影到每个局部坐标系 $\{p\}$ 中，生成新的轨迹点 $\{ \{ \mathbf{s}_{n,h}^{(p)}, \boldsymbol{\xi}_{n,h}^{(p)} \}_{n=1}^N \}_{h=1}^H$，变换公式（公式 44）为：

$$ \begin{bmatrix} \mathbf{s}_{n,h}^{(p)} \\ \boldsymbol{\xi}_{n,h}^{(p)} \end{bmatrix} = \begin{bmatrix} \mathbf{A}^{(p)}_s & \mathbf{0} \\ \mathbf{0} & \mathbf{A}^{(p)}_\xi \end{bmatrix}^{-1} \left( \begin{bmatrix} \mathbf{s}_{n,h} \\ \boldsymbol{\xi}_{n,h} \end{bmatrix} - \begin{bmatrix} \mathbf{b}^{(p)}_s \\ \mathbf{b}^{(p)}_\xi \end{bmatrix} \right) $$

其中 $\mathbf{A}^{(p)}_s = \mathbf{A}^{(p)}_\xi = \mathbf{A}^{(p)}$，$\mathbf{b}^{(p)}_s = \mathbf{b}^{(p)}_\xi = \mathbf{b}^{(p)}$。这表示输入 $\mathbf{s}$ 和输出 $\boldsymbol{\xi}$ 使用相同的旋转和平移变换。

局部参考数据库

在每个局部坐标系 $\{p\}$ 中，根据第 2.1 节的方法，从投影后的示范 $\{ \mathbf{s}_{n,h}^{(p)}, \boldsymbol{\xi}_{n,h}^{(p)} \}$ 中提取局部参考数据库 $D^{(p)} = \{ \mathbf{s}_n^{(p)}, \hat{\boldsymbol{\mu}}_n^{(p)}, \hat{\boldsymbol{\Sigma}}_n^{(p)} \}_{n=1}^N$。

轨迹调制

为了简化讨论，作者只考虑通过途径点或终点的轨迹调制（轨迹叠加可以类似处理）。给定基坐标系 $\{O\}$ 中的期望数据库 $\{ \bar{\mathbf{s}}_m, \bar{\boldsymbol{\mu}}_m, \bar{\boldsymbol{\Sigma}}_m \}_{m=1}^M$，将其投影到每个局部坐标系中，生成局部期望数据库 $\bar{D}^{(p)} = \{ \bar{\mathbf{s}}_m^{(p)}, \bar{\boldsymbol{\mu}}_m^{(p)}, \bar{\boldsymbol{\Sigma}}_m^{(p)} \}_{m=1}^M$，使用公式 44 进行变换。

然后，在每个局部坐标系 $\{p\}$ 中，使用第 3.1 节的更新规则（公式 34）更新局部参考数据库 $D^{(p)}$，生成新的局部参考数据库。

预测阶段

对于基坐标系 $\{O\}$ 中的新查询点 $\mathbf{s}^*$：

1. 根据新任务需求更新 $P$ 个局部坐标系（$\mathbf{A}^{(p)}$ 和 $\mathbf{b}^{(p)}$ 可能变化）。
2. 使用公式 44 将 $\mathbf{s}^*$ 投影到每个局部坐标系，得到局部输入 $\{ \mathbf{s}^{*(p)} \}_{p=1}^P$。
3. 在每个局部坐标系 $\{p\}$ 中，使用 KMP 预测局部轨迹点 $\tilde{\boldsymbol{\xi}}^{(p)}(\mathbf{s}^{*(p)}) \sim \mathcal{N}(\boldsymbol{\mu}_*^{(p)}, \boldsymbol{\Sigma}_*^{(p)})$，均值和协方差分别通过公式 21 和 26 计算。

将局部轨迹点转换回基坐标系

每个局部轨迹点 $\tilde{\boldsymbol{\xi}}^{(p)}(\mathbf{s}^{*(p)})$ 需要转换回基坐标系 $\{O\}$。根据公式 44 的逆变换，局部坐标系中的轨迹点转换回基坐标系为：

$$ \boldsymbol{\xi}^{(p)} = \mathbf{A}^{(p)}_\xi \tilde{\boldsymbol{\xi}}^{(p)} + \mathbf{b}^{(p)}_\xi $$

因此，每个局部轨迹点服从分布：

$$ \boldsymbol{\xi}^{(p)} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) $$

其中：

- $\tilde{\boldsymbol{\mu}}_p = \mathbf{A}^{(p)}_\xi \boldsymbol{\mu}_*^{(p)} + \mathbf{b}^{(p)}_\xi$
- $\tilde{\boldsymbol{\Sigma}}_p = \mathbf{A}^{(p)}_\xi \boldsymbol{\Sigma}_*^{(p)} (\mathbf{A}^{(p)}_\xi)^\top$

最终轨迹点的计算

在基坐标系 $\{O\}$ 中，查询点 $\mathbf{s}^*$ 对应的轨迹点 $\tilde{\boldsymbol{\xi}}(\mathbf{s}^*)$ 通过最大化 $P$ 个变换后高斯分布的乘积来确定（公式 45）：

$$ \tilde{\boldsymbol{\xi}}(\mathbf{s}^\*) = \arg\max*{\boldsymbol{\xi}} \prod*{p=1}^P \mathcal{N}(\boldsymbol{\xi} | \tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) $$

从公式 45 到公式 46 的推导

我们推导如何从公式 45 得到公式 46。

1. **高斯分布乘积**：  
   公式 45 的目标是最大化 $P$ 个高斯分布的乘积：

   $$ \prod\_{p=1}^P \mathcal{N}(\boldsymbol{\xi} | \tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) $$

   每个高斯分布的概率密度为：

   $$ \mathcal{N}(\boldsymbol{\xi} | \tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) = \frac{1}{(2\pi)^{\mathcal{O}/2} |\tilde{\boldsymbol{\Sigma}}\_p|^{1/2}} \exp\left( -\frac{1}{2} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p)^\top \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) \right) $$

   取对数，最大化乘积等价于最大化对数似然：

   $$ \log \left( \prod*{p=1}^P \mathcal{N}(\boldsymbol{\xi} | \tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) \right) = \sum*{p=1}^P \log \mathcal{N}(\boldsymbol{\xi} | \tilde{\boldsymbol{\mu}}\_p, \tilde{\boldsymbol{\Sigma}}\_p) $$

   忽略常数项，目标函数为：

   $$ \sum\_{p=1}^P \left( -\frac{1}{2} \log |\tilde{\boldsymbol{\Sigma}}\_p| - \frac{1}{2} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p)^\top \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) \right) $$

   最大化对数似然等价于最小化：

   $$ \sum\_{p=1}^P (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p)^\top \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) $$

2. **求导**：  
   对 $\boldsymbol{\xi}$ 求导：

   $$ \frac{\partial}{\partial \boldsymbol{\xi}} \sum*{p=1}^P (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p)^\top \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) = \sum*{p=1}^P 2 \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) $$

   令导数为零：

   $$ \sum\_{p=1}^P \tilde{\boldsymbol{\Sigma}}\_p^{-1} (\boldsymbol{\xi} - \tilde{\boldsymbol{\mu}}\_p) = 0 $$

   $$ \left( \sum*{p=1}^P \tilde{\boldsymbol{\Sigma}}\_p^{-1} \right) \boldsymbol{\xi} = \sum*{p=1}^P \tilde{\boldsymbol{\Sigma}}\_p^{-1} \tilde{\boldsymbol{\mu}}\_p $$

   解出 $\boldsymbol{\xi}$：

   $$ \tilde{\boldsymbol{\xi}}(\mathbf{s}^\*) = \left( \sum*{p=1}^P \tilde{\boldsymbol{\Sigma}}\_p^{-1} \right)^{-1} \sum*{p=1}^P \tilde{\boldsymbol{\Sigma}}\_p^{-1} \tilde{\boldsymbol{\mu}}\_p $$

   这就是公式 46。

算法 2：局部 KMP 的实现

算法 2 总结了局部 KMP 的流程：

1. **初始化**：

   - 定义核函数 $k(\cdot, \cdot)$ 和正则化参数 $\lambda$。
   - 确定 $P$ 个局部坐标系 $\{ \mathbf{A}^{(p)}, \mathbf{b}^{(p)} \}_{p=1}^P$。

2. **从局部示范中学习**：

   - 收集示范 $\{ \{ \mathbf{s}_{n,h}, \boldsymbol{\xi}_{n,h} \}_{n=1}^N \}_{h=1}^H$。
   - 使用公式 44 将示范投影到局部坐标系。
   - 提取局部参考数据库 $\{ \mathbf{s}_n^{(p)}, \hat{\boldsymbol{\mu}}_n^{(p)}, \hat{\boldsymbol{\Sigma}}_n^{(p)} \}_{n=1}^N$。

3. **更新局部参考数据库**：

   - 使用公式 44 将途径点或终点投影到局部坐标系。
   - 使用公式 34 更新局部参考数据库。
   - 更新每个坐标系 $\{p\}$ 中的核矩阵 $\mathbf{K}^{(p)}$、均值 $\boldsymbol{\mu}^{(p)}$ 和协方差 $\boldsymbol{\Sigma}^{(p)}$。

4. **使用局部 KMP 预测**：
   - 输入查询点 $\mathbf{s}^*$。
   - 根据新任务需求更新局部坐标系。
   - 使用公式 44 将 $\mathbf{s}^*$ 投影到局部坐标系，得到 $\{ \mathbf{s}^{*(p)} \}_{p=1}^P$。
   - 在每个坐标系 $\{p\}$ 中使用 KMP 预测局部轨迹点。
   - 使用公式 46 计算基坐标系中的轨迹点 $\tilde{\boldsymbol{\xi}}(\mathbf{s}^*)$。

局部 KMP 就像给 KMP 加了一个“地图转换器”：你在一个城市学会了走路（示范），但现在要去另一个城市（新任务空间）。KMP 通过“坐标转换”（局部坐标系和仿射变换），把你在第一个城市学到的走法“搬”到新城市，还能根据新城市的特点调整。公式 46 就像把多个城市的建议（局部轨迹点）加权平均，生成最终路线。

## 4 .1节：时间驱动的核化运动基元

> [!note]+ 4.1 节的目的
> 4 .1节通过联合建模位置和速度，扩展了 KMP 到时间驱动场景。核矩阵的计算考虑了基函数及其导数，适用于时间输入，但难以推广到高维输入。

时间驱动轨迹的建模

与 ProMP（Probabilistic Movement Primitives）类似，作者将轨迹参数化为`包含位置和速度`的形式（公式 47）：

$$ \begin{bmatrix} \boldsymbol{\xi}(t) \\ \dot{\boldsymbol{\xi}}(t) \end{bmatrix} = \boldsymbol{\Theta}(t)^\top \mathbf{w} $$

其中：

- $\boldsymbol{\xi}(t)$ 是位置，$\dot{\boldsymbol{\xi}}(t)$ 是速度，联合向量维度为 $2\mathcal{O}$（$\mathcal{O}$ 是输出维度，例如 3 D 位置的 $\mathcal{O}=3$）。
- $\boldsymbol{\Theta}(t) \in \mathbb{R}^{B\mathcal{O} \times 2\mathcal{O}}$ 是基函数矩阵，定义为（公式 48）：

$$ \boldsymbol{\Theta}(t) = \begin{bmatrix} \boldsymbol{\varphi}(t) & \mathbf{0} & \cdots & \mathbf{0} & \dot{\boldsymbol{\varphi}}(t) & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\varphi}(t) & \cdots & \mathbf{0} & \mathbf{0} & \dot{\boldsymbol{\varphi}}(t) & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & \boldsymbol{\varphi}(t) & \mathbf{0} & \mathbf{0} & \cdots & \dot{\boldsymbol{\varphi}}(t) \end{bmatrix} $$

- $\boldsymbol{\varphi}(t) \in \mathbb{R}^B$ 是基函数向量（例如高斯基函数），$B$ 是基函数数量。
- $\dot{\boldsymbol{\varphi}}(t)$ 是基函数对时间的导数。
- $\mathbf{w} \in \mathbb{R}^{B\mathcal{O}}$ 是权重向量。

通过包含 $\dot{\boldsymbol{\xi}}(t)$ 和 $\dot{\boldsymbol{\varphi}}(t)$，公式 47 能够编码运动的动态特性（例如速度信息）。

概率建模

为了捕捉示范中的变异性，作者使用高斯混合模型（`GMM`）建模联合概率 $P(t, \boldsymbol{\xi}, \dot{\boldsymbol{\xi}})$，类似于第 2.1 节的方法。通过高斯混合回归（`GMR`），可以提取时间输入 $t_n$ 对应的条件概率分布：

$$ P(\hat{\boldsymbol{\xi}}\_n, \dot{\hat{\boldsymbol{\xi}}}\_n | t_n) \sim \mathcal{N}(\hat{\boldsymbol{\mu}}\_n, \hat{\boldsymbol{\Sigma}}\_n) $$

其中 $\hat{\boldsymbol{\mu}}_n$ 和 $\hat{\boldsymbol{\Sigma}}_n$ 分别是联合向量 $[\hat{\boldsymbol{\xi}}_n^\top, \dot{\hat{\boldsymbol{\xi}}}_n^\top]^\top$ 的均值和协方差。

时间驱动 KMP 的推导

有了参考分布，我们可以按照第 2.2 节的推导方法，构建时间驱动的 KMP。损失函数形式与之前相同，但输入从 $\mathbf{s}$ 变为时间 $t$，输出从 $\boldsymbol{\xi}$ 扩展为 $[\boldsymbol{\xi}^\top, \dot{\boldsymbol{\xi}}^\top]^\top$。

1. **损失函数**：  
   损失函数为：

   $$ J(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}\_w) = \sum_{n=1}^N D\_{\text{KL}}(P_p([\boldsymbol{\xi}, \dot{\boldsymbol{\xi}}] | t_n) \| P_r([\hat{\boldsymbol{\xi}}_n, \dot{\hat{\boldsymbol{\xi}}}_n] | t_n)) $$

   其中：

   - $P_p([\boldsymbol{\xi}, \dot{\boldsymbol{\xi}}] | t_n) = \mathcal{N}([\boldsymbol{\xi}, \dot{\boldsymbol{\xi}}] | \boldsymbol{\Theta}(t_n)^\top \boldsymbol{\mu}_w, \boldsymbol{\Theta}(t_n)^\top \boldsymbol{\Sigma}_w \boldsymbol{\Theta}(t_n))$
   - $P_r([\hat{\boldsymbol{\xi}}_n, \dot{\hat{\boldsymbol{\xi}}}_n] | t_n) = \mathcal{N}([\hat{\boldsymbol{\xi}}_n, \dot{\hat{\boldsymbol{\xi}}}_n] | \hat{\boldsymbol{\mu}}_n, \hat{\boldsymbol{\Sigma}}_n)$

2. **优化**：  
   按照第 2.2.2 节的方法，优化 $\boldsymbol{\mu}_w$ 和 $\boldsymbol{\Sigma}_w$，最终得到：

   $$ \boldsymbol{\mu}\_w^\* = \boldsymbol{\Phi} (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \boldsymbol{\Sigma})^{-1} \boldsymbol{\mu} $$

   其中 $\boldsymbol{\Phi} = [\boldsymbol{\Theta}(t_1), \ldots, \boldsymbol{\Theta}(t_N)]$，$\boldsymbol{\mu}$ 和 $\boldsymbol{\Sigma}$ 是基于参考分布的。

核矩阵的计算

在计算核矩阵 $\mathbf{k}(t_i, t_j) = \boldsymbol{\Theta}(t_i)^\top \boldsymbol{\Theta}(t_j)$ 时（公式 16-18），由于 $\boldsymbol{\Theta}(t)$ 包含 $\boldsymbol{\varphi}(t)$ 和 $\dot{\boldsymbol{\varphi}}(t)$，需要计算四种内积：

- $\boldsymbol{\varphi}(t_i)^\top \boldsymbol{\varphi}(t_j)$
- $\boldsymbol{\varphi}(t_i)^\top \dot{\boldsymbol{\varphi}}(t_j)$
- $\dot{\boldsymbol{\varphi}}(t_i)^\top \boldsymbol{\varphi}(t_j)$
- $\dot{\boldsymbol{\varphi}}(t_i)^\top \dot{\boldsymbol{\varphi}}(t_j)$

作者提出使用有限差分法近似 $\dot{\boldsymbol{\varphi}}(t)$：

$$ \dot{\boldsymbol{\varphi}}(t) \approx \frac{\boldsymbol{\varphi}(t + \delta) - \boldsymbol{\varphi}(t)}{\delta} $$

其中 $\delta > 0$ 是一个极小的常数。基于核函数定义 $\boldsymbol{\varphi}(t_i)^\top \boldsymbol{\varphi}(t_j) = k(t_i, t_j)$，核矩阵 $\mathbf{k}(t_i, t_j)$ 为（公式 49）：

$$ \mathbf{k}(t*i, t_j) = \boldsymbol{\Theta}(t_i)^\top \boldsymbol{\Theta}(t_j) = \begin{bmatrix} k*{tt}(i,j) \mathbf{I}_\mathcal{O} & k_{td}(i,j) \mathbf{I}_\mathcal{O} \\ k_{dt}(i,j) \mathbf{I}_\mathcal{O} & k_{dd}(i,j) \mathbf{I}\_\mathcal{O} \end{bmatrix} $$

其中（公式 50）：

- $k_{tt}(i,j) = k(t_i, t_j)$
- $k_{td}(i,j) = \frac{k(t_i, t_j + \delta) - k(t_i, t_j)}{\delta}$
- $k_{dt}(i,j) = \frac{k(t_i + \delta, t_j) - k(t_i, t_j)}{\delta}$
- $k_{dd}(i,j) = \frac{k(t_i + \delta, t_j + \delta) - k(t_i + \delta, t_j) - k(t_i, t_j + \delta) + k(t_i, t_j)}{\delta^2}$

替代方法：扩展基函数矩阵

作者指出，可以不使用 $\dot{\boldsymbol{\varphi}}(t)$，而是直接扩展基函数矩阵：

$$ \boldsymbol{\Theta}(t) = \text{blockdiag}(\boldsymbol{\varphi}(t), \boldsymbol{\varphi}(t), \ldots, \boldsymbol{\varphi}(t)) $$

此时 $\boldsymbol{\Theta}(t) \in \mathbb{R}^{2B\mathcal{O} \times 2\mathcal{O}}$，$\mathbf{w} \in \mathbb{R}^{2B\mathcal{O}}$。这种方法避免了计算 $\dot{\boldsymbol{\varphi}}(t)$，但增加了维度。

相比之下，使用公式 48 的定义（包含 $\dot{\boldsymbol{\varphi}}(t)$），$\mathbf{w}$ 的维度为 $B\mathcal{O}$，更紧凑。

高维输入的挑战

时间驱动的 KMP 适用于输入为时间 $t$ 的情况，但难以推广到高维输入 $\mathbf{s}$。原因在于：

- 对于时间 $t$，可以使用有限差分法近似 $\dot{\boldsymbol{\varphi}}(t)$。
- 对于高维输入 $\mathbf{s}$，计算导数 $\dot{\boldsymbol{\varphi}}(\mathbf{s}) = \frac{\partial \boldsymbol{\varphi}(\mathbf{s})}{\partial \mathbf{s}} \frac{\partial \mathbf{s}}{\partial t}$ 需要额外的动态模型来描述 $\mathbf{s}$ 和 $t$ 的关系，这是一个非平凡问题。

因此，对于高维输入 $\mathbf{s}$，通常采用扩展基函数矩阵的方法（公式 2）：

$$ \boldsymbol{\Theta}(\mathbf{s}) = \text{blockdiag}(\boldsymbol{\varphi}(\mathbf{s}), \boldsymbol{\varphi}(\mathbf{s}), \ldots, \boldsymbol{\varphi}(\mathbf{s})) $$

示例：手写字母“G”

文中通过手写字母“G”的示范展示了时间驱动 KMP 的效果（图 1）：

- 图 1 (a)：展示了“G”的轨迹，起点和终点分别用“\*”和“+”标记。
- 图 1 (b)：使用 GMM 估计的分布，椭圆表示高斯分量。
- 图 1 (c)：通过 GMR 提取的参考轨迹分布，红色实线和阴影区域分别表示均值和标准差。

时间驱动 KMP 就像教机器人写字：你先示范写字母“G”（包含位置和速度），KMP 学会这些轨迹的模式（用 GMM/GMR 建模）。然后，KMP 用这些模式生成新的“G”，即使起点或终点变了，也能写得差不多。核矩阵的计算就像在比较两笔画的相似性（包括速度），有限差分法是用来估算笔画速度的“近似公式”。

## 4.2 节：时间驱动 KMP 的时间尺度调制

> [!note]+ 4.2 节的目的
> 第 4.2 节通过定义时间变换函数 $\tau(t^*)$，实现了时间驱动 KMP 的时间尺度调制。新查询时间 $t^*$ 映射到原始时间 $\tau(t^*)$，然后通过 KMP 预测调整后的轨迹。

问题背景

在时间驱动的轨迹中，机器人运动的持续时间可能需要根据新任务调整。例如，原始示范的运动持续时间为 $t_N$（即参考轨迹的时间长度），而新任务要求运动持续时间为 $t_D$，可能是更短（加速）或更长（减慢）。这`需要对轨迹进行时间尺度调制。`

方法：`时间变换`

为了生成适应新持续时间 $t_D$ 的轨迹，作者定义了一个单调函数 $\tau: [0, t_D] \to [0, t_N]$，用于将新时间范围 $[0, t_D]$ 映射到原始时间范围 $[0, t_N]$。这种时间变换也被称为“相位变换”（phase transformation），在文献中（如 Ijspeert et al., 2013; Paraschos et al., 2013）有类似讨论。

具体实现

1. **时间变换函数**：  
   $\tau(t^*)$ 将新任务中的查询时间 $t^* \in [0, t_D]$ 映射到原始时间范围中的 $\tau(t^*) \in [0, t_N]$。

   - 如果 $t_D < t_N$，运动加速，$\tau(t^*)$ 增长得更快。
   - 如果 $t_D > t_N$，运动减慢，$\tau(t^*)$ 增长得更慢。

   一个简单的线性变换示例是：

   $$ \tau(t^_) = \frac{t_N}{t_D} t^_ $$

   这种线性映射确保：

   - 当 $t^* = 0$ 时，$\tau(0) = 0$。
   - 当 $t^* = t_D$ 时，$\tau(t_D) = t_N$。

2. **使用 KMP 预测**：  
   对于新任务中的查询时间 $t^* \in [0, t_D]$，我们不直接使用 $t^*$ 作为 KMP 的输入，而是使用变换后的时间 $\tau(t^*)$。  
   根据第 4 节的定义，KMP 预测轨迹点为：

   $$ \begin{bmatrix} \boldsymbol{\xi}(\tau(t^_)) \\ \dot{\boldsymbol{\xi}}(\tau(t^_)) \end{bmatrix} = \boldsymbol{\Theta}(\tau(t^_))^\top \boldsymbol{\mu}\_w^_ $$

   其中 $\boldsymbol{\mu}_w^*$ 是通过 KMP 优化得到的参数（参考第 2.2.2 节）。

   使用核形式预测：

   $$ \mathbb{E}\left[ \begin{bmatrix} \boldsymbol{\xi}(\tau(t^*)) \\ \dot{\boldsymbol{\xi}}(\tau(t^*)) \end{bmatrix} \right] = \mathbf{k}(\tau(t^\*)) (\mathbf{K} + \lambda \boldsymbol{\Sigma})^{-1} \boldsymbol{\mu} $$

   其中：

   - $\mathbf{k}(\tau(t^*))$ 是核向量，基于 $\tau(t^*)$ 计算。
   - $\mathbf{K}$ 是核矩阵，基于原始时间 $\{ t_n \}_{n=1}^N$。

3. **速度的调整**：  
   注意到 $\dot{\boldsymbol{\xi}}(t)$ 是对原始时间的导数，而新轨迹的时间尺度已改变。设原始轨迹为 $\boldsymbol{\xi}(\tau(t^*))$，新轨迹的速度需要通过链式法则计算：

   $$ \dot{\boldsymbol{\xi}}\_{\text{new}}(t^_) = \frac{d}{dt^_} \boldsymbol{\xi}(\tau(t^_)) = \dot{\boldsymbol{\xi}}(\tau(t^_)) \cdot \frac{d\tau(t^_)}{dt^_} $$

   对于线性变换 $\tau(t^*) = \frac{t_N}{t_D} t^*$：

   $$ \frac{d\tau(t^_)}{dt^_} = \frac{t_N}{t_D} $$

   因此：

   $$ \dot{\boldsymbol{\xi}}\_{\text{new}}(t^_) = \dot{\boldsymbol{\xi}}(\tau(t^_)) \cdot \frac{t_N}{t_D} $$

   这表明速度会根据时间尺度比例 $\frac{t_N}{t_D}$ 缩放：

   - 如果 $t_D < t_N$（加速），$\frac{t_N}{t_D} > 1$，速度增大。
   - 如果 $t_D > t_N$（减慢），$\frac{t_N}{t_D} < 1$，速度减小。

时间尺度调制就像调整视频播放速度：你有一段录制的机器人动作（持续时间 $t_N$），但新任务要求更快或更慢地完成（持续时间 $t_D$）。通过时间变换 $\tau(t^*)$，KMP 就像“重新播放”这段动作，但速度变了——如果 $t_D$ 更短，就像快进；如果 $t_D$ 更长，就像慢放。速度也会相应调整，确保动作看起来自然。
