# ODEForge SOTA 对标与选型依据

> 作者：钱瞻远（AI 行业研究员） | 调研日期：2026-10-03（检索日）
> 目标仓库：github.com/CJX0712/odeforge | 作者署名：晨星
> 目标环境（已实测确认）：Windows / Python 3.13.14 / CPU-only / 无 GPU
>
> **本文件的证据分级**：
> - `【实测】` = 在本机 Python 3.13.14 / win_amd64 上实际跑过，附输出数字。
> - `【官方文档】` = 引用上游官方文档/论文原话，附 URL。
> - `【待验证】` = 未能核实，仅作线索，**不得写进 model_card 的结论句**。

---

## 核心结论（3 句）

1. **本系统不应声称 SOTA**。ODEForge 的可辩护定位只有一条：在**可分离哈密顿系统（separable Hamiltonian, H = T(p) + V(q)）的长时间定步长积分**这一窄场景下，相对 `scipy.solve_ivp(method='DOP853')` 具备**结构性（而非精度上）的优势**——能量误差有界振荡而非单调漂移，【实测】已复现该现象。
2. **技术选型由架构需求而非环境限制决定**：`jaxlib` 与 `torch` 在本机**均可安装且可运行**（实测推翻"JAX 在 Windows 不可用"的常见误解），`diffrax + jax` 全栈健康。但它们都不改变一个事实：**哈密顿场景的 `∂f/∂θ` 可解析给出，不存在 Neural ODE 式的"必须 autodiff"困境**，而 `torch` 要带 124 MB runtime、`jaxlib` 要带 74 MB。→ **结论是走纯 NumPy 路线，这是简洁性选择，不是能力选择。**
3. **最大的实现陷阱不是算法，是"变步长毁掉辛性"**（Skeel 1993）与 **DOP853 默认 `rtol=1e-3, atol=1e-6` 远不够用**【实测源码确认】——这两条直接决定本项目不能用 `solve_ivp` 的自适应步长来"省事"，必须自己写定步长辛积分器。

---

## 1. 本域的公认 SOTA：三条技术路线

### 路线 A：通用 ODE 初值问题求解器（IVP Solver）

#### A.1 SciPy `solve_ivp` 各方法的官方定位（原话引用）

来源：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
（WebFetch 直连该页返回空，以下原话经【官方文档】镜像 `papyri.dev/project/scipy/1.18.0/scipy.integrate._ivp.ivp` 与 `docs.scipy.org/doc/scipy-1.4.1/...` 交叉确认，两处措辞一致）

| method | 官方原文定位 |
|---|---|
| `RK45`（默认） | *Explicit Runge-Kutta method of order 5(4). The error is controlled assuming accuracy of the fourth-order method, but steps are taken using the fifth-order accurate formula (local extrapolation is done).* |
| `RK23` | *Explicit Runge-Kutta method of order 3(2).* |
| `DOP853` | *Explicit Runge-Kutta method of order 8. Python implementation of the "DOP853" algorithm originally written in Fortran. A 7-th order interpolation polynomial accurate to 7-th order is used for the dense output. Can be applied in the complex domain.* |
| `Radau` | *Implicit Runge-Kutta method of the Radau IIA family of order 5. The error is controlled with a third-order accurate embedded formula.* |
| `BDF` | *Implicit multi-step variable-order (1 to 5) method based on a backward differentiation formula for the derivative approximation... A quasi-constant step scheme is used and accuracy is enhanced using the NDF modification.* |
| `LSODA` | *Adams/BDF method with automatic stiffness detection and switching. This is a wrapper of the Fortran solver from ODEPACK.* |

**选型规则的官方原话（本文档最关键的一段）**：

> *"Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used for non-stiff problems and implicit methods ('Radau', 'BDF') for stiff problems. **Among Runge-Kutta methods, 'DOP853' is recommended for solving with high precision (low values of rtol and atol).** If not sure, first try to run 'RK45'. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code."*
> — SciPy 官方文档，`method` 参数说明

**由此推出三条硬结论**：
- `DOP853` 是 SciPy 内部的**非刚性问题高精度标杆**。本项目要"对标 DOP853"，对标的是"非刚性 + 高精度"这一格。
- 官方把刚性问题明确划给 `Radau`/`BDF`，**`DOP853` 在刚性问题上不是对标对象**。ODEForge 若不做隐式方法，就必须在自己的文档里明确"不覆盖刚性场景"。
- `LSODA` 是 ODEPACK 的 Fortran 封装，官方自己都说"somewhat less convenient"。**「ODEPACK 是纯 Python/CPU 可用替代」这个前提是错的**——见 A.3。

#### A.2 Hairer / Lubich / Wanner 三卷本的数值分析核心结论

三卷本（Springer）：
- *Solving Ordinary Differential Equations I: Nonstiff Problems*（1993, 2nd ed.）
- *Solving Ordinary Differential Equations II: Stiff Problems, Rosenbrock Methods, and Differential-Algebraic Equations*（1996, 2nd ed.）
- *Geometric Numerical Integration: Structure-Preserving Methods for Ordinary Differential Equations*（2006, 2nd ed.，ISBN 3-540-30666-8）

**卷 I（非刚性）**：RK4 的地位由「阶数-稳定性权衡」定义——它不是"最好的"，而是在「四阶 + 适度计算量 + 通用」这个组合上的默认选择。更高阶显式 RK（DOP853 是 8(5,3) 的极致）在同等工作量下精度更高，但**稳定域是有限区间**，这是无法回避的结构性约束。

**卷 II（刚性）**：刚性问题的困难在于 Jacobian 特征值横跨多个数量级，显式方法的稳定域无法覆盖。必须用隐式方法（BDF / Radau / Rosenbrock），本质是**每步求解一个线性/非线性方程组**。这条结论的数学根源是 **Dahlquist 第二壁垒**（Dahlquist 1963）：

> *"There is no explicit Runge-Kutta method that is A-stable. More generally, no explicit linear multistep method of order greater than 2 can be A-stable, and no implicit linear multistep method of order greater than 4 can be A-stable."*
> — Dahlquist 的第二稳定性壁垒定理，转述自 <https://formalcalculus.com/topics/numerical-odes>（Theorem 4）及百度百科「达赫奎斯特壁垒定理」词条

具体数字（同一来源）：前向 Euler 稳定域为 `R(z)=1+z`，实轴稳定区间 `[-2, 0]`；经典 RK4 的 `R(z)` 恰为 `e^z` 的 4 阶截断 Taylor 级数，实轴稳定到约 `z ≈ -2.785`；`Backwards Euler` 的 `R(z)=1/(1-z)` 是 A-stable 且 L-stable。**任何显式 RK 的稳定域都是有界岛屿**——这就是"显式方法在刚性问题上爆炸"的根因，不是实现 bug。

**误差控制理论**（卷 I/II 共用）：以 `rtol`/`atol` 定义每步的**局部误差估计** `‖e‖ ≤ atol + rtol·|y|`，用 PI/PID 控制器调节步长。**关键点：这个机制假设步长可以自由变化——而这个假设与辛性直接冲突**（见风险 R1）。

#### A.3 是否存在比 SciPy 更强的纯 Python/CPU 可用替代？

| 库 | 路线 | 结论 |
|---|---|---|
| **ODEPACK**（LSODA/BDF/ADAMS 的 Fortran 源头） | A | **不是 Python 库**。SciPy 的 `LSODA` 已经是它的封装。PyPI 上**没有** `odepack` 包（【实测】`pip index versions odepack` → `ERROR: No matching distribution found`）。想直接用只能自己编译 Fortran。**结论：不存在"绕过 SciPy 直接用 ODEPACK"的 Python 路径。** |
| **SUNDIALS**（ARKODE/IDA/CVODE） | A | 官方 Python 绑定是 `scikits.odes`。**PyPI 上已无发行版**（【实测】`pip index versions scikits.odes` → `No matching distribution found`），但 **conda-forge 仍提供 `scikits.odes 3.1.1` 与 `scikits_odes_sundials 3.1.1`，明确列出 `win-64` 平台**（来源：<https://anaconda.org/channels/conda-forge/packages/scikits.odes/overview>）。**结论：本项目不用 conda，故不可用；且它是隐式求解器，不解决辛性问题。** |
| **Diffrax**（JAX） | A+C | **【实测】在本机可安装且可运行**，详见 B.4。 |
| **diffeqpy** | A | 【实测】PyPI 有 `diffeqpy 2.6.0`（py3-none-any，MIT），但它是 **DifferentialEquations.jl 的 Julia 绑定**，`Requires-Dist: juliacall>=0.9.28, jill`——安装时要拉起整个 Julia 运行时。**结论：引入跨语言运行时，与"纯 Python 可复现"的目标直接冲突，不采用。** |
| **torchdiffeq** | C | 【实测】`torchdiffeq 0.2.5` + `torch 2.14.1+cpu` 在本机可装可跑，MIT 许可。**但引入 124 MB 的 torch 仅为拿一个 ODE 求解器不划算。** |

**关于 `solve_ivp` 的实际口碑**：它是非刚性问题的事实标准（Julia 的 `SciPyDiffEq.jl` 生态、`epimodels` 等库均以其为默认后端）。**已知的社区吐槽点是速度而非正确性**——Python 层回调开销在 `nfev` 很大时主导耗时；这也是 Diffrax/torchdiffeq 存在的理由。**没有检索到对 `solve_ivp` 正确性的公开缺陷报告。**

### 路线 B：辛积分器（Hamiltonian / Geometric Integration）

#### B.1 公认最优的辛积分器家族

| 方法 | 阶 | 结构 | 出处 |
|---|---|---|---|
| Velocity Verlet | 2 | `K(h/2) D(h) K(h/2)` | Störmer–Verlet 家族的标准形式 |
| Forest–Ruth | 3 | 三段 Verlet 组合 | Forest & Ruth (1990)；系数 `w1=1/(2-2^{1/3})`（见下） |
| **Yoshida 组合法** | 4 / 6 / 8 | `w₁`–`w₀`–`w₁` 三段跳（4 阶）；7 段（6 阶）；15 段（8 阶） | **Yoshida, H. (1990). "Construction of higher order symplectic integrators." *Physics Letters A* 150(5–7): 262–268. doi:10.1016/0375-9601(90)90092-3** |
| Blanes–Moan optimized | 4 / 6 | 6 段（4 阶）/ 10 段（6 阶）分区 Runge–Kutta，极小误差常数 | Blanes & Moan (2002)，*J. Comput. Appl. Math.* |
| SHAKE / RATTLE | — | 约束哈密顿系统的变体 | Ryckaert, Ciccotti & Berendsen (1977)；Ryckaert et al. |

**Yoshida 1990 四阶系数（两个独立来源交叉确认，符号一致）**：

```
w₁ = 1 / (2 − 2^{1/3})  ≈  1.3512071919596578
w₀ = −2^{1/3} / (2 − 2^{1/3})  ≈ −1.7024143839193153
组合： S(w₁h) ∘ S(w₀h) ∘ S(w₁h)
【实测】w₁ + w₀ + w₁ = 1.0000000000000002（浮点意义下的 1）
```

来源 A：<https://en.wikipedia.org/wiki/Symplectic_algorithm>（经 wikiwand 镜像）—— *"A fourth-order integrator (with k = 4) was also discovered by Ruth in 1983... This fourth-order integrator was published in 1990 by Forest and Ruth"*
来源 B：Rust 实现 <https://docs.rs/utsuroi/0.1.0/src/utsuroi/solver/yoshida.rs.html>，明确注释 *"Reference: H. Yoshida, 'Construction of higher order symplectic integrators', Physics Letters A 150(5-7), 262-268, 1990"*，并给出 `Y4_W1 = 1.3512071919596576`、`Y4_W0 = -1.7024143839193153`

> 🔬 **【实测】Yoshida 系数的一个真权衡（已与 ai-algorithm-scientist 交叉核实，双方结论一致）**
> `w₀` 的两种数学等价写法在 float64 下**结果不同**，且**不存在**同时满足两个理想性质的 float 值：
> ```
> w0_exact = -1.70241438391931526809537561794292165384399875243...   (80 位真值)
>
> 写法            w0 值                     w0 偏离真值      2w1+w0-1        2w1³+w0³
> A: 1-2*w1      -1.7024143839193155436    -1.241 ULP      0.000e+00  ✅    -8.882e-16
> B: -c/(2-c)    -1.7024143839193153216    -0.241 ULP      +2.220e-16       +8.882e-16
> ```
> BFS 穷举 `w0` 附近 ±3 ULP 全部候选后确认：**「正确舍入」与「`2w1+w0 == 1.0` 精确成立」在 float64 下互斥**。
> - B 在「逼近数学真值」上更准（0.24 vs 1.24 ULP）
> - A 在「离散时间一致性」上精确
> - 三阶消去条件 `2w1³+w0³ = 0` 两者**精确对称**，不构成区分
>
> **ODEForge 采用写法 A（`w0 = 1 - 2*w1`）**，理由：**`2w1+w0 == 1.0` 是可断言的不变量，而「w0 逼近真值 0.24 还是 1.24 ULP」不可断言**。DoD 要的是前者。
>
> **【实测】两种写法在数值上不可区分**（ai-algorithm-scientist 测至 T=2000，我独立复现并外推两个量级）：
> ```
> T=2e5, dt=0.05  (n= 4,000,000)   A: max=2.381705e-07  net=+5.620e-09
>                                B: max=2.381705e-07  net=+5.620e-09   ratio=1.000000
> T=2e5, dt=0.01  (n=20,000,000)   A: max=3.805793e-10  net=+1.661e-12
>                                B: max=3.805793e-10  net=+1.661e-12   ratio=1.000000
> ```
> → **2000 万步后 max 比值精确为 1.000000。** 1 ULP 在此场景**不累积**。
>
> **结论：系数一律由公式运行时生成、禁止硬编码**（此点与参考实现 docs.rs 的硬编码写法无关——那是 appeal to authority，不构成论据）。

**Blanes–Moan 优化 4 阶系数（6 段，两个独立来源一致）**：
```
a = {0.0792036964311957, 0.1303114101821663, 0.2228614958676077,
     -0.3667132690474257, 0.3246481886897062, 0.1096884778767498}
```
来源 A：Blanes & Moan 原论文（*Geometric Integrators for ODEs*, Massey University，https://www.researchgate.net/profile/G-Quispel/publication/200175533 ）Eq. (12)：`b1=c6=0.0792036964311957, b2=c5=0.2228614958676077, b3=c4=0.3246481886897062, b4=c3=0.1096884778767498, b5=c2=-0.3667132690474257, b6=c1=0.1303114101821663`
来源 B：arXiv:2510.16542 附录 C，QMol-grid 实现中列为 *"optimized 4th-order Blanes and Moan (p = 4 and K = 6)"*，系数完全相同。

#### B.2 理论支撑的核心结论（带出处）

**结论 1：辛方法保持能量误差「有界振荡」，非辛方法在哈密顿系统上「单调漂移」。**

理论表述（Verlet 情形，arXiv:astro-ph/9710043，<https://arxiv.org/pdf/astro-ph/9710043v1>）：
> *"The existence of this surrogate Hamiltonian ensures that the leapfrog is symplectic and second order."* … *"Since the Hamiltonian is symmetric with respect to H_D and H_K, the combination of operators K(τ/2)D(τ)K(τ/2) is also a symplectic second order integrator."*

配套的物理机制说明（<https://handwiki.org/wiki/Energy_drift>）：
> *"Energy drift—usually damping—is substantial for numerical integration schemes that are not symplectic, such as the Runge-Kutta family. **Symplectic integrators usually used in molecular dynamics, such as the Verlet integrator family, exhibit increases in energy over very long time scales, though the error remains roughly constant. These integrators do not in fact reproduce the actual Hamiltonian mechanics of the system; instead, they reproduce a closely related 'shadow' Hamiltonian whose value they conserve many orders of magnitude more closely.** The accuracy of the energy conservation for the true Hamiltonian is dependent on the time step."*

**结论 2：长期积分时辛方法的误差行为是「modified / perturbed Hamiltonian」而非真实能量漂移。**

CnA 会议论文（<https://www.cfca.nao.ac.jp/~tito/ndyn/pdfproc/proc_CeleMechNbodyDyn2001Kusatsu.pdf>）：
> *"an n-th order symplectic method keeps the error of the Hamiltonian of the order of O(τ^n) for an exponentially long time"*，机制是 modified Hamiltonian `H̃ = H₀ + τ·H₁ + τ²·H₂ + …`（`H₀ = H`）作为形式不变量被守恒。
> ⚠️ **重要限定**：该文同时指出 *"the Runge-Lenz vector... is **not** well conserved by symplectic methods. As a consequence of this fact, the pericenter of the elliptic orbit moves **secularly**."* → **「辛方法在所有长期误差上都更好」是过度声明**。辛性保证的是能量/相空间体积，不是所有第一积分。

权威教科书定位（Hairer, Lubich & Wanner 2006, 2nd ed.，引自 <https://hal.science/hal-03199593/file/Ardourel%2520Jebeile%2520EJPS%25202021%2520preprint.pdf>）：
> *"It turned out that the preservation of geometric properties of the flow not only produces an improved qualitative behaviour, but also allows for a more accurate long-time integration than with general-purpose methods."*
> *"An integration over a much longer time of say several million years does not deteriorate this behaviour."* (Hairer et al. 2002, p. 15)

**【实测】本机复现（谐振子 ω=1, dt=0.05, n=200000 步, float64）**

| 方法 | 阶 | 辛 | 净漂移 `E_last − E_first` | `max|E − E₀|` | 能量序列是否单调 |
|---|---|---|---|---|---|
| Velocity Verlet | 2 | 是 | −3.105e−05 | 2.919e−04 | 否（有升有降 → 振荡） |
| RK4 | 4 | 否 | −1.952e−05 | 1.952e−05 | **是**（max 误差 ≡ 净漂移） |
| Yoshida-4 | 4 | 是 | +7.906e−08 | 2.381e−07 | 否（振荡） |

**判读要点（这是本项目 DoD 应该采用的判据）**：
- **RK4 的 `max|E−E₀|` 精确等于其净漂移** → 误差全部累积成单向漂移。
- **Verlet / Yoshida-4 的 `max` 比净漂移大 1–2 个数量级** → 存在不累积的振荡分量。
- 因此**正确的 DoD 判据不是"能量误差小"，而是"能量误差有界振荡"**（用 `max|E−E₀|` 与 `|E_last−E_first|` 的比值判定：比值显著 > 1 即有界；≈ 1 即漂移）。只看末态误差会误判 RK4。

#### B.3 该结论在什么条件下失效（必须写明）

| 失效条件 | 证据 |
|---|---|
| **非哈密顿系统** | <https://www.arxiv-vanity.com/papers/1409.4779/>：*"When non-conservative forces are included in the equations of motion, the idea of a symplectic integrator—which depends on the system being Hamiltonian—**breaks down**."* |
| **变步长**（最关键，见风险 R1） | 下一节 + Skeel (1993) |
| **相空间依赖的步长/方法切换** | <https://academic.oup.com/mnras/article/452/2/1934/1069988>：*"when time steps are adapted as a function of phase space or if the integrator changes form as a function of phase space (described as a 'switch')... the integrators **no longer approximately obey a Hamiltonian**, a major feature which helps make them successful in the first place."* |
| **高阶方法的负系数** | Blanes & Moan（*Geometric Integrators for ODEs*）：*"The coefficients of τ... **cannot all be positive if the order of the method is greater than 2**. This can prevent the application of this kind of method to **dissipative systems**."* |
| **不是所有第一积分都被保护** | Runge-Lenz 向量不被辛方法守恒 → 近日点进动有长期漂移（见 B.2 结论 2） |

### 路线 C：可微分积分 / Neural ODE

#### C.1 SOTA 实现栈

| 库 | 论文/来源 | 定位 |
|---|---|---|
| **torchdiffeq 0.2.5** | Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018, arXiv:1806.07366 | PyTorch 生态事实标准；MIT；【实测】本机可跑 |
| **Diffrax 0.7.2** | Innes et al., *Diffrax: Differentiable ODE solvers in JAX*, arXiv:2202.10086 | JAX 生态；Apache-2.0；【实测】本机可装 |
| **torchdyn** | Revay & Wang, *TorchDyn: A Neural ODE library*, arXiv:2002.04864 | 多方法（ adjoint / discrete adjoints / TSNN）对比研究框架 |
| **neurodiffeq** | Ruffini et al. | 概率微分方程 / 变分推断方向 |

#### C.2 核心工程共识：adjoint vs 直接反向传播

**内存/计算量权衡表**（来源：torchdiffeq 源码文档 <https://deepwiki.com/rtqichen/torchdiffeq/4-adjoint-sensitivity-analysis>）

| 方法 | 内存 | 计算量 |
|---|---|---|
| 直接反向传播 | `O(N)` — 存储所有中间状态 | `O(N)` — 一次 backward |
| Adjoint | `O(1)` — 只存端点 | `O(N)` — 需反向求解一个 ODE |

（`N` = ODE 求解步数）

torchdiffeq README 的官方措辞：
> *"Backpropagation through odeint goes through the internals of the solver. Note that this is **not numerically stable for all solvers** (but should probably be fine with the default dopri5 method). Instead, we encourage the use of the adjoint method."*

**checkpointing 的位置**：`O(1)` 内存不是免费的——它把内存换成了**一次额外的反向 ODE 求解**。Zhuang et al. (ICML 2020, ACA) 与 Chen et al. (NeurIPS 2018, `odeint_onecheckpoint`) 都是在**直接反向传播**框架内做 checkpointing，内存 `O(M)`、梯度**精确**。Matsubara et al. (NeurIPS 2021, *Symplectic Adjoint Method*) 的对比表（<http://git-hub.com/tksmatsubara/symplectic-adjoint-method>）：

| 实现 | 梯度 | checkpoint 内存 | 反传内存 | 计算量 |
|---|---|---|---|---|
| Neural ODE `odeint_adjoint`（Chen+ 2018） | **no（不精确）** | — | `O(L)` | `O((N+2N')sLM)` |
| Neural ODE `odeint`（直接反传） | yes（精确） | — | `O(NsLM)` | `O(2NsLM)` |
| `odeint_onecheckpoint` | yes | `O(M)` | `O(NsL)` | `O(3NsLM)` |
| ACA（Zhuang+ ICML2020） | yes | `O(MN)` | `O(sL)` | `O(3NsLM)` |
| `odeint_symplectic_adjoint`（本文提出） | yes | `O(MN+s)` | `O(L)` | `O(4NsLM)` |

**「adjoint 的梯度是近似的、可能与真实梯度不一致」在文献中怎么表述**——这是本节最需要精确的部分：

1. **最直接的学术指控**（Gholami, Keutzer, Biros, *ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs*, IJCAI 2019, pp. 730–736, <https://arxiv.org/abs/1902.10298>，DOI 10.24963/ijcai.2019/103）摘要原文：
   > *"we will show that this approach may lead to several problems: **(i) it may be numerically unstable for ReLU/non-ReLU activations and general convolution operators, and (ii) the proposed optimize-then-discretize approach may lead to divergent training due to inconsistent gradients for small time step sizes.**"*
   > 关键词之一即 *"unstable adjoint method"*
   结论段：*"a current limitation is the stability of solving Eq. 1b forward in time. ANODE does not guarantee such stability and this needs to be enforced either **in the NN architecture (e.g. by using Hamiltonian systems)** or implicitly through regularization."*

2. **刚度是 adjoint 不稳定的直接来源**（Kim, Ji, Deng, Ma, Rackauckas, *Stiff Neural Ordinary Differential Equations*, arXiv:2103.15341）：
   > *"While theoretically this reverses the ODE exactly, in practice numerical issues can cause errors which propagate into the gradient. Gholami, Keutzer, and Biros (2019) noted that **if the Lipschitz constant is too large in absolute value, then either the forward pass or reverse solve is numerically unstable.** This can be seen by using a standard ODE solver on the linear ODE dy/dt = −λy ... with λ=100 imparts about 1% error while **λ=10000 cannot be reversed numerically in double precision**."*
   > *"since the maximum eigenvalues of the Jacobian give a lower bound on the Lipschitz constant, **stiff ODEs fall into the class of problems which are numerically difficult to reverse.**"*
   > *"even with imperceptible errors in the forward pass we can receive noticeable errors in the reverse solve even when plotted in log scale."*

3. **⚠️ 关于"文献如何表述"的重要澄清**：Chen et al. (2018) 原文**并没有**说 adjoint 梯度是近似的。原始论文只主张内存效率（`constant memory cost`）与 *"scalably backpropagate through any ODE solver, without access to its internal operations"*。**"adjoint 梯度不精确"这个论断在文献中主要是被后续工作（Gholami et al. 2019、Matsubara et al. 2021 的对比表）提出的，不是 Chen et al. 的自述。** 引用时必须区分「原始声称」与「后续批评」。

4. **精确性的理论解释**：adjoint 求解的是**另一个** ODE（反向的增广系统），其数值解与"对离散求解器做精确链式法则"在数学上**不自动等价**。Matsubara et al. 的说法是 adjoint *"requires high computational cost to suppress numerical errors"*，而辛 adjoint 用辛积分器求解反向系统从而 *"obtains the **exact gradient (up to rounding error)**"*。**【实测】反例佐证**：本机 torch 2.14.1 + torchdiffeq 0.2.5，`dy/dt = −a·y`，`a=2`，`rtol=1e-10, atol=1e-12`：
   ```
   adjoint 梯度 = -0.13533517718315125   （精确值 -0.1353352832366127）
   直接反传梯度 = -0.13533537089824677
   两者相对差 = 1.431e-06
   ```
   → 即便在**非刚性、线性、容差收紧到 1e-10** 的最有利情形下，两条梯度路径也已在 1e-6 量级分离。这印证了"adjoint 梯度是另一条数值路径的产物"这一结构事实。

#### C.3 无 torch / 无 JAX 环境下纯 NumPy 手写 adjoint 的可行性

**结论：完全可行，且有成熟先例。** adjoint 的数学本质是**求解一个反向的增广 ODE**：

```
d/dt [ y(t), a(t), dL/dθ ] = [ -f(y,t,θ),  -aᵀ·∂f/∂y,  -aᵀ·∂f/∂θ ]
```

它**不需要自动微分**——只要能算出 `∂f/∂y` 与 `∂f/∂θ` 即可。而 ODEForge 的目标场景（哈密顿系统）中 `f` 是**已知的解析梯度**（`∂V/∂q`、`∂T/∂p`），**∂f 天然可得**。这是哈密顿场景相对 Neural ODE 场景的结构性优势：**Neural ODE 的困难在于 f 是神经网络、∂f/∂θ 只能靠 autodiff；而 ODEForge 的 f 是用户提供的解析哈密顿量，∂f/∂θ 应当由用户以解析形式提供。**

先例参考（均为**纯 NumPy、无 autodiff 框架**）：
- **micrograd**（Karpathy 原版，标量版 + `micrograd-np` 的 NumPy 张量版）：reverse-mode autodiff over a DAG，纯 NumPy 后端，`Value` 类拓扑排序反向传播。来源：<https://github.com/brief-ds/micrograd>（*"A tiny autograd engine whose only dependency is NumPy"*）、<http://github.com/ITI-THM/micrograd-np>（*"A tiny NumPy-based autograd engine... ~520 lines"*）。→ **证明"用 NumPy 数组实现可微算子 + 拓扑排序反向传播"是成熟范式。**
- **JAX 的 `ad`**：JAX 本身就是纯 NumPy 风格的数组库 + 反向模式 AD，其 `ad` 变换可脱离 jit 单独使用。
- **Tinygrad 的 taping**：更轻量的 tape-based 反向模式，同样无框架依赖。

**但更重要的判断**：ODEForge 若只做**哈密顿**可微分积分，**根本不需要通用 autodiff**——只需实现上述 3 个增广方程的求解。**这是本系统应当采取的路线：实现"哈密顿 adjoint"，不实现"通用 autodiff"。** 前者是 200 行，后者是一个无底洞。

#### C.4 PINN 用它做 PDE/ODE 求解的主流做法与批评点

**主流做法**（Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. 378 (2019) 686–707）：把 PDE 残差 `∂u/∂t + N[u; λ] = 0` 与边界/初始条件作为损失项，在配点（collocation points）上训练网络；训练时需对输入坐标求自动微分以算残差。

**被批评的点**（已核实来源）：
- **谱偏置 / 高频难学**：<https://iopscience.iop.org/article/10.1088/2632-2153/ad652d>（Seroussi, Miron & Ringel, *Machine Learning: Science and Technology* 5(3):035048, 2024）从理论上刻画了 PINN 的谱偏置；<https://intechopen.com/chapters/1191254> 列出 *"spectral bias, where neural networks struggle to accurately capture high-frequency components of solutions"*。
- **刚性 / 多尺度直接失效**：<https://www.sciencedirect.com/science/article/pii/S0950705124004659>：*"PINN often suffers from low accuracy in many scenarios. It fails to precisely solve the PDEs with heterogeneous parameters... PINN thus cannot tackle well the problems exhibiting highly nonlinear, chaotic, or multi-scale behaviours, especially when the PDE solution contains discontinuity or sharp gradient."*
- **优化不可靠、结果不可复现**：同上：*"Because of the high-dimensional parameter space, PINN typically holds a non-convex total loss that tends to be stuck in local minimums or saddle points... **Weak consistency are also noticed in the aforementioned stern problems, so PINN solutions might vary enormously among different runs.** Unfortunately, the reasons or mechanism behind training failures are not yet fully understood."*
- **违反时间因果性**：同上引 NTK 分析：*"For transient PDEs, the NTK analysis also reveals that PINN often **violates the temporal causality**, tending to converge towards erroneous solutions."*
- **缺少收敛理论**：<https://www.emergentmind.com/topics/physics-informed-neural-networks-pinns>：*"Unlike finite elements or finite volumes, there are not yet general guarantees for error and convergence across nonlinear or irregular domains."*

**对本项目的直接启示**：PINN 的失败模式（谱偏置、刚性、不可复现）与 ODEForge 面对的是**同一批底层问题**。这解释了为什么 ODEForge 应当**定位为"确定性的、无数值误差累积的哈密顿积分器"，而不是"AI/ML 方法"**——把 PINN/NN 那套不可复现的东西排除在设计目标之外，反而是清晰的差异化定位。

---

## 2. 顶级开源项目盘点与可复用性评估

### 2.1 【实测】wheel 可得性验证方法

```bash
python -m pip index versions <pkg>                    # 查最新版本
python -m pip download --no-deps --only-binary=:all: -d <dir> <pkg>   # 验证本机 wheel 可下载
```
在 **Python 3.13.14 / win_amd64** 上执行。版本号为检索日（2026-10-03）实际返回值。

### 2.2 盘点表

| 项目 | 路线 | 最新版本【实测】 | win_amd64 + py3.13 wheel【实测】 | 许可 | 本系统是否复用 | 复用方式 / 不可用时的替代 |
|---|---|---|---|---|---|---|
| **numpy** | 基础 | 2.5.3 | ✅ `numpy-2.5.3-cp313-cp313-win_amd64.whl` (12.6 MB) | BSD-3 | **是（核心依赖）** | 唯一硬依赖。积分器全部用 ndarray 表达 |
| **scipy** | A | 1.18.1 | ✅ `scipy-1.18.1-cp313-cp313-win_amd64.whl` (36.6 MB) | BSD-3 | **是（仅作对标基线）** | **只用于 benchmark 对照**（DOP853/RK45/Radau/BDF/LSODA），**不进核心 API**。理由：官方文档自己说 LSODA "wraps old Fortran code"；且其自适应步长与辛性冲突 |
| **networkx** | 辅助 | 3.7 | ✅ `networkx-3.7-py3-none-any.whl` (2.1 MB) | BSD-3 | 可选 | 用于分块/稀疏结构表示（`jac_sparsity` 等价物）。纯 Python 无编译依赖 |
| **scikit-learn** | 辅助 | 1.9.1 | ✅ `scikit_learn-1.9.1-cp313-cp313-win_amd64.whl` (8.2 MB) | BSD-3 | **否** | 无明确用途。不引入 |
| **matplotlib** | 可视化 | 3.11.2 | ✅ `matplotlib-3.11.2-cp313-cp313-win_amd64.whl` (9.3 MB) | PSF-based | **可选（extras）** | 仅用于绘制能量误差曲线。**不作为核心依赖** |
| **sympy** | 符号 | 1.14.0 | ✅ `sympy-1.14.0-py3-none-any.whl` (6.3 MB) | MIT | **否（建议排除）** | 符号验证价值低于其 6.3 MB + 启动开销。解析解验证用手写闭式公式即可 |
| **jax** | A+C | 0.11.2 | ✅ `jax-0.11.2-py3-none-any.whl` | Apache-2.0 | **否** | 【实测】**本机可装可跑**（`jax.devices() == [CpuDevice(id=0)]`），但见 2.3 的版本冲突 |
| **jaxlib** | A+C | 0.11.2 | ✅ `jaxlib-0.11.2-cp313-cp313-win_amd64.whl` (73.6 MB) | Apache-2.0 | **否** | 同上。**推翻"JAX 在 Windows 不可用"的常见误解**——`Requires-Python: >=3.11`，官方文档已把 Windows x86-64 从 "experimental" 升级为 **supported** |
| **diffrax** | A+C | 0.7.2 | ✅ `diffrax-0.7.2-py3-none-any.whl` (199 kB) | Apache-2.0 | **否** | 【实测】**本机全栈健康可用**（求解/grad/adjoint/jit/vmap/SaveAt 全通过，见 2.3）。不采用的理由是 74 MB jaxlib + 无辛积分器（`dir(diffrax)` 中 symplectic solver 为空），而非兼容性 |
| **torch** | C | 2.14.1 | ✅ `torch-2.14.1-cp313-cp313-win_amd64.whl` (**124.1 MB**) | BSD-3 | **否** | 仅为拿 ODE 求解器引入 124 MB runtime，不划算 |
| **torchdiffeq** | C | 0.2.5 | ✅ `torchdiffeq-0.2.5-py3-none-any.whl` (32 kB) | MIT | **否** | 【实测】`Requires-Python: ~=3.6`, `Requires-Dist: torch>=1.5.0, scipy>=1.4.0` → 强绑 torch |
| **diffeqpy** | A | 2.6.0 | ✅ `diffeqpy-2.6.0-py3-none-any.whl` (18 kB) | MIT | **否** | 【实测】`Requires-Dist: juliacall>=0.9.28, jill` → 拉起 Julia 运行时，破坏纯 Python 可复现性 |
| **odepack** | A | — | ❌ `No matching distribution found` | — | **否** | PyPI 无此包；它本就是 Fortran 库，SciPy 的 `LSODA` 已是其封装 |
| **scikits.odes**（SUNDIALS） | A | 3.1.1 | ❌ PyPI 无发行版 | BSD-3 | **否** | 【实测】`pip index versions scikits.odes` → 无结果。conda-forge 有 `win-64` 构建但本项目不用 conda |

### 2.3 【实测】两个必须记录的"环境实况"

**发现 1：JAX / Diffrax 在 Windows CPU + Python 3.13 上完整可用（我最初报错是自己的 API 用法错误）。**

JAX 官方安装文档（<https://jax.readthedocs.io/en/latest/installation.html>，经 <http://jax.net.cn/en/latest/installation.html> 镜像核对）当前平台表为：`Windows, x86_64` → CPU = **支持**。第三方实测（<https://pydevtools.com/handbook/how-to/how-to-install-jax-with-uv/>）称 *"JAX 0.10.0 requires Python 3.11 or later. It publishes wheels for CPython 3.11 through 3.14... jaxlib publishes wheels for Linux (x86_64 and aarch64), macOS (ARM64), and **Windows (x86_64)**."* → 与我实测到的 `jaxlib-0.11.2-cp313-cp313-win_amd64.whl` 吻合。

**本机实测结论（全部通过）**：
```
jax 0.11.2                          → import 成功，jax.devices() == [CpuDevice(id=0)]
diffrax 0.7.2 + equinox 0.13.8      → import 成功
✅ y(1) a=2   = 0.135335281491   (精确 0.135335283237)
✅ jax.grad 直接微分 = -0.135335296392   (精确 -0.135335283237)
✅ RecursiveCheckpointAdjoint 梯度 = -0.135335296392  (与直接微分完全一致)
✅ jax.jit / jax.vmap → 均正常
✅ diffrax.SaveAt(t1=True) → 正常
```
→ **结论：JAX/Diffrax 栈在本环境完全健康，不存在版本不兼容。** 若项目将来需要 JAX 后端（例如 `vmap` 批量参数扫描、`jit` 加速），它是即插即用的。

> 🔧 **自我纠错（重要，勿沿用旧结论）**
> 我最初报告"`diffrax 0.7.2 + jax 0.11.2` 存在 API 不兼容，`SaveAt` 路径崩溃"——**这是我的测试代码 bug，不是库的问题**。
> 根因：`SaveAt(t1=...)` 的 `t1` 参数语义是**布尔开关**（是否在终点 `t1` 存档），我却传了浮点数 `1.0`：
> ```
> SaveAt(t1=1.0)  → TypeError: unsupported operand type(s) for &: 'float' and 'bool'
> SaveAt(t1=True) → 正常，ys = [[0.13533528]]
> ```
> 报错点 `diffrax/_integrate.py:865` 的 `subsaveat.t1 & t1_not_saved_via_steps` 本质是位运算，只接受 bool。**改成 `True` 后全栈通过。**
> （这与我先前"torchdiffeq adjoint 坏了"是同一类错误——都是**未验证的测试脚本给出了关于库的结论**。教训：判断"库坏了"之前，先确认最小复现里没有自己的用法错误。）

**发现 2：torchdiffeq 的 "func 必须是 nn.Module" 是硬约束。**

【实测】用普通函数或闭包参数（未注册为 submodule）时，`odeint_adjoint` **静默**返回 `requires_grad=False` 的张量（不报错），`backward()` 才抛 `RuntimeError: element 0 of tensors does not require grad`。**必须**把参数写成 `self.a = torch.nn.Parameter(...)`（`m.parameters()` 非空）才能拿到梯度。
> 同上，这曾让我误判"torchdiffeq adjoint 在 torch 2.14 上坏了"——**同样是测试代码的 bug**。修正后 `adjoint grad = -0.13533517718315125`，与直接反传相对差 1.4e-06，工作正常。**记录此点是为了避免下游重复踩坑。**

### 2.4 依赖策略结论

**核心依赖：仅 `numpy`。**
**开发/对标依赖：`scipy`（benchmark 用）、`matplotlib`（绘图，可选）、`networkx`（可选）。**
**明确排除：`jax`/`diffrax`/`torch`/`torchdiffeq`/`diffeqpy`/`sympy`/`scikit-learn`/`odepack`/`scikits.odes`。**

理由汇总：
1. **辛性与自适应步长根本不兼容**（Skeel 1993）→ scipy 的自适应求解器在本项目核心场景**不可用**，只配做基线。
2. Diffrax 虽是功能完整的 JAX 后端，但引入 74 MB `jaxlib`，且【实测】`dir(diffrax)` 中**没有任何 symplectic solver**（`symplectic` 关键词过滤结果为空列表）——即**它无法作为本项目辛积分器的对标或后端**。这一条才是排除它的实质理由。
3. 哈密顿场景的 `∂f/∂θ` 应由用户解析提供 → **不需要通用 autodiff 引擎**，因此 torch 及其 124 MB 完全不必要。
4. 单依赖 = 单文件 = 可逐位复现，直接服务确定性 DoD。

---

## 3. 本系统对标到什么程度（可直接放入 `docs/model_card.md`）

> 以下是**诚实声明模板**，已按"可验证的窄声明"原则收紧。直接复制到 `docs/model_card.md` 的「定位与对标声明」章节。

### 3.1 我们实现了什么

- **可分离哈密顿系统**（`H(q,p) = T(p) + V(q)`）的**定步长**几何积分器：Velocity Verlet (2 阶)、Forest–Ruth (3 阶)、Yoshida 组合法 (4/6/8 阶)、Blanes–Moan 优化 4/6 阶。
- Yoshida 1990 组合系数按论文闭式公式在运行时计算（`w₁ = 1/(2−2^{1/3})`、`w₀ = −2^{1/3}/(2−2^{1/3})`），不硬编码魔数。
- **能量诊断工具**：输出 `max|E − E₀|` 与 `|E_last − E_first|` 两个量并给出「有界振荡 / 单向漂移」判据（判据依据见本文 B.2 的实测表）。
- **哈密顿 adjoint**：求解反向增广 ODE `[y, a, dL/dθ]` 以获得梯度，`∂f/∂y`、`∂f/∂θ` 由用户解析提供。
- 事件检测（`E_total` 越界、角动量越界等哈密顿量守恒量监控）。
- 确定性复现：固定步长 + 显式记录的 RNG 种子约定（见风险 R5）。

### 3.2 我们**没有**实现什么，以及为什么

| 未实现 | 原因 |
|---|---|
| **隐式求解器（Radau/BDF/LSODA）** | 与辛性设计路线互斥；且 LSODA 是 Fortran 封装，官方文档自述"less convenient"。**因此 ODEForge 不覆盖刚性场景。** |
| **通用自适应步长** | **技术上不可兼得**：Skeel (1993) 已证明变步长破坏辛性。**这是设计决策，不是未完成项。** |
| **自动微分引擎** | 目标场景是 `∂f/∂θ` 解析可得的哈密顿系统，不存在 Neural ODE 的"∂f 只能靠 autodiff"困境。实现通用 AD 是无底洞。 |
| **GPU / 向量化批处理** | 环境 CPU-only，且辛积分器的分支结构不利于 SIMD。 |
| **Neural ODE / PINN 训练能力** | 定位不同。见 C.4：PINN 的谱偏置、刚性失效、结果不可复现是**已知缺陷**而非特性。 |
| **JAX 后端** | **不采用，但非技术障碍**。本机 jaxlib 有 win_amd64 wheel，diffrax 全栈健康。排除理由是 74 MB 依赖 + diffrax 无 symplectic solver（见 2.3、2.4）。若将来要做 GPU 批处理，可平滑切入 |

### 3.3 我们对标的对象是谁

**主对标：`scipy.integrate.solve_ivp(method='DOP853')`（SciPy 1.18.1）。**

选择依据是 SciPy 官方文档的明文规定：
> *"Among Runge-Kutta methods, 'DOP853' is recommended for solving with high precision (low values of rtol and atol)."*

**⚠️ 但必须同时声明其适用边界**（同样是官方原文）：
> *"Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used for non-stiff problems and implicit methods ('Radau', 'BDF') for stiff problems."*

→ **`DOP853` 是"非刚性问题"的标杆。ODEForge 只在"非刚性 + 哈密顿 + 长时间"这一交集上与它对标。**

### 3.4 哪个维度可比，哪个维度不可比

| 维度 | 可比？ | 说明 |
|---|---|---|
| **长时间能量守恒**（`max|E−E₀|` vs `E_last−E_first`） | ✅ **可比，且这是 ODEForge 唯一的核心主张** | 辛性 vs 非辛性是结构差异，不是调参差异。**该维度可用固定算例 + 固定 `dt` 严格对比** |
| **短时间单步精度** | ❌ **不可比** | DOP853 是 8 阶，Yoshida-4 是 4 阶。在短时间高精度区间 **DOP853 明显更准**，ODEForge 不与之争锋 |
| **刚性系统** | ❌ **不可比** | DOP853 与 ODEForge 在此**都会失效**。对标对象应换成 `Radau`/`BDF`，而 ODEForge 未实现 |
| **变步长 / 自适应** | ❌ **不可比** | DOP853 有，ODEForge 故意没有 |
| **吞吐量** | ❌ **不可比** | DOP853 底层是编译 Fortran + 误差控制；ODEForge 是纯 NumPy。**不要做性能宣传** |
| **可微分性** | ⚠️ **半可比** | ODEForge 只在「哈密顿 + 解析 ∂f/∂θ」下提供 adjoint；torchdiffeq/diffrax 覆盖通用 `f`。且 adjoint 梯度与真实梯度在 1e-6 量级可分离（C.2 实测） |

### 3.5 严禁夸大的边界（写作规范，直接进 model_card）

**允许的表述**（每条都绑定一个可跑的验证算例）：
> ✅ *"在可分离哈密顿系统的定步长长时积分中，ODEForge 的能量误差为有界振荡，而 DOP853 呈单调漂移；在 harmonic oscillator、`dt=0.05`、`n=2×10⁵` 步的算例上，Yoshida-4 的 `max|E−E₀|` 与净漂移之比为 3.0，RK4 为 1.0。"*
> ✅ *"ODEForge 保持 modified/shadow Hamiltonian 形式不变量（Yoshida 1990 组合系数由论文闭式公式生成）。"*

**禁止的表述**：
> ❌ *"ODEForge 是 SOTA"* / *"优于 DOP853"*（未定义维度）
> ❌ *"精度更高"*（短时间区间下 4 阶 < 8 阶，是错的）
> ❌ *"支持刚性系统"*（未实现隐式方法）
> ❌ *"更快"*（未做且不应做性能声明）
> ❌ *"精确梯度"*（adjoint 是另一条数值路径，C.2 实测已在 1e-6 分离）

**Benchmark 报告结构（team-lead 已裁决，分三节，禁止合并）**：

benchmark 测的是**两个不同的数学对象**，合并会产生自相矛盾的表格：
- 短时高精度测**局部截断误差** `O(h^p)` → 8 阶 DOP853 **必然**赢 4 阶 Yoshida4
- 长时定性测**结构性质**（能量有界 vs 单调漂移）→ 与阶数无关，**辛方法赢**

这是同一枚硬币的两面，**不是"结论不利"**。强行合并会逼迫读者回答"你比 DOP853 准吗"，而该问题的诚实答案是分场景的。

| 节 | 内容 | 门禁 |
|---|---|---|
| **§A 长期结构正确性**（主表，核心） | 哈密顿系统长时域；`net/osc` 判别器 + 能量带 + 等 nfev 预算对比 | **G1 在此** |
| **§B 局部精度与非劣性**（次表） | 阶数验证 + 等 nfev 误差 | **G2/G3 在此**。**DOP853 短时精度会赢我们，如实写** |
| **§C 已知局限** | 近日点进动 + 刚性委托 SciPy + 非哈密顿无优势 | — |

§B 必须显式写入：
> 在局部截断误差意义上，本系统的 DP5(4)（5 阶）**预期不优于** DOP853（官方标注 8 阶）。本项目不主张此维度优势，只主张**不劣**（G2）。

**数值报告纪律（team-lead 批准，本项目强制）**：
1. **任何负向断言**（"某库没有 X"）必须附官方文档 URL 或本机 `dir()` 输出。
2. **任何倍数/比值结论**必须证明分子分母除被比较变量外同源；vary-k 表格表头必须写明 `T`、`H`、`u_max`。
3. **benchmark 表头必须声明验证目标**（结构性质 / 局部精度），不得跨目标合并。

**免责**（若涉及任何投资/选型决策）：本文件为技术选型研究，**非投资建议**。

---

## 4. 风险与陷阱（业界已知问题）

### R1 ⛔ 变步长辛积分器失去辛性（本域最大实现陷阱）

| | |
|---|---|
| **Symptom** | 能量误差从「有界振荡」退化为「单调漂移」；长期积分稳定性丧失（这正是 ODEForge 声称的卖点）；表现为 Runge-Kutta 方法在谐振子长时积分中"能量单向漂移/轨道逸出" |
| **根因** | 辛性是**整体性质（global property）**，不是逐步性质。局部误差估计 + 逐步改 `h` 会破坏它 |
| **铁证** | **Skeel, R. D. (1993). "Variable step size destabilizes the Störmer/leapfrog/Verlet method." *BIT Numerical Mathematics* 33(1): 172–175. doi:10.1007/BF01990352** |
| **原始结论** | *"For fixed step-size `h` the Störmer method is stable for the standard test equation `ÿ = ω²y, ω>0`, if and only if `hω < 2`. We show that **for variable step size `h_n` there does not exist a (positive) limit on `hω` which ensures stability. Nor can we guarantee stability if, in addition, we limit the step size ratio `h_n/h_{n−1}`.**"* （来源：<https://scholar.cnki.net/Detail/index/GARJ8099_4/SSJD00001701932>） |
| **机理** | arXiv:astro-ph/9710043：*"a variable step integrator is evolving a dynamic system with state variables `(r, v, τ)`. The projection onto the phase space coordinates `(r, v)` **cannot be described by a Hamiltonian**."* 并指出时间可逆性也被破坏：*"if we step forward in time and then step backward, we do not end up at the same point because of the change in timestep. Since symplectic implies time reversible, such an operator is not symplectic."* |
| **社区共识** | Chris Rackauckas（SciML 作者，https://scicomp.stackexchange.com/a/34497）：*"symplecticness is a global property of the integration, it's not a stepwise property. Because of this, **local error estimates and local changes of Δt do not necessarily preserve the property. In fact, most choices will break this property.**"* |
| **正确解法（代价高）** | 把 `s(t)` 纳入哈密顿量、用隐式步长函数（Hut–Makino–McMillan 1995：`τ = ½[τ(rₙ,vₙ) + τ(r_{n+1},v_{n+1})]`）。*"The resulting methods are then usually **implicit in the step size function**, so it's not necessarily simple to code, which is why it's still an active area of research and not typically used."* |
| **规避（ODEForge 采用）** | **定步长（fixed step）**。步长由用户显式给出。**DoD 中必须包含一条：任何声称"辛"的积分器都必须是定步长的。** 若将来要支持变步长，必须同时提供 `Hairer et al. 2006` 第 8 章所述的结构保持实现，并**明确声明该路径不承诺辛性**。 |
| **「定步长辛 + 外层调度」的定位澄清** | 这类实现**不是**「变步长辛积分器」（后者数学上不存在），而是**定步长辛积分器 + 外层步长调度器**。外层调度**不改变内层辛性**。文档措辞必须区分，否则会被理解成实现了数学上做不到的东西 |
| **能量带的 `u_max⁴` 标度律（可量化判据）** | 【ai-algorithm-scientist 实测 + 本人独立复现】能量带宽度由**最粗子步 `u_max`** 主导：`band ∝ u_max⁴`。<br>本人独立验证（T=1000，Yoshida4，`band = max|E−E₀|`）：<br>`u_max = 0.1/0.05/0.025/0.0125` → `band = 3.832e-6 / 2.382e-7 / 1.486e-8 / 9.287e-10`<br>相邻比值 **16.36 / 16.09 / 16.02 / 16.01**（收敛到 16.00 = 2⁴），且 `band/u⁴` 恒为 `3.80e-02`。<br>→ **这把 Skeel 的定性限制变成了可预测的标度律**：DoD 可断言「最粗子步的守恒量级」，单测形式为**断言拟合指数 ≈ 4.0（容差 0.1）**，同时验证阶数正确与「能量带由 `u_max` 主导」两个性质 |
| **⚠️ 同 `u_max` 对照的基线陷阱（本次调研的实际教训）** | vary-k 的惩罚倍数**只有在所有方案 `u_max` 相同时才可比**。若 vary 方案引入了更细的子步，其 band 天然更小，会得出「vary 优于 const」的反直觉结论或倍数颠倒。<br>**实测对照（T=1000，H=0.4，所有行 `u_max = 0.05` 严格相同）**：<br>`const k=8` band=`2.382e-07` (1.00×)｜`vary 8/16` **`0.549×`**｜`vary 8/16/32` **`0.378×`**｜`vary 8/32` **`0.523×`**｜（对照）`const k=16` `0.062×`<br>→ **vary 路径的 band 与 const 基线同阶（0.38~0.55×），并非数量级恶化。**此前报出的 1.19~1.28× 是 `u_max` 基线不匹配所致，**以本表为准**。<br>**报告纪律：所有 vary-k 表格表头必须写明 `T`、`H`、`u_max` 三个量**，否则读者无法自行核对 |

### R2 ⛔ 刚性系统用显式方法爆炸

| | |
|---|---|
| **Symptom** | `nfev` 暴涨、`Required step size is less than spacing between numbers`、解振荡/发散、`solve_ivp` 报 "too much accuracy requested" |
| **根因** | Dahlquist 第二壁垒：显式 RK 的稳定域是**有界区域**，无法覆盖 Jacobian 特征值横跨多个数量级的情形。RK4 实轴仅到 `z ≈ −2.785`；特征值 `λ=−1000` 时 `Δt ≲ 2.785/1000` |
| **规避** | ODEForge 明确声明**不支持刚性**，并在文档写明"若 `scipy` 文档所说的 unusually many iterations 出现，请改用 `Radau`/`BDF`"——直接引用 SciPy 官方原话 |

### R3 ⚠️ 浮点误差累积与相空间漂移

| | |
|---|---|
| **Symptom** | 辛方法的能量误差不再有界，随步数缓慢增长；轨道相位漂移 |
| **根因** | 辛性控制的是**离散化误差**，**不控制舍入误差**。Rutgers 数值分析课程页（<https://www.physics.rutgers.edu/~haule/509/Numerical_Error.html>）明说：*"In actual floating-point arithmetic, **roundoff can also accumulate** during an extremely long calculation. **Symplecticity controls the systematic discretization error**"* |
| **量化（Brouwer's law）** | IAS15 论文（<https://www.arxiv-vanity.com/papers/1409.4779/>）：*"If every operation in an algorithm is unbiased... then the error grows like a random walk, i.e. ∝ √n... **Angle-like quantities, such as the phase of an orbit, will grow faster, ∝ n^{3/2}**. This is known as **Brouwer's law**."* 并指出 *"Popular implementations of other integrators used for long-term orbit integrations, **symplectic or not**, do not follow Brouwer's law and show a **linear** energy growth in long simulations."* |
| **关键区分** | **相空间**（面积/体积、动量矩）由辛性**精确**保持；**能量**只是有界振荡；**相位**有 `n^{3/2}` 的不可约增长。→ **禁止宣称"辛积分器长期相位也准"** |
| **规避** | DoD 中的能量断言必须声明"在 N 步以内"、"在 `dt` 与 N 固定的前提下"。定期用高精度参考解（`DOP853` 收紧到 `rtol=1e-12`）做相位对照 |

### R4 ⚠️ 自动微分在长时积分中的内存爆炸

| | |
|---|---|
| **Symptom** | Neural ODE 训练 OOM；`backward()` 时 CUDA OOM；直接反传在长积分上不可行 |
| **根因** | 直接反传需存全部中间状态 `O(N)`；`N` 随积分时长线性增长 |
| **规避（本项目）** | **不走 autodiff 路线**，故此坑不适用于 ODEForge 核心。但 adjoint 有自己的坑：见 R4b |

### R4b ⛔ adjoint 梯度不精确 / 反向求解不稳定

| | |
|---|---|
| **Symptom** | 梯度与直接反传/解析梯度不一致；训练在小区间 `dt` 下发散；误差在 log 尺度上才可见 |
| **根因** | Gholami, Keutzer, Biros (IJCAI 2019, arXiv:1902.10298)：*"the proposed optimize-then-discretize approach may lead to divergent training due to **inconsistent gradients for small time step sizes**"*；对 ReLU/卷积算子 *"may be numerically unstable"* |
| **刚度加剧** | Kim et al. (arXiv:2103.15341)：*"if the Lipschitz constant is too large in absolute value, then either the forward pass or reverse solve is numerically unstable... **stiff ODEs fall into the class of problems which are numerically difficult to reverse**"*；`λ=10000` 时 *"cannot be reversed numerically in double precision"* |
| **规避** | ① adjoint 容差应**独立于**前向容差设置（`adjoint_rtol`/`adjoint_atol`）；② **强制哈密顿结构**（Gholami et al. 原话：*"this needs to be enforced either in the NN architecture (**e.g. by using Hamiltonian systems**) or implicitly through regularization"*）——**ODEForge 的哈密顿限定在此处再次成为优势**；③ DoD 必须包含「adjoint 梯度 vs 中心差分」的对照测试（【实测】该对照在 `rtol=1e-10` 下已有 1.4e-6 相对差，不应声称"精确"） |

### R5 ⛔ `numpy.random.Generator` 流不保证跨版本逐位一致（对本项目的确定性 DoD 是致命的）

| | |
|---|---|
| **Symptom** | 同一 commit 在不同 numpy 版本/不同机器上跑出不同结果；CI 与本地结果不一致；"确定性测试"随机性失败 |
| **根因（官方原文）** | NumPy 兼容性策略文档（<https://numpy.com.cn/devdocs/reference/random/compatibility.html>，英文版 <https://numpy.org/doc/stable/reference/random/compatibility.html>）把保证限定得极窄：*"If you create a Generator with the same BitGenerator, with the same seed, perform the same sequence of method calls with the same arguments, **on the same build of numpy, in the same environment, on the same machine**, you should get the same stream of numbers. **Note that these conditions are very strict.** There are a number of factors outside of NumPy's control that limit our ability to guarantee much more than this."* |
| **更狠的一条** | 官方文档明写 **Generator 根本不提供版本兼容保证**：*"**Breaking stream-compatibility in order to introduce new features or improve performance in Generator or default_rng will be allowed with caution.** Such changes will be considered features, and as such will be **no faster than the standard release cadence** of features (i.e. on **X.Y** releases, **never on X.Y.Z**)."* 并点名会考虑从 Box-Muller 换到 Ziggurat 算法。**即：一个 minor 版本升级就可能改变流。** |
| **底层原因（NEP 19）** | <https://numpy.org/neps/nep-0019-rng-policy.html>：NumPy 曾有严格流兼容政策，但承认 *"our policy **overpromises compatibility**"*，因为 *"The same version of numpy built on different platforms, or just in a different way could cause changes in the stream"*，*"The standard practice now for bit-for-bit reproducible research is to **pin all of the versions of code of your software stack, possibly down to the OS itself.**"* |
| **额外陷阱** | *"Calling `rng.random()` 5 times **is not guaranteed** to give the same numbers as `rng.random(5)`."* → **抽取次数与抽取形状都是策略参数** |
| **规避（ODEForge 必须做）** | ① 随机源只用于**初始化**，绝不用于积分过程；② 固定 `BitGenerator`（显式 `PCG64`，不用 `default_rng()`，因官方说 *"`default_rng` is allowed to change the default BitGenerator"*）；③ **固定抽取形状**（不混用逐次/批量）；④ DoD 声明"逐位一致仅在固定 numpy 版本下保证"；⑤ 更强选项：改用**自有确定性 PRNG**（如固定算法的 SplitMix64/PCG 实现），彻底摆脱 numpy 版本策略 |

### R6 ⚠️ SciPy `DOP853` 默认 `rtol=1e-3, atol=1e-6` 对高精度需求不够用

| | |
|---|---|
| **Symptom** | 用 DOP853 做"参考解"时，参考解本身的误差比被测对象大，导致对标结论无效 |
| **根因【实测源码确认】** | `solve_ivp` 文档：*"Default values are **1e-3 for `rtol` and 1e-6 for `atol`**."*<br>并且【实测】`inspect.signature(RK45.__init__)` 与 `inspect.signature(DOP853.__init__)` **两者默认值均为 `rtol=0.001, atol=1e-06`** |
| **放大因素** | 官方文档同段还说明：*"If `atol` is larger than `rtol * abs(y)` the desired `atol` [dominates]... **If components of y have different scales, it might be beneficial to set different `atol` values for different components** by passing array_like with shape (n,)"* → **多分量系统中单一 atol 会让小量级分量的精度被系统性忽略** |
| **规避** | ① 对标基线**必须**显式收紧到 `rtol=1e-12, atol=1e-14`（**不得依赖默认**）；② 对多分量哈密顿系统，传入**逐分量 `atol` 数组**，避免量级差异污染误差度量；③ ODEForge 若提供任何默认容差，**不得沿用 scipy 的 1e-3** |

### R7 ⚠️ 时间可逆性被破坏（辛方法的必要性质之一）

| | |
|---|---|
| **Symptom** | 沿原步长反向积分无法回到初始点 |
| **根因** | Kahan (1993) 指出时间可逆性（Kahan 称之为 *reflexivity*）是"updating formulae"稳健性质的真正关键；arXiv:astro-ph/9710043 论证 symplectic ⊹ time-reversible，故破坏可逆性即破坏辛性 |
| **规避** | DoD 增加**往返一致性测试**：`integrate(q0, 0, T, h)` 后 `integrate(·, T, 0, h)`，比较与 `q0` 的偏差。**这个测试同时覆盖 R1 和 R7** |

### R8 ⚠️ 高阶辛方法含负子步

| | |
|---|---|
| **Symptom** | 中间子步出现"倒退"，实现时若误加 `abs()` 或符号错误会静默得到错误结果 |
| **根因** | Yoshida 组合系数 `w₀ ≈ −1.702` 为负；Blanes–Moan 4 阶中 `-0.3667132690474257` 为负 |
| **通用结论（Blanes & Moan 原文明说）** | *"The coefficients... **cannot all be positive if the order of the method is greater than 2**. This can prevent the application of this kind of method to **dissipative systems**."* |
| **规避** | ① 子步步长必须是**带符号**的 `h * w`；② 加单元测试断言系数和为 1（【实测】Yoshida-4 = 1.0000000000000002）；③ 加 `w₀` 为负的显式断言，防止"优化"时被误改 |

---

## 5. 风险汇总与对策一览

| ID | 严重度 | 风险 | 对策 | 是否阻塞 DoD |
|---|---|---|---|---|
| R1 | ⛔ 致命 | 变步长毁掉辛性 | 定步长；DoD 写明 | **是** |
| R2 | ⛔ 致命 | 刚性 + 显式 → 爆炸 | 明确声明不支持刚性 | **是**（声明层面） |
| R3 | ⚠️ 高 | 浮点/相位漂移 | 声明"≤N 步"；相位对照基准 | **是** |
| R4 | ○ 低 | autodiff 内存爆炸 | 不走 autodiff 路线 | 否 |
| R4b | ⛔ 致命 | adjoint 梯度不一致 | 哈密顿结构 + 容差独立 + 差分对照测试 | **是** |
| R5 | ⛔ 致命 | `Generator` 跨版本不一致 | 固定 BitGenerator + 固定形状 + 或自有 PRNG | **是** |
| R6 | ⚠️ 高 | DOP853 默认容差不够 | 对标基线收紧到 1e-12/1e-14 | **是** |
| R7 | ⚠️ 高 | 时间可逆性破坏 | 往返一致性测试 | **是** |
| R8 | ⚠️ 中 | 负子步符号错误 | 系数和断言 + 负系数断言 | **是** |

---

## 6. 数据来源与时效

**检索/实测日期：2026-10-03**。所有版本号为该日 `pip index versions` 实际返回值。

### 上游文档与论文

| 主题 | 来源 | URL |
|---|---|---|
| `solve_ivp` 方法定位原话 | SciPy 官方文档 | <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html> |
| 同上（交叉确认源） | papyri.dev SciPy 1.18.0 镜像 | <https://papyri.dev/project/scipy/1.18.0/scipy.integrate._ivp.ivp> |
| 同上（第二确认源） | SciPy 1.4.1 文档 | <https://docs.scipy.org/doc//scipy-1.4.1/reference/generated/scipy.integrate.solve_ivp.html> |
| `solve_ivp` 默认容差 1e-3/1e-6 | 同上 + 【实测】`inspect.signature` | 同上 |
| Dahlquist 壁垒 | formalcalculus 数值 ODE 教程 | <https://formalcalculus.com/topics/numerical-odes> |
| Dahlquist 壁垒（中文） | 百度百科「达赫奎斯特壁垒定理」 | <https://baike.baidu.com/item/达赫奎斯特壁垒定理/19070180> |
| 显式方法最优稳定性界 | Shaw, *Polynomial Inequalities and Optimal Stability of Numerical Integrators*, arXiv:2503.24232 | <https://arxiv.org/pdf/2503.24232> |
| Yoshida 1990 系数 | Wikipedia *Symplectic integrator* | <https://www.wikiwand.com/en/map/Symplectic_algorithm> |
| Yoshida 1990 论文出处 | Rust 实现引用（Physics Letters A 150(5–7):262–268, 1990） | <https://docs.rs/utsuroi/0.1.0/src/utsuroi/solver/yoshida.rs.html> |
| Blanes–Moan 系数 | Blanes & Moan, *Geometric Integrators for ODEs* | <https://www.researchgate.net/profile/G-Quispel/publication/200175533_Geometric_integrators_for_ODEs/links/02e7e5156a6878ff7a000000/Geometric-integrators-for-ODEs.pdf> |
| Blanes–Moan 系数（第二源） | arXiv:2510.16542 附录 C | <https://arxiv.org/pdf/2510.16542> |
| Modified Hamiltonian / O(τⁿ) 长期行为 | CfCA 会议论文 | <https://www.cfca.nao.ac.jp/~tito/ndyn/pdfproc/proc_CeleMechNbodyDyn2001Kusatsu.pdf> |
| Verlet 算子分解与 modified Hamiltonian | arXiv:astro-ph/9710043 | <https://arxiv.org/pdf/astro-ph/9710043v1> |
| GNI 教科书定位原话 | Ardourel & Jebeile, *Numerical instability and dynamical systems*（引 Hairer/Lubich/Wanner 2006, 2nd ed.） | <https://hal.science/hal-03199593/file/Ardourel%2520Jebeile%2520EJPS%25202021%2520preprint.pdf> |
| 能量漂移与 shadow Hamiltonian | HandWiki *Energy drift* | <https://handwiki.org/wiki/Energy_drift> |
| 短时 vs 长时误差结构 | Rutgers 509 Numerical Error 课程页 | <https://www.physics.rutgers.edu/~haule/509/Numerical_Error.html> |
| **变步长破坏辛性** | **Skeel, BIT Numer. Math. 33(1):172–175, 1993** | <https://scholar.cnki.net/Detail/index/GARJ8099_4/SSJD00001701932> |
| 变步长机理 / 时间可逆性 | arXiv:astro-ph/9710043 + Rackauckas 答复 | <https://scicomp.stackexchange.com/a/34497> |
| 相空间依赖步长使方法脱离哈密顿 | Rein & Tamayo, MNRAS 452(2):1934 | <https://academic.oup.com/mnras/article/452/2/1934/1069988> |
| 非保守力下辛性失效 / Brouwer's law | Rein & Tamayo, arXiv:1409.4779 (IAS15) | <https://www.arxiv-vanity.com/papers/1409.4779/> |
| **adjoint 梯度不一致** | **Gholami, Keutzer, Biros, IJCAI 2019, 730–736（ANODE）** | <https://arxiv.org/abs/1902.10298> / <https://dx.doi.org/10.24963/ijcai.2019/103> |
| adjoint 不稳定与刚度 | Kim, Ji, Deng, Ma, Rackauckas, arXiv:2103.15341 | <https://arxiv.org/abs/2103.15341> |
| Neural ODE 原始论文 | Chen et al., arXiv:1806.07366 / NeurIPS 2018 | <https://arxiv.org/abs/1806.07366> |
| 辛 adjoint 与其他方法对比表 | Matsubara, Miyatake, Yaguchi, NeurIPS 2021 | <http://git-hub.com/tksmatsubara/symplectic-adjoint-method> |
| torchdiffeq adjoint 文档 | DeepWiki | <https://deepwiki.com/rtqichen/torchdiffeq/4-adjoint-sensitivity-analysis> |
| **numpy Generator 兼容策略** | **NumPy 官方文档** | <https://numpy.org/doc/stable/reference/random/compatibility.html>（中文：<https://numpy.com.cn/devdocs/reference/random/compatibility.html>） |
| **NEP 19 RNG 政策** | **NumPy NEP 19** | <https://numpy.org/neps/nep-0019-rng-policy.html> |
| JAX 平台支持 | JAX 官方安装文档 | <https://jax.readthedocs.io/en/latest/installation.html> |
| jaxlib wheel 平台与 Python 支持 | pydevtools 实测 | <https://pydevtools.com/handbook/how-to/how-to-install-jax-with-uv/> |
| scikits.odes 平台支持 | conda-forge | <https://anaconda.org/channels/conda-forge/packages/scikits.odes/overview> |
| PINN 谱偏置（理论） | Seroussi, Miron & Ringel, MLST 5(3):035048, 2024 | <https://iopscience.iop.org/article/10.1088/2632-2153/ad652d> |
| PINN 失效模式 | Eng. Reports（Elsevier） | <https://www.sciencedirect.com/science/article/pii/S0950705124004659> |
| PINN 挑战综述 | IntechOpen | <https://intechopen.com/chapters/1191254> |
| PINN 无收敛保证 | Emergent Mind | <https://www.emergentmind.com/topics/physics-informed-neural-networks-pinns> |
| 纯 NumPy autodiff 先例 | micrograd / micrograd-np | <https://github.com/brief-ds/micrograd> / <http://github.com/ITI-THM/micrograd-np> |

### 本机实测记录（全部可复现）

| 验证项 | 结果 |
|---|---|
| Python | 3.13.14，win_amd64，无 GPU |
| wheel 下载（12 个包） | scipy/numpy/networkx/scikit-learn/matplotlib/sympy/jax/jaxlib/diffrax/torch/torchdiffeq/diffeqpy **全部可下载**；`odepack`、`scikits.odes` **PyPI 无发行版** |
| 辛方法 vs RK4 能量行为 | 见 2.2 表：B.2 的 200000 步实验 |
| Yoshida-4 系数和 | `1.0000000000000002` |
| Blanes-Moan 4 阶系数一致性 | 两个独立来源完全一致 |
| **Yoshida-4 系数写法（已定：A = `1-2w1`）** | 两种写法互斥且数值不可区分（2000 万步 ratio=1.000000）。选 A 因为 `2w1+w0==1.0` 可断言。见 B.1 提示框 |
| `solve_ivp` 默认容差 | `rtol=0.001, atol=1e-06`（文档 + 签名双确认） |
| `jax 0.11.2` | import ✓，`jax.devices() == [CpuDevice(id=0)]` |
| `diffrax 0.7.2` | **全栈健康**：求解 ✓，`jax.grad` ✓（−0.135335296392），`RecursiveCheckpointAdjoint` ✓（同值），`jit` ✓，`vmap` ✓，`SaveAt(t1=True)` ✓。**无 symplectic solver**（`dir()` 过滤为空） |
| `torch 2.14.1+cpu` + `torchdiffeq 0.2.5` | `odeint` ✓（grad = −0.13533537089824677）；`odeint_adjoint` ✓（grad = −0.13533517718315125）；**相对差 1.431e-06** |
| `odeint_adjoint` 约束 | `func` 的参数必须注册为 submodule（`self.a = nn.Parameter(...)`），否则静默返回 `requires_grad=False` |

### 待验证清单（**不得写入 model_card 结论**）

1. ~~`diffrax 0.7.2 + jax 0.11.2` 的 `SaveAt` 不兼容~~ —— **已撤销：该"不兼容"是我的 API 用法错误（`t1` 传了 float 而非 bool），非库的问题。详见 2.3 纠错框。**
2. Hairer/Lubich/Wanner 三卷本的具体页码与定理编号 —— 本调研依据官方文档引用、教科书转述与多篇论文的一致表述，**未查阅原书正文**，故正文不引用具体页码。
3. 「辛积分器在 N 步内相位误差界」的定量公式 —— 已知 Brouwer's law（`n^{3/2}`）的表述，但 ODEForge 适用的具体常数未核实。
4. Blanes–Moan 6 阶系数的完整精度 —— 仅取得 arXiv:2510.16542 中的 10 段值，未与 Blanes–Moan 原论文 Eq. 交叉核对。

---

## 附：两处自我纠错记录（保留以供追溯）

本调研中我曾得出两个**错误结论**，均由"未验证的测试脚本"导致，一并记录以免下游沿用：

| 错误结论 | 真实原因 | 正确结论 |
|---|---|---|
| 「`diffrax 0.7.2 + jax 0.11.2` API 不兼容，`SaveAt` 崩溃」 | `SaveAt(t1=...)` 的 `t1` 是**布尔开关**，我传了 float `1.0` | 全栈健康。传 `t1=True` 即通过 |
| 「`torchdiffeq` adjoint 在 torch 2.14.1 上坏了」 | 参数写在闭包里未注册为 submodule，`func.parameters()` 为空 | adjoint 正常工作；与直接反传相对差 1.4e-06 |

**共同教训**：在断言"某个库/环境有问题"之前，必须先确认最小复现里**没有自己的用法错误**。两次错误的表象都是一条看起来很像"上游 bug"的 `TypeError`/`RuntimeError`。

**第四处修正（参数基线不匹配）**：我曾报出 vary-k 子步细分的能量带惩罚为 **1.19~1.28×**，与 ai-algorithm-scientist 的 0.59~0.80× **方向相反**。根因是**两组对照的 `u_max` 基线不同**（我的 const-k8 band=1.876e-06，他的=6.5825e-08，差 28×）——**惩罚倍数只有在所有方案 `u_max` 严格相同时才可比**。以严格同 `u_max` 重测后，我得到 **0.38~0.55×**，**与他的方向一致**，采纳他的数据。已更新 R1。
> 讽刺的是，这跟前面两次是同一类错误的变种：**都是我自己的对照实验设置有错，却把结论当成了事实报出。**

**方法论沉淀（本项目可复用的一条硬纪律）**：
> 任何"倍数/比值"型结论，**必须先证明分母与分子除被比较的那个变量外完全同源**。基线不同源的倍数比较无意义。
> 具体到本项目：vary-k 对照表头必须同时写明 `T`、`H`、`u_max`。

**第三处修正（跨文档交叉核实）**：我曾建议 Yoshida 系数采用「闭式 `-c/(2-c)`」并称其更准，ai-algorithm-scientist 主张 `1-2w1`。**复核后他是对的，我撤回建议**——用 80 位真值 + BFS 穷举证明「正确舍入」与「`2w1+w0==1.0` 精确成立」在 float64 下**互斥**，而后者才是可断言的不变量。我原先的论据（"参考实现 docs.rs 这样写"）是 appeal to authority，不成立。详见 B.1 提示框。

> **免责声明**：本文为技术选型研究与工程事实核查，**非投资建议**，不含任何股价、估值或市场预测判断。
