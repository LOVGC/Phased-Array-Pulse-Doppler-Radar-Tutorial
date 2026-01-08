# 2D Phased array signal model

ref: https://chatgpt.com/share/695f2292-a288-8007-9c37-e89e61a0f1e9
---

## A) 参数与符号（全部先列出）

### A1. 阵列几何（2D UPA）

* 阵列尺寸：(P\times Q)
* 阵元索引：(p=0,\dots,P-1), (q=0,\dots,Q-1)
* 阵元间距：(d_x,\ d_y)
* 阵元位置向量（以阵列参考点/相位中心为原点）：
  [
  \mathbf r_{p,q}=\begin{bmatrix}p,d_x\ q,d_y\ 0\end{bmatrix}
  ]

### A2. 角度参数（目标方向）

* 第 (\ell) 个目标：azimuth (\theta_\ell)，elevation (\phi_\ell)
* 对应单位方向向量（从阵列指向目标）：
  [
  \mathbf u(\theta,\phi)=
  \begin{bmatrix}
  \cos\phi\cos\theta\
  \cos\phi\sin\theta\
  \sin\phi
  \end{bmatrix},\qquad
  \mathbf u_\ell=\mathbf u(\theta_\ell,\phi_\ell)
  ]

### A3. 雷达波形与体制（pulse-Doppler LFM）

* LFM 脉冲脉宽：(T_p)
* LFM 带宽：(B)
* chirp rate：(\mu=B/T_p)
* 载频：(f_c)
* 光速：(c)
* 波长：(\lambda=c/f_c)
* 基带采样率：(f_s)
* PRI：(T_r)
* CPI 内脉冲数：(M)

### A4. 目标参数（点目标，(L) 个）

* 目标数：(L)
* 第 (\ell) 个目标的空间位置：(\mathbf r_\ell(t))
* 阵列参考点到目标的瞬时斜距（**定义**）：
  [
  \boxed{R_\ell(t)\triangleq |\mathbf r_\ell(t)|}
  ]
* 径向速度（CPI 内近似常数）：
  [
  v_\ell=\frac{dR_\ell(t)}{dt}
  ]
* 两程多普勒频率：
  [
  f_{D,\ell}=\frac{2v_\ell}{\lambda}
  ]
* 目标复散射系数（CPI 内常数）：
  [
  \alpha_\ell\in\mathbb C
  ]
  （包含 RCS、传播损耗、系统增益、初始相位等）

### A5. 噪声

* RX complex baseband 噪声：(n[p,q](t))（常用模型：复高斯白噪声）

---

## B) 假设（全部先写清）

1. **远场/平面波假设**：目标到阵列的波前在阵列孔径内近似为平面波。
2. **角度常量假设**：在一个 CPI 内，((\theta_\ell,\phi_\ell)) 不随时间变化。
3. **点目标假设**：每个目标为单散射中心（无扩展散射）。
4. **速度常量近似**：在 CPI 内 (v_\ell) 近似常数，常用 (R_\ell(t)\approx R_{\ell,0}+v_\ell t)。
5. **单程 RX steering（你指定）**：只考虑**接收端**阵列几何导致的时延/相位；**不考虑发射端阵列几何**对波形的时延/相位（相当于发射为各向/单通道参考点发射）。
6. （可选）**窄带/小孔径近似**：若需要可进一步令波形包络对阵元时延不敏感，把阵列影响仅放入 steering 相位（后面会单独给出“可选简化式”）。

---

## C) 信号模型（按链路顺序逐步给出）

### C1) TX complex baseband（连续时间）

单个 LFM 脉冲（示例）：
[
s(t)=\operatorname{rect}!\left(\frac{t}{T_p}\right)\exp!\big(j\pi\mu t^2\big),\qquad \mu=\frac{B}{T_p}
]
CPI 内脉冲串：
[
\boxed{
x_{\text{tx}}(t)=\sum_{m=0}^{M-1} s(t-mT_r)
}
]

---

### C2) TX passband（RF）

[
\boxed{
x_{\text{RF}}(t)=\Re\left{x_{\text{tx}}(t),e^{j2\pi f_c t}\right}
}
]

---

### C3) 传播时延（两程距离 + RX 单程阵列几何）

* 参考点两程时延：
  [
  \tau_\ell(t)=\frac{2R_\ell(t)}{c}
  ]
* RX 单程阵列几何附加时延（远场）：
  [
  \Delta\tau_{\ell,p,q}= -\frac{\mathbf u_\ell^{\mathsf T}\mathbf r_{p,q}}{c}
  ]
* 合并得到阵元 ((p,q)) 的等效时延（满足“只做 RX 单程 steering”）：
  [
  \boxed{
  \tau_{\ell,p,q}(t)=\frac{2R_\ell(t)}{c}-\frac{\mathbf u_\ell^{\mathsf T}\mathbf r_{p,q}}{c}
  }
  ]

---

### C4) RX passband（RF）在阵元 ((p,q))

[
\boxed{
r_{\text{RF}}[p,q](t)=
\sum_{\ell=1}^{L}\Re\left{
\alpha_\ell,x_{\text{tx}}!\big(t-\tau_{\ell,p,q}(t)\big),
e^{j2\pi f_c\big(t-\tau_{\ell,p,q}(t)\big)}
\right}+n_{\text{RF}}[p,q](t)
}
]

---

### C5) 下变频后的 RX complex baseband（连续时间最终式）

对 (r_{\text{RF}}) 乘 (e^{-j2\pi f_c t}) 并低通，得到：
[
\boxed{
x_{\text{rx}}[p,q](t)=
\sum_{\ell=1}^{L}
\alpha_\ell,
x_{\text{tx}}!\big(t-\tau_{\ell,p,q}(t)\big),
e^{-j2\pi f_c,\tau_{\ell,p,q}(t)}
+n[p,q](t)
}
]

把指数项分解成 “距离/多普勒项 × 接收 steering 项”：
[
e^{-j2\pi f_c,\tau_{\ell,p,q}(t)}
=================================

\underbrace{e^{-j\frac{4\pi}{\lambda}R_\ell(t)}}*{\text{两程距离相位（含多普勒）}}
\cdot
\underbrace{\exp!\Big(j\frac{2\pi}{\lambda}\mathbf u*\ell^{\mathsf T}\mathbf r_{p,q}\Big)}*{\text{RX 单程 steering}}
]
因此等价写成：
[
\boxed{
x*{\text{rx}}[p,q](t)=
\sum_{\ell=1}^{L}
\alpha_\ell,
x_{\text{tx}}!\big(t-\tau_{\ell,p,q}(t)\big),
e^{-j\frac{4\pi}{\lambda}R_\ell(t)}
\exp!\Big(j\frac{2\pi}{\lambda}\mathbf u_\ell^{\mathsf T}\mathbf r_{p,q}\Big)
+n[p,q](t)
}
]


---


# 任务要求

## 写一个 python 应用，依次 implment 信号模型：
- TX complex baseband（连续时间）
- 下变频后的 RX complex baseband（连续时间最终式）

## 要求是，用户可以 specify 参数：
- 阵列的 geometry
- 目标的参数：目标数目, 距离，角度，速度等
- 雷达波形与体制参数
- 噪声：可以加入噪声，也可以不加入噪声。如果用户想加入噪声，就用复高斯白噪声，而且用户还可以通过调整噪声的 std 来控制噪声的大小，i.e. 控制信噪比。

## 仿真时间窗口：时间就是一个 CPI 的时间，仿真的输出就是所有 antenna element 接收到的离散时间信号 X[p,q, fast_time, slow_time], 这个信号就是 (p,q)-th antenna 接收到的信号，并且以 fast_time, slow_time 组成。

## 测试代码：
- 测试 TX complex baseband（连续时间），使用一个具体的信号参数测试，画出信号在一个 PRI 时间窗的时域图和频域图，看看信号的带宽和 duration 是不是对的。
- 测试下变频后的 RX complex baseband（连续时间最终式），使用一个具体的信号参数以及雷达体制和目标参数测试，然后画出每一个天线在一个 CPI 时间窗的时域图，和频域图，看看信号的带宽和 duration 是不是对的。
- 画图不用存，直接以 interactive 的形式 show 出来

## 代码组织：
把仿真代码写成一个 package, i.e. 放在一个 folder 里面。
把 test 代码放在另外一个 folder 里面。