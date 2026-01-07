
# Signal Model Formulation

以下是一个 2D 相控阵雷达的 RX 信号模型。

下面给出重新整理后的、可直接用于仿真的 **(X[p,q,n,m])** 信号模型。

---

## 1) 几何与符号

### 2D URA 阵列

* 阵列尺寸：(P\times Q)
* 阵元索引：(p=0..P-1,\ q=0..Q-1)
* 阵元间距：(d_x, d_y)
* 波长：(\lambda=c/f_c)

阵元位置：
[
\mathbf{r}_{p,q}=[p,d_x,;q,d_y,;0]^T
]

### 角度与方向余弦（CPI 内常量）

* azimuth：(\theta_\ell)
* elevation：(\alpha_\ell)（从水平面向上）

方向余弦：
[
u_\ell=\cos\alpha_\ell\cos\theta_\ell,\qquad
v_\ell=\cos\alpha_\ell\sin\theta_\ell
]

空间相位（阵列流形项）：
[
\phi_{p,q}(\theta_\ell,\alpha_\ell)=\frac{2\pi}{\lambda}\Big(p,d_x,u_\ell+q,d_y,v_\ell\Big)
]

---

## 2) 波形与采样（一个 CPI）

* 单脉冲复基带：(s(t))
* PRI：(T_r)，PRF：(f_r=1/T_r)
* CPI 内脉冲数：(M)
* fast-time 采样率：(F_s)
* fast-time 采样点：(t_n=n/F_s,\ n=0..N-1)
* slow-time 脉冲时刻：(t_m=mT_r,\ m=0..M-1)

---

## 3) 目标运动（CPI 内近似匀速，仅影响 Doppler 与轻微距离漂移）

你说“移动很小”，通常意味着：

* **幅度 (A_\ell) 不变**
* **角度不变**
* **多普勒频率近似常量**
* （可选）距离随 (m) 线性变化，产生一个慢时间相位或 range walk；最简模型可忽略或保留

径向速度 (v_{r,\ell})（常量）对应 Doppler：
[
f_{D,\ell}=\frac{2v_{r,\ell}}{\lambda}
]

---

## 4) 接收基带离散信号模型（核心）

令双程时延在 CPI 内近似常量（最简）：
[
\tau_\ell \approx \frac{2R_\ell}{c}
]
（如果你愿意保留轻微 range-walk，可以把 (\tau_\ell) 改成 (\tau_\ell(m))，见文末备注。）

那么 **每个阵元、每个脉冲、每个快时间采样**的复基带观测为：

[
\boxed{
X[p,q,n,m]=
\sum_{\ell=1}^{L}
A_\ell;
s!\left(t_n-\tau_\ell\right);
e^{j2\pi f_{D,\ell}, mT_r};
e^{-j\phi_{p,q}(\theta_\ell,\alpha_\ell)}
;+;W[p,q,n,m]
}
]

其中：

* 这里 \ell 的意思是 \ell-th target, 一共有 L 个点目标
* (A_\ell\in\mathbb{C})：复散射系数（CPI 内常量，可包含 (1/R^2)、RCS、TX/RX 增益等）
* (s(t_n-\tau_\ell))：延迟后的发射波形（需要分数延迟插值/滤波）
* (e^{j2\pi f_{D,\ell} mT_r})：慢时间 Doppler 相位推进
* (e^{-j\phi_{p,q}(\theta_\ell,\alpha_\ell)})：二维阵列空间相位
* (W[p,q,n,m]\sim\mathcal{CN}(0,\sigma^2))：复高斯噪声

把空间相位展开写得更“代码友好”一点就是：

[
e^{-j\phi_{p,q}}=
\exp\left(
-j\frac{2\pi}{\lambda}
\left(
p,d_x,\cos\alpha_\ell\cos\theta_\ell
+
q,d_y,\cos\alpha_\ell\sin\theta_\ell
\right)
\right)
]

---


## 6) 保留 CPI 内的轻微距离漂移（range walk）

把 (\tau_\ell) 改成：
[
\tau_\ell(m)=\frac{2(R_\ell+v_{r,\ell} mT_r)}{c}
]
则模型变为：


[
\boxed{
X[p,q,n,m]=
\sum_{\ell=1}^{L}
A_\ell;
s!\left(t_n-\tau_\ell(m)\right);
e^{j2\pi f_{D,\ell}, mT_r};
e^{-j\phi_{p,q}(\theta_\ell,\alpha_\ell)}
;+;W[p,q,n,m]
}
]


（这时 Doppler 既会体现在慢时间相位，也会体现在“回波包络随 (m)”的轻微滑动上；做高保真仿真时可以考虑。）

---

# 任务描述
根据 signal model formulation。写一个 python 的 module 来 implement 这个信号模型。
要求：
- baseband signal 是一个 chirp signal with certain duration and chirp rate. 
- 首先 figure out 这个信号模型需要什么样的参数，然后用户可以根据需要定义这些参数
- 然后，implement 这个信号模型
- 接下来，设计一些 tests, 来对你的 implementation 进行一个 sanity check。确保 implementation is correct. 测试代码可以画出一些 rx 信号的时域和频域的图，看看频率和 delay 是否正确。
- 信号模型的代码放在 simulations 这个文件夹里，测试代码 tests 这个文件夹里

使用 packages: 使用 numpy 来作为主要计算工具。