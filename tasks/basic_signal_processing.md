有了每个阵元的 **RX complex baseband** 之后，后端基本就是把它从“原始时域回波”一步步变成“目标参数/目标列表/轨迹/分类结果”。对 pulse-Doppler + 2D 相控阵来说，最典型的是把数据组织成一个 **4D data cube** 然后做三大类处理：**距离、速度、角度**（外加检测/跟踪/分类）。

下面按工程里最常见的一条链路讲（也给出你熟悉的张量形状）。

---

## 1) 数字化与数据立方体（Data Cube）

你现在的连续时间 (x_{\text{rx}}[p,q](t)) 进入 ADC 后，按 PRI 把时间切成

* slow-time：脉冲索引 (m=0,\dots,M-1)
* fast-time：脉内采样索引 (n=0,\dots,N-1)，采样间隔 (T_s=1/f_s)

形成
[
X[p,q,n,m];\triangleq;x_{\text{rx}}[p,q]\big(t=mT_r+nT_s\big)
]
这就是后端所有处理的“原料”。

> 实际系统里在进 FFT 前还会做：DC 去除、IQ 失衡校正、通道幅相标定、时间对齐、脉冲间相位稳相（coherent correction）等。

---

## 2) 距离处理：脉冲压缩 / 匹配滤波（Range Compression）

对每个阵元、每个脉冲，沿 fast-time 做匹配滤波：
[
Y[p,q,k,m] ;=; \sum_{n} X[p,q,n,m]; h^*[n-k]
]
其中 (h[n]) 是发射基带 (s[n]) 的匹配滤波器（连续时间就是 (h(t)=s^*(-t)) 的离散版）。

频域实现更常见：
[
Y[p,q,\cdot,m] = \text{IFFT}\Big(\text{FFT}(X[p,q,\cdot,m])\cdot \text{FFT}(h)^*\Big)
]

输出变成 **距离像（range profile）**：

* 输入：(X[p,q,n,m])
* 输出：(Y[p,q,k,m])，这里 (k) 是 range bin（可映射到距离 (R_k\approx \frac{c}{2},kT_s)）

---

## 3) 杂波/干扰抑制（可选但常用）

在做 Doppler 前常做一类“慢时间滤波”：

* **MTI/高通**（去静止杂波）：对 (m) 做差分或 IIR/ FIR 高通
* **脉冲间相位补偿**：稳定相干性（不然 Doppler FFT 会糊）
* **空域抑制干扰**：如空域零陷（nulling）、自适应波束形成等（如果干扰很强）

输出仍在 (Y[p,q,k,m]) 这个域里。

---

## 4) 速度处理：慢时间 FFT（Doppler Processing）

对每个阵元、每个距离 bin，沿 slow-time 做 (M) 点 FFT：
[
Z[p,q,k,\nu] ;=; \sum_{m=0}^{M-1} Y[p,q,k,m]; w[m];e^{-j2\pi \nu m/M}
]

* (\nu) 是 Doppler bin
* (w[m]) 是慢时间窗（抑制旁瓣）

此时每个阵元都有一个 **Range-Doppler 图**：

* (Z[p,q,k,\nu])

---

## 5) 角度处理：数字波束形成 DBF（2D Beamforming）

你是 2D 阵列，所以对每个 ((k,\nu)) 在空间维度 ((p,q)) 上做波束形成/角度 FFT/或高分辨 DOA。

### 5.1 常规（Bartlett/相位补偿求和）DBF

给定扫描角 ((\theta,\phi))，接收 steering：
[
a_{p,q}(\theta,\phi)=\exp!\Big(j\frac{2\pi}{\lambda}\mathbf u(\theta,\phi)^{\mathsf T}\mathbf r_{p,q}\Big)
]
波束形成输出：
[
S(\theta,\phi,k,\nu)=\sum_{p,q} w_{p,q}(\theta,\phi)^*; Z[p,q,k,\nu]
]
常见取 (w(\theta,\phi)\propto a(\theta,\phi))（加窗归一化）。

### 5.2 均匀阵的快捷实现：2D 空域 FFT

如果阵列是规则 UPA 且你用等间隔扫描，可以在 ((p,q)) 上做 2D FFT 得到角谱（本质上也是一类 DBF）。

输出就是你之前提到的 “**角-距-速**” 立方体：
[
S[\text{az},\text{el},k,\nu]
]

---

## 一句话总结后端主线

**Data cube** (X[p,q,n,m])
→（fast-time 匹配滤波）距离 (Y[p,q,k,m])
→（slow-time FFT）速度 (Z[p,q,k,\nu])
→（空域 DBF/DOA）角度 (S[\text{az},\text{el},k,\nu])

---

现在我们可以用 radar_sim\signal_model.py 来仿真 2D phased array 的 rx signal X[p,q,n,m] 了，然后在 folder basic_radar_signal_processing 里面写 matched filtering 的 process, doppler processing, 以及 DBF (分别 implement Bartlett DBF, 以及均匀阵的快捷实现)。

对每一个 basic signal processing, 在 tests/ 里面写测试代码：
- 先用 radar_sim\signal_model.py 来仿真出一个场景下的 rx signal X[p,q,n,m]，然后调用 matched filtering process, doppler process, DBF, 然后分别 visualize 每一步处理后的结果，以 interactive plot 的形式画出来。这样做是为了 check implementation 是否正确，是否能探测到目标的位置，速度，角度，这些信息。 


