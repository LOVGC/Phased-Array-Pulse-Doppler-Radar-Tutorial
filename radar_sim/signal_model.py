from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import constants


def unit_direction(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    '''Compute a unit direction vector (直角坐标) from azimuth and elevation angles.'''
    cos_el = np.cos(elevation_rad)
    return np.array(
        [
            cos_el * np.cos(azimuth_rad),
            cos_el * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )


def rect_pulse(t: np.ndarray, width: float) -> np.ndarray:
    t = np.asarray(t)
    return np.where((t >= 0.0) & (t < width), 1.0, 0.0)


def lfm_pulse(t: np.ndarray, pulse_width: float, bandwidth: float) -> np.ndarray:
    t = np.asarray(t)
    chirp_rate = bandwidth / pulse_width
    phase = np.pi * chirp_rate * t**2
    return rect_pulse(t, pulse_width) * np.exp(1j * phase)


def tx_baseband(t: np.ndarray, waveform: "Waveform") -> np.ndarray:
    '''这里可以生成一个 pri 的信号，也可以生成多个 pri 的信号。这个方法用来 implementation time delay很棒啊'''
    t = np.asarray(t)
    output = np.zeros_like(t, dtype=np.complex128)
    for pulse_index in range(waveform.num_pulses):
        t_shift = t - pulse_index * waveform.pri
        output += lfm_pulse(t_shift, waveform.pulse_width, waveform.bandwidth)
    return output


@dataclass
class ArrayGeometry:
    num_x: int
    num_y: int
    dx: float # element spacing in x direction
    dy: float # element spacing in y direction

    def __post_init__(self) -> None:
        if self.num_x <= 0 or self.num_y <= 0:
            raise ValueError("Array dimensions must be positive.")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("Element spacing must be positive.")

    def element_positions(self) -> np.ndarray:
        x = np.arange(self.num_x) * self.dx
        y = np.arange(self.num_y) * self.dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        # 反正核心概念是这个 tensor 存的就是每个 element 的直角坐标, 至于怎么去 access 这些坐标，你试试就知道了
        return np.stack([xx, yy, np.zeros_like(xx)], axis=-1) # shape (num_x, num_y, 3), 这个 tensor 其实就是存的所有 element 的直角系坐标



@dataclass
class Target:
    range_m: float
    velocity_m_s: float
    azimuth_rad: float
    elevation_rad: float
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        if self.range_m <= 0.0:
            raise ValueError("Target range must be positive.")

    def unit_vector(self) -> np.ndarray:
        '''return 的是目标的单位方向向量(直角坐标)'''
        return unit_direction(self.azimuth_rad, self.elevation_rad)


@dataclass
class NoiseConfig:
    std: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.std < 0.0:
            raise ValueError("Noise std must be non-negative.")


@dataclass(frozen=True)
class RadarConfig:
    '''雷达规格参数'''
    unambiguous_range_m: float
    max_unambiguous_velocity_m_s: float
    range_resolution_m: float
    velocity_resolution_m_s: float
    azimuth_resolution_rad: float
    elevation_resolution_rad: float
    prf_hz: float
    cpi_duration_s: float
    doppler_resolution_hz: float

@dataclass
class Waveform:
    '''雷达波形参数'''
    pulse_width: float
    bandwidth: float
    carrier_frequency: float
    sample_rate: float
    pri: float
    num_pulses: int

    def __post_init__(self) -> None:
        if self.pulse_width <= 0.0:
            raise ValueError("Pulse width must be positive.")
        if self.bandwidth <= 0.0:
            raise ValueError("Bandwidth must be positive.")
        if self.carrier_frequency <= 0.0:
            raise ValueError("Carrier frequency must be positive.")
        if self.sample_rate <= 0.0:
            raise ValueError("Sample rate must be positive.")
        if self.pri <= 0.0:
            raise ValueError("PRI must be positive.")
        if self.num_pulses <= 0:
            raise ValueError("Number of pulses must be positive.")

    @property
    def chirp_rate(self) -> float:
        return self.bandwidth / self.pulse_width

    @property
    def wavelength(self) -> float:
        return constants.c / self.carrier_frequency

    @property
    def fast_time_samples(self) -> int: 
        '''这里的 fast_time samples 指的是一个 pri 内的采样点数'''
        return int(np.round(self.pri * self.sample_rate))

    @property
    def prf(self) -> float:
        return 1.0 / self.pri

    @property
    def unambiguous_range_m(self) -> float:
        return constants.c * self.pri / 2.0

    @property
    def range_resolution_m(self) -> float:
        return constants.c / (2.0 * self.bandwidth)

    @property
    def max_unambiguous_velocity_m_s(self) -> float:
        return self.wavelength / (4.0 * self.pri)

    @property
    def cpi_duration_s(self) -> float:
        return self.num_pulses * self.pri

    @property
    def doppler_resolution_hz(self) -> float:
        return 1.0 / self.cpi_duration_s

    @property
    def velocity_resolution_m_s(self) -> float:
        return self.wavelength / (2.0 * self.cpi_duration_s)

class RadarSimulator:
    def __init__(self, geometry: ArrayGeometry, waveform: Waveform) -> None:
        self.geometry = geometry
        self.waveform = waveform
        self._positions = geometry.element_positions()
        az_res_rad, el_res_rad = self._angle_resolution_rad()
        self.radar_config = RadarConfig(
            unambiguous_range_m=waveform.unambiguous_range_m,
            max_unambiguous_velocity_m_s=waveform.max_unambiguous_velocity_m_s,
            range_resolution_m=waveform.range_resolution_m,
            velocity_resolution_m_s=waveform.velocity_resolution_m_s,
            azimuth_resolution_rad=az_res_rad,
            elevation_resolution_rad=el_res_rad,
            prf_hz=waveform.prf,
            cpi_duration_s=waveform.cpi_duration_s,
            doppler_resolution_hz=waveform.doppler_resolution_hz,
        )

    def _angle_resolution_rad(self) -> tuple[float, float]:
        # Approximate 3 dB beamwidth for a uniformly weighted UPA (broadside).
        aperture_x = (self.geometry.num_x - 1) * self.geometry.dx
        aperture_y = (self.geometry.num_y - 1) * self.geometry.dy
        wavelength = self.waveform.wavelength
        az_res = np.inf if aperture_x <= 0.0 else 0.886 * wavelength / aperture_x
        el_res = np.inf if aperture_y <= 0.0 else 0.886 * wavelength / aperture_y
        return az_res, el_res

    def radar_config_summary(self) -> str:
        config = self.radar_config
        return (
            "Radar performance summary:\n"
            f"  unambiguous_range_m       : {config.unambiguous_range_m:.6g}\n"
            f"  max_unambig_velocity_m_s  : {config.max_unambiguous_velocity_m_s:.6g}\n"
            f"  range_resolution_m        : {config.range_resolution_m:.6g}\n"
            f"  velocity_resolution_m_s   : {config.velocity_resolution_m_s:.6g}\n"
            f"  azimuth_resolution_deg    : {np.rad2deg(config.azimuth_resolution_rad):.6g}\n"
            f"  elevation_resolution_deg  : {np.rad2deg(config.elevation_resolution_rad):.6g}\n"
            f"  prf_hz                    : {config.prf_hz:.6g}\n"
            f"  cpi_duration_s            : {config.cpi_duration_s:.6g}\n"
            f"  doppler_resolution_hz     : {config.doppler_resolution_hz:.6g}"
        )

    def fast_time_axis(self) -> np.ndarray:
        num_fast = self.waveform.fast_time_samples # 指的是一个 pri 内的采样点数
        return np.arange(num_fast) / self.waveform.sample_rate

    def slow_time_axis(self) -> np.ndarray:
        return np.arange(self.waveform.num_pulses) * self.waveform.pri

    def tx_baseband(self, t: np.ndarray) -> np.ndarray:
        return tx_baseband(t, self.waveform)

    def rx_baseband(
        self,
        targets: Sequence[Target],
        noise: NoiseConfig | None = None,
        print_radar_config: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if print_radar_config:
            print(self.radar_config_summary())
        fast_time = self.fast_time_axis()
        slow_time = self.slow_time_axis()
        # time_grid 可以理解成雷达观测的时间窗口，时间本来是一维的就行了，但是为了方便计算，我们把它变成二维的而已。
        # 这在概念上并没有什么新的东西。你在概念上，把这个 time_grid 拉直，看成一个大的时间窗也是可以的。这样理解更 natural 一些。
        time_grid = slow_time[:, None] + fast_time[None, :] # time_grid 中每一个 cell 存的其实就是这个 cell 对应的采样时间，而且是真实时间，不是 Index

        rx = np.zeros(
            (
                self.geometry.num_x,
                self.geometry.num_y,
                fast_time.size,
                slow_time.size,
            ),
            dtype=np.complex128,
        )  # rx[p, q, :, :] 对应的是第 (p, q) 个 element 接收到的信号矩阵，矩阵的行对应 fast time，列对应 slow time

        for target in targets:
            direction = target.unit_vector()
            geom_delay = (
                -np.tensordot(self._positions, direction, axes=([2], [0]))
                / constants.c
            ) # geom_delay shape (num_x, num_y), 存的就是每个 antenna element 相对于雷达相位中心的几何时延.
            # range_time 就是 R(t),i.e. 目标相对于雷达的实际距离随时间的变化, 还有一点就是，R(t) 在概念上是函数，但是在计算机里就是用数组来表示。
            range_time = target.range_m - target.velocity_m_s * time_grid # v>0 目标靠近雷达
            
            # 这里的这个 tau_range 其实也包含了 doppler effect, 因为 range_time 里面有速度。
            tau_range = 2.0 * range_time / constants.c # 这个就是 two-way travel time delay 了

            for p in range(self.geometry.num_x):
                for q in range(self.geometry.num_y):
                    tau = tau_range + geom_delay[p, q] # 信号测量到的 total delay = two-way range delay + geometry delay
                    t_emit = time_grid - tau
                    tx = tx_baseband(t_emit, self.waveform) # 这个信号就是 x_tx(t-tau),i.e. baseband 经历的 time delay。这里的 baseband 指的是整个 pulse train
                    phase = np.exp(
                        -1j * 2.0 * np.pi * self.waveform.carrier_frequency * tau
                    )
                    rx[p, q, :, :] += (target.amplitude * tx * phase).T

        if noise and noise.std > 0.0:
            rng = np.random.default_rng(noise.seed)
            sigma = noise.std / np.sqrt(2.0)
            noise_samples = sigma * (
                rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape)
            )
            rx += noise_samples
        # 这里的 fast_time: 0, Ts, 2Ts, ..., (N-1)Ts
        # slow_time: 0, PRI, 2PRI, ..., (M-1)PRI
        # fast_time 和 slow_time 一起定义了 rx 里面的每一个采样点对应的实际时间
        return rx, fast_time, slow_time
