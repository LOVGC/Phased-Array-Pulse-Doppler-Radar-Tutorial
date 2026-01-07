from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import constants


def unit_direction(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
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
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.num_x <= 0 or self.num_y <= 0:
            raise ValueError("Array dimensions must be positive.")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("Element spacing must be positive.")

    def element_positions(self) -> np.ndarray:
        x = np.arange(self.num_x) * self.dx
        y = np.arange(self.num_y) * self.dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.stack([xx, yy, np.zeros_like(xx)], axis=-1)


@dataclass
class Waveform:
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
    unambiguous_range_m: float
    max_unambiguous_velocity_m_s: float
    range_resolution_m: float
    velocity_resolution_m_s: float
    prf_hz: float
    cpi_duration_s: float
    doppler_resolution_hz: float


class RadarSimulator:
    def __init__(self, geometry: ArrayGeometry, waveform: Waveform) -> None:
        self.geometry = geometry
        self.waveform = waveform
        self._positions = geometry.element_positions()
        self.radar_config = RadarConfig(
            unambiguous_range_m=waveform.unambiguous_range_m,
            max_unambiguous_velocity_m_s=waveform.max_unambiguous_velocity_m_s,
            range_resolution_m=waveform.range_resolution_m,
            velocity_resolution_m_s=waveform.velocity_resolution_m_s,
            prf_hz=waveform.prf,
            cpi_duration_s=waveform.cpi_duration_s,
            doppler_resolution_hz=waveform.doppler_resolution_hz,
        )

    def radar_config_summary(self) -> str:
        config = self.radar_config
        return (
            "Radar performance summary:\n"
            f"  unambiguous_range_m       : {config.unambiguous_range_m:.6g}\n"
            f"  max_unambig_velocity_m_s  : {config.max_unambiguous_velocity_m_s:.6g}\n"
            f"  range_resolution_m        : {config.range_resolution_m:.6g}\n"
            f"  velocity_resolution_m_s   : {config.velocity_resolution_m_s:.6g}\n"
            f"  prf_hz                    : {config.prf_hz:.6g}\n"
            f"  cpi_duration_s            : {config.cpi_duration_s:.6g}\n"
            f"  doppler_resolution_hz     : {config.doppler_resolution_hz:.6g}"
        )

    def fast_time_axis(self) -> np.ndarray:
        num_fast = self.waveform.fast_time_samples
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
        time_grid = slow_time[:, None] + fast_time[None, :]

        rx = np.zeros(
            (
                self.geometry.num_x,
                self.geometry.num_y,
                fast_time.size,
                slow_time.size,
            ),
            dtype=np.complex128,
        )

        for target in targets:
            direction = target.unit_vector()
            geom_delay = (
                -np.tensordot(self._positions, direction, axes=([2], [0]))
                / constants.c
            )
            range_time = target.range_m - target.velocity_m_s * time_grid # v>0 目标靠近雷达
            tau_range = 2.0 * range_time / constants.c

            for p in range(self.geometry.num_x):
                for q in range(self.geometry.num_y):
                    tau = tau_range + geom_delay[p, q]
                    t_emit = time_grid - tau
                    tx = tx_baseband(t_emit, self.waveform)
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

        return rx, fast_time, slow_time
