from __future__ import annotations

import numpy as np

from radar_sim.signal_model import ArrayGeometry, NoiseConfig, RadarSimulator, Target, Waveform


def build_demo_scene() -> tuple[Waveform, ArrayGeometry, list[Target]]:
    '''builds a shared demo scene and RX simulation.'''
    waveform = Waveform(
        pulse_width=20e-6,
        bandwidth=5e6,
        carrier_frequency=10e9,
        sample_rate=10e6,
        pri=200e-6,
        num_pulses=16,
    )
    geometry = ArrayGeometry(
        num_x=8,
        num_y=8,
        dx=0.5 * waveform.wavelength,
        dy=0.5 * waveform.wavelength,
    )
    targets = [
        Target(
            range_m=6_000.0,
            velocity_m_s=12.0,
            azimuth_rad=np.deg2rad(12.0),
            elevation_rad=np.deg2rad(4.0),
            amplitude=1.0 + 0.0j,
        ),
        Target(
            range_m=12_000.0,
            velocity_m_s=-8.0,
            azimuth_rad=np.deg2rad(-18.0),
            elevation_rad=np.deg2rad(2.0),
            amplitude=0.8 + 0.1j,
        ),
        Target(
            range_m=18_000.0,
            velocity_m_s=18.0,
            azimuth_rad=np.deg2rad(25.0),
            elevation_rad=np.deg2rad(-3.0),
            amplitude=0.7 - 0.05j,
        ),
    ]
    return waveform, geometry, targets


def simulate_rx(
    waveform: Waveform, geometry: ArrayGeometry, targets: list[Target]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulator = RadarSimulator(geometry, waveform)
    rx, fast_time, slow_time = simulator.rx_baseband(
        targets, noise=NoiseConfig(std=0.01, seed=0)
    )
    return rx, fast_time, slow_time
