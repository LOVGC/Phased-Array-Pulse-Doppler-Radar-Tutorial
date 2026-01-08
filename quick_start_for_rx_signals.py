from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from radar_sim.signal_model import ArrayGeometry, NoiseConfig, RadarSimulator, Target, Waveform


def main() -> None:
    # 1) Define waveform and array geometry.
    waveform = Waveform(
        pulse_width=20e-6,  # LFM pulse duration (s)
        bandwidth=5e6,  # LFM bandwidth (Hz)
        carrier_frequency=10e9,  # RF carrier (Hz)
        sample_rate=20e6,  # baseband sample rate (Hz)
        pri=200e-6,  # pulse repetition interval (s)
        num_pulses=8,  # pulses per CPI
    )
    geometry = ArrayGeometry(
        num_x=4,  # elements along x
        num_y=4,  # elements along y
        dx=0.5 * waveform.wavelength,  # x spacing (m)
        dy=0.5 * waveform.wavelength,  # y spacing (m)
    )

    # 2) Define a few targets (range in meters, velocity in m/s).
    targets = [
        Target(
            range_m=6_000.0,  # initial range (m)
            velocity_m_s=15.0,  # radial velocity (m/s), + means approaching
            azimuth_rad=np.deg2rad(10.0),  # azimuth (rad)
            elevation_rad=np.deg2rad(4.0),  # elevation (rad)
            amplitude=1.0 + 0.0j,  # complex reflectivity
        ),
        Target(
            range_m=9_000.0,  # initial range (m)
            velocity_m_s=-8.0,  # radial velocity (m/s), - means receding
            azimuth_rad=np.deg2rad(-20.0),  # azimuth (rad)
            elevation_rad=np.deg2rad(2.0),  # elevation (rad)
            amplitude=0.8 + 0.1j,  # complex reflectivity
        ),
    ]

    # 3) Run the simulator (rx is shaped as [x, y, fast_time, slow_time]).
    simulator = RadarSimulator(geometry, waveform)
    rx, fast_time, slow_time = simulator.rx_baseband(
        targets,
        noise=NoiseConfig(std=0.01, seed=0),  # noise std (linear), seed for repeatability
    )

    # 4) Inspect a single element.
    element_signal = rx[0, 0]  # shape: [fast_time, slow_time]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # One PRI time-series (I/Q) for the first pulse.
    axes[0].plot(fast_time * 1e6, np.real(element_signal[:, 0]), label="I")
    axes[0].plot(fast_time * 1e6, np.imag(element_signal[:, 0]), label="Q", alpha=0.8)
    axes[0].set_xlabel("Fast time (us)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Element (0, 0) baseband, pulse 0")
    axes[0].legend()

    # Magnitude over the CPI (slow time vs fast time).
    im = axes[1].imshow(
        np.abs(element_signal).T,
        aspect="auto",
        origin="lower",
        extent=[
            fast_time[0] * 1e6,
            fast_time[-1] * 1e6,
            slow_time[0] * 1e3,
            slow_time[-1] * 1e3,
        ],
        cmap="viridis",
    )
    axes[1].set_xlabel("Fast time (us)")
    axes[1].set_ylabel("Slow time (ms)")
    axes[1].set_title("Element (0, 0) magnitude over CPI")
    fig.colorbar(im, ax=axes[1], label="Magnitude")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
