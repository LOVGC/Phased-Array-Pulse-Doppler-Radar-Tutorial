from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from basic_radar_signal_processing.processing import (
    bartlett_beamform,
    build_steering_matrix,
    doppler_axis_hz,
    doppler_process,
    fft_beamform_2d,
    range_axis_m,
    range_compress,
    spatial_frequency_axes,
)
from radar_sim.signal_model import ArrayGeometry, NoiseConfig, RadarSimulator, Target, Waveform


def main() -> None:
    # --- 1) Simulation parameters (documenting each parameter) ---
    waveform = Waveform(
        pulse_width=20e-6,  # LFM pulse duration (s)
        bandwidth=5e6,  # LFM bandwidth (Hz) -> range resolution ~ c/(2B)
        carrier_frequency=10e9,  # carrier frequency (Hz)
        sample_rate=10e6,  # baseband sample rate (Hz)
        pri=200e-6,  # pulse repetition interval (s)
        num_pulses=16,  # pulses per CPI (slow-time length)
    )
    geometry = ArrayGeometry(
        num_x=8,  # number of elements along x
        num_y=8,  # number of elements along y
        dx=0.5 * waveform.wavelength,  # x spacing (m)
        dy=0.5 * waveform.wavelength,  # y spacing (m)
    )
    targets = [
        Target(
            range_m=6_000.0,  # target range (m)
            velocity_m_s=12.0,  # radial velocity (m/s), + means approaching
            azimuth_rad=np.deg2rad(12.0),  # azimuth angle (rad)
            elevation_rad=np.deg2rad(4.0),  # elevation angle (rad)
            amplitude=1.0 + 0.0j,  # complex reflectivity
        ),
        Target(
            range_m=12_000.0,
            velocity_m_s=-8.0,  # negative -> receding
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

    # --- 2) Simulate RX data cube X[p,q,fast_time,slow_time] ---
    simulator = RadarSimulator(geometry, waveform)
    rx, fast_time, slow_time = simulator.rx_baseband(
        targets,
        noise=NoiseConfig(std=0.01, seed=0),  # complex noise std and random seed
    )

    # --- 3) Range compression (matched filtering) ---
    # Compress each fast-time pulse to get range profiles Y[p,q,range,slow_time].
    range_cube = range_compress(rx, waveform)
    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)

    # --- 4) Doppler processing (slow-time FFT) ---
    # Transform along slow-time to get Z[p,q,range,doppler].
    doppler_cube = doppler_process(range_cube, window="hann")
    doppler_axis = doppler_axis_hz(waveform.num_pulses, waveform.pri)
    velocity_axis = doppler_axis * waveform.wavelength / 2.0

    # --- 5) Angle processing (DBF) ---
    # Bartlett beamforming on a selected range/Doppler bin.
    target = targets[0]
    range_idx = int(np.argmin(np.abs(range_axis - target.range_m)))
    doppler_power = np.sum(np.abs(doppler_cube[:, :, range_idx, :]) ** 2, axis=(0, 1))
    doppler_idx = int(np.argmax(doppler_power))
    snapshot = doppler_cube[:, :, range_idx, doppler_idx].reshape(-1)

    az_deg = np.arange(-60.0, 60.1, 2.0)
    el_deg = np.arange(0.0, 20.1, 2.0)
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    steering = build_steering_matrix(geometry, waveform.wavelength, az_rad, el_rad)
    bartlett = bartlett_beamform(snapshot, steering, el_rad.size, az_rad.size)
    bartlett_db = 20.0 * np.log10(np.abs(bartlett) / np.max(np.abs(bartlett)) + 1e-12)
    bartlett_peak_idx = np.unravel_index(np.argmax(np.abs(bartlett)), bartlett.shape)
    bartlett_peak_az = az_deg[bartlett_peak_idx[1]]
    bartlett_peak_el = el_deg[bartlett_peak_idx[0]]

    # 2D FFT beamforming (fast DBF for UPA), same range/Doppler bin.
    doppler_slice = doppler_cube[:, :, range_idx : range_idx + 1, doppler_idx : doppler_idx + 1]
    fft_spectrum = fft_beamform_2d(
        doppler_slice, window="hann", fft_size_x=64, fft_size_y=64
    )[:, :, 0, 0]
    u_x, u_y = spatial_frequency_axes(
        geometry.num_x,
        geometry.num_y,
        geometry.dx,
        geometry.dy,
        waveform.wavelength,
        fft_size_x=64,
        fft_size_y=64,
    )
    fft_db = 20.0 * np.log10(np.abs(fft_spectrum) / np.max(np.abs(fft_spectrum)) + 1e-12)
    fft_peak_idx = np.unravel_index(np.argmax(np.abs(fft_spectrum)), fft_spectrum.shape)
    fft_peak_u_x = u_x[fft_peak_idx[0]]
    fft_peak_u_y = u_y[fft_peak_idx[1]]
    cos_el = np.sqrt(fft_peak_u_x**2 + fft_peak_u_y**2)
    cos_el = np.clip(cos_el, 0.0, 1.0)
    fft_peak_el_deg = np.rad2deg(np.arccos(cos_el))
    fft_peak_az_deg = np.rad2deg(np.arctan2(fft_peak_u_y, fft_peak_u_x))

    # --- 6) Visualize each step ---
    fig_raw, ax_raw = plt.subplots(figsize=(9, 4))
    ax_raw.plot(fast_time * 1e6, np.real(rx[0, 0, :, 0]), label="I")
    ax_raw.plot(fast_time * 1e6, np.imag(rx[0, 0, :, 0]), label="Q", alpha=0.8)
    ax_raw.set_xlabel("Fast time (us)")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.set_title("Raw RX (element 0,0), pulse 0")
    ax_raw.legend()

    fig_range, ax_range = plt.subplots(figsize=(9, 4))
    ax_range.plot(range_axis / 1e3, np.abs(range_cube[0, 0, :, 0]))
    ax_range.set_xlabel("Range (km)")
    ax_range.set_ylabel("Magnitude")
    ax_range.set_title("Range-compressed profile (element 0,0), pulse 0")

    rd_power = np.sum(np.abs(doppler_cube) ** 2, axis=(0, 1))
    rd_db = 10.0 * np.log10(rd_power + 1e-12)
    fig_rd, ax_rd = plt.subplots(figsize=(9, 6))
    extent_rd = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0] / 1e3,
        range_axis[-1] / 1e3,
    ]
    im_rd = ax_rd.imshow(
        rd_db,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="viridis",
    )
    ax_rd.set_xlabel("Velocity (m/s)")
    ax_rd.set_ylabel("Range (km)")
    ax_rd.set_title("Range-Doppler Map")
    fig_rd.colorbar(im_rd, ax=ax_rd, label="Power (dB)")

    fig_bartlett, ax_bartlett = plt.subplots(figsize=(9, 6))
    extent_bartlett = [az_deg[0], az_deg[-1], el_deg[0], el_deg[-1]]
    im_bartlett = ax_bartlett.imshow(
        bartlett_db,
        origin="lower",
        aspect="auto",
        extent=extent_bartlett,
        cmap="magma",
    )
    ax_bartlett.scatter(
        bartlett_peak_az,
        bartlett_peak_el,
        marker="x",
        color="white",
        s=60,
    )
    ax_bartlett.text(
        bartlett_peak_az,
        bartlett_peak_el,
        f" peak ({bartlett_peak_az:.1f} deg, {bartlett_peak_el:.1f} deg)",
        color="white",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax_bartlett.set_xlabel("Azimuth (deg)")
    ax_bartlett.set_ylabel("Elevation (deg)")
    ax_bartlett.set_title("Bartlett DBF (range/Doppler of target 1)")
    fig_bartlett.colorbar(im_bartlett, ax=ax_bartlett, label="Response (dB)")

    fig_fft, ax_fft = plt.subplots(figsize=(7, 6))
    extent_fft = [u_x[0], u_x[-1], u_y[0], u_y[-1]]
    im_fft = ax_fft.imshow(
        fft_db.T,
        origin="lower",
        aspect="auto",
        extent=extent_fft,
        cmap="viridis",
    )
    ax_fft.scatter(fft_peak_u_x, fft_peak_u_y, marker="x", color="white", s=60)
    ax_fft.text(
        fft_peak_u_x,
        fft_peak_u_y,
        f" peak az={fft_peak_az_deg:.1f} deg, el={fft_peak_el_deg:.1f} deg",
        color="white",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax_fft.set_xlabel("Direction cosine u_x")
    ax_fft.set_ylabel("Direction cosine u_y")
    ax_fft.set_title("2D FFT Beamforming (range/Doppler of target 1)")
    fig_fft.colorbar(im_fft, ax=ax_fft, label="Response (dB)")

    plt.show()


if __name__ == "__main__":
    main()
