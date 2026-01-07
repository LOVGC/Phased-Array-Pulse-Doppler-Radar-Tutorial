from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from radar_sim.signal_model import (
    ArrayGeometry,
    NoiseConfig,
    RadarSimulator,
    Target,
    Waveform,
    lfm_pulse,
)


def next_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return int(2 ** np.ceil(np.log2(value)))


def matched_filter(rx: np.ndarray, waveform: Waveform) -> np.ndarray:
    num_fast = rx.shape[2]
    num_ref = int(np.round(waveform.pulse_width * waveform.sample_rate))
    if num_ref < 1:
        raise ValueError("Pulse width too small for the current sample rate.")

    t_ref = np.arange(num_ref) / waveform.sample_rate
    reference = lfm_pulse(t_ref, waveform.pulse_width, waveform.bandwidth)
    filter_taps = np.conj(reference[::-1])

    nfft = next_pow2(num_fast + num_ref - 1)
    filter_f = np.fft.fft(filter_taps, n=nfft)
    rx_f = np.fft.fft(rx, n=nfft, axis=2)
    compressed_full = np.fft.ifft(rx_f * filter_f[None, None, :, None], axis=2)

    start = num_ref - 1
    return compressed_full[:, :, start : start + num_fast, :]


def doppler_processing(compressed: np.ndarray, num_pulses: int) -> np.ndarray:
    window = np.hanning(num_pulses)
    windowed = compressed * window[None, None, None, :]
    return np.fft.fftshift(np.fft.fft(windowed, axis=3), axes=3)


def build_steering_matrix(
    positions: np.ndarray, wavelength: float, az_rad: np.ndarray, el_rad: np.ndarray
) -> np.ndarray:
    az = az_rad[None, :]
    el = el_rad[:, None]
    cos_el = np.cos(el)
    dir_x = cos_el * np.cos(az)
    dir_y = cos_el * np.sin(az)
    dir_z = np.broadcast_to(np.sin(el), dir_x.shape)
    directions = np.stack([dir_x, dir_y, dir_z], axis=-1)

    directions_flat = directions.reshape(-1, 3)
    positions_flat = positions.reshape(-1, 3)
    phase = (2.0 * np.pi / wavelength) * (directions_flat @ positions_flat.T)
    return np.exp(1j * phase)


def compute_ra_map(
    doppler_cube: np.ndarray,
    range_axis_m: np.ndarray,
    positions: np.ndarray,
    wavelength: float,
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    max_range_m: float,
    range_decimation: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_range_idx = int(np.searchsorted(range_axis_m, max_range_m))
    range_indices = np.arange(0, max_range_idx, range_decimation)
    steering = build_steering_matrix(positions, wavelength, az_rad, el_rad)

    ra_power = np.zeros((range_indices.size, az_rad.size))
    for out_idx, range_idx in enumerate(range_indices):
        doppler_slice = doppler_cube[:, :, range_idx, :]
        doppler_power = np.sum(np.abs(doppler_slice) ** 2, axis=(0, 1))
        best_bin = int(np.argmax(doppler_power))
        snapshot = doppler_slice[:, :, best_bin].reshape(-1)

        beam = steering.conj() @ snapshot
        beam_power = np.abs(beam) ** 2
        beam_power = beam_power.reshape(el_rad.size, az_rad.size)
        ra_power[out_idx, :] = beam_power.max(axis=0)

    return range_axis_m[range_indices], ra_power


def validate_targets(
    targets: list[Target], unambiguous_range_m: float, max_velocity_m_s: float
) -> None:
    for idx, target in enumerate(targets, start=1):
        if not (0.0 < target.range_m < unambiguous_range_m):
            raise ValueError(f"Target {idx} range is outside the unambiguous range.")
        if abs(target.velocity_m_s) >= max_velocity_m_s:
            raise ValueError(f"Target {idx} velocity is outside the unambiguous range.")


def check_resolution_spacing(
    values: list[float], resolution: float, label: str
) -> None:
    values_sorted = np.sort(values)
    if values_sorted.size < 2:
        return
    min_spacing = np.min(np.diff(values_sorted))
    if min_spacing < resolution:
        print(
            f"Warning: {label} spacing {min_spacing:.3g} is below resolution "
            f"{resolution:.3g}."
        )


def main() -> None:
    waveform = Waveform(
        pulse_width=20e-6,
        bandwidth=5e6,
        carrier_frequency=10e9,
        sample_rate=10e6,
        pri=200e-6,
        num_pulses=32,
    )
    geometry = ArrayGeometry(
        num_x=10,
        num_y=10,
        dx=0.5 * waveform.wavelength,
        dy=0.5 * waveform.wavelength,
    )

    simulator = RadarSimulator(geometry, waveform)
    config = simulator.radar_config

    targets = [
        Target(
            range_m=6_000.0,
            velocity_m_s=10.0,
            azimuth_rad=np.deg2rad(12.0),
            elevation_rad=np.deg2rad(6.0),
            amplitude=1.0 + 0.0j,
        ),
        Target(
            range_m=12_000.0,
            velocity_m_s=-15.0,
            azimuth_rad=np.deg2rad(-18.0),
            elevation_rad=np.deg2rad(3.0),
            amplitude=0.8 + 0.1j,
        ),
        Target(
            range_m=18_000.0,
            velocity_m_s=22.0,
            azimuth_rad=np.deg2rad(28.0),
            elevation_rad=np.deg2rad(-4.0),
            amplitude=0.7 - 0.05j,
        ),
    ]

    validate_targets(
        targets, config.unambiguous_range_m, config.max_unambiguous_velocity_m_s
    )
    check_resolution_spacing(
        [target.range_m for target in targets],
        config.range_resolution_m,
        "range",
    )
    check_resolution_spacing(
        [target.velocity_m_s for target in targets],
        config.velocity_resolution_m_s,
        "velocity",
    )

    rx, fast_time, _ = simulator.rx_baseband(
        targets, noise=NoiseConfig(std=0.01, seed=1)
    )

    compressed = matched_filter(rx, waveform)
    doppler_cube = doppler_processing(compressed, waveform.num_pulses)

    range_axis_m = constants.c * fast_time / 2.0
    doppler_hz = np.fft.fftshift(
        np.fft.fftfreq(waveform.num_pulses, d=waveform.pri)
    )
    velocity_axis = doppler_hz * waveform.wavelength / 2.0

    rd_map = np.sum(doppler_cube, axis=(0, 1))
    rd_db = 20.0 * np.log10(np.abs(rd_map) + 1e-12)

    az_grid_deg = np.arange(-60.0, 60.1, 2.0)
    el_grid_deg = np.arange(-20.0, 20.1, 2.0)
    az_grid_rad = np.deg2rad(az_grid_deg)
    el_grid_rad = np.deg2rad(el_grid_deg)
    range_axis_sel_m, ra_power = compute_ra_map(
        doppler_cube,
        range_axis_m,
        geometry.element_positions(),
        waveform.wavelength,
        az_grid_rad,
        el_grid_rad,
        max_range_m=min(config.unambiguous_range_m, 20_000.0),
        range_decimation=2,
    )
    ra_db = 10.0 * np.log10(ra_power / np.max(ra_power) + 1e-12)

    fig_rd, ax_rd = plt.subplots(figsize=(9, 6))
    extent_rd = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis_m[0] / 1e3,
        range_axis_m[-1] / 1e3,
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
    fig_rd.colorbar(im_rd, ax=ax_rd, label="Magnitude (dB)")

    fig_ra, ax_ra = plt.subplots(figsize=(9, 6))
    extent_ra = [
        az_grid_deg[0],
        az_grid_deg[-1],
        range_axis_sel_m[0] / 1e3,
        range_axis_sel_m[-1] / 1e3,
    ]
    im_ra = ax_ra.imshow(
        ra_db,
        aspect="auto",
        origin="lower",
        extent=extent_ra,
        cmap="magma",
    )
    ax_ra.set_xlabel("Azimuth (deg)")
    ax_ra.set_ylabel("Range (km)")
    ax_ra.set_title("Range-Azimuth Map (max over elevation)")
    fig_ra.colorbar(im_ra, ax=ax_ra, label="Power (dB)")

    plt.show()


if __name__ == "__main__":
    main()
