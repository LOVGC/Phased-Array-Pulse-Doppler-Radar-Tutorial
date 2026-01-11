from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import (
    build_steering_matrix,
    ca_cfar_2d,
    doppler_axis_hz,
    doppler_process,
    music_spectrum_from_snapshots,
    range_axis_m,
    range_compress,
)
from radar_sim.signal_model import ArrayGeometry, NoiseConfig, RadarSimulator, Target, Waveform


def simulate_scene(
    waveform: Waveform,
    geometry: ArrayGeometry,
    targets: list[Target],
    noise_std: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulator = RadarSimulator(geometry, waveform)
    rx, fast_time, slow_time = simulator.rx_baseband(
        targets,
        noise=NoiseConfig(std=noise_std, seed=seed),
        print_radar_config=False,
    )
    return rx, fast_time, slow_time


def compute_rd(
    range_cube: np.ndarray, waveform: Waveform
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    doppler_cube = doppler_process(range_cube, window="hann")
    rd_power = np.sum(np.abs(doppler_cube) ** 2, axis=(0, 1))
    rd_db = 10.0 * np.log10(rd_power + 1e-12)
    return doppler_cube, rd_power, rd_db


def compute_doa(
    range_cube: np.ndarray,
    range_axis: np.ndarray,
    geometry: ArrayGeometry,
    waveform: Waveform,
    targets: list[Target],
    az_deg: np.ndarray,
    el_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    steering = build_steering_matrix(geometry, waveform.wavelength, az_rad, el_rad)
    doa_maps = []
    for target in targets:
        range_idx = int(np.argmin(np.abs(range_axis - target.range_m)))
        snapshots = range_cube[:, :, range_idx, :].reshape(
            geometry.num_x * geometry.num_y, -1
        )
        doa_map = music_spectrum_from_snapshots(
            snapshots,
            steering,
            num_sources=1,
            diagonal_loading=1e-3,
        ).reshape(el_deg.size, az_deg.size)
        doa_maps.append(doa_map)
    doa_power = np.maximum.reduce(doa_maps)
    doa_db = 10.0 * np.log10(doa_power / np.max(doa_power) + 1e-12)
    return doa_power, doa_db, steering


def compute_ra(
    doppler_cube: np.ndarray,
    range_axis: np.ndarray,
    steering: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    range_decimation: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    doppler_cube_dec = doppler_cube[:, :, ::range_decimation, :]
    range_axis_dec = range_axis[::range_decimation]
    num_elements = doppler_cube_dec.shape[0] * doppler_cube_dec.shape[1]
    num_range = doppler_cube_dec.shape[2]
    num_doppler = doppler_cube_dec.shape[3]
    snapshots = doppler_cube_dec.reshape(num_elements, -1)
    responses = steering.conj() @ snapshots
    responses = responses.reshape(el_deg.size, az_deg.size, num_range, num_doppler)
    power = np.abs(responses) ** 2
    rad_cube = np.max(power, axis=0)
    rad_cube = np.transpose(rad_cube, (1, 0, 2))
    ra_power = np.max(rad_cube, axis=2)
    ra_db = 10.0 * np.log10(ra_power / np.max(ra_power) + 1e-12)
    return rad_cube, ra_power, ra_db, range_axis_dec


def build_scenes() -> list[tuple[str, Waveform, ArrayGeometry, list[Target]]]:
    scenes: list[tuple[str, Waveform, ArrayGeometry, list[Target]]] = []

    waveform_a = Waveform(
        pulse_width=20e-6,
        bandwidth=5e6,
        carrier_frequency=10e9,
        sample_rate=10e6,
        pri=200e-6,
        num_pulses=16,
    )
    geometry_a = ArrayGeometry(
        num_x=8,
        num_y=8,
        dx=0.5 * waveform_a.wavelength,
        dy=0.5 * waveform_a.wavelength,
    )
    targets_a = [
        Target(6_000.0, 12.0, np.deg2rad(12.0), np.deg2rad(4.0), 1.0 + 0.0j),
        Target(12_000.0, -8.0, np.deg2rad(-18.0), np.deg2rad(2.0), 0.8 + 0.1j),
        Target(18_000.0, 18.0, np.deg2rad(25.0), np.deg2rad(-3.0), 0.7 - 0.05j),
    ]
    scenes.append(("Scene A (8x8, 16 pulses)", waveform_a, geometry_a, targets_a))

    waveform_b = Waveform(
        pulse_width=15e-6,
        bandwidth=8e6,
        carrier_frequency=9e9,
        sample_rate=16e6,
        pri=150e-6,
        num_pulses=32,
    )
    geometry_b = ArrayGeometry(
        num_x=10,
        num_y=6,
        dx=0.5 * waveform_b.wavelength,
        dy=0.5 * waveform_b.wavelength,
    )
    targets_b = [
        Target(4_500.0, 8.0, np.deg2rad(15.0), np.deg2rad(3.0), 1.0 + 0.0j),
        Target(9_000.0, -12.0, np.deg2rad(-25.0), np.deg2rad(6.0), 0.9 - 0.1j),
        Target(16_000.0, 20.0, np.deg2rad(30.0), np.deg2rad(-4.0), 0.7 + 0.05j),
    ]
    scenes.append(("Scene B (10x6, 32 pulses)", waveform_b, geometry_b, targets_b))

    waveform_c = Waveform(
        pulse_width=25e-6,
        bandwidth=4e6,
        carrier_frequency=10e9,
        sample_rate=8e6,
        pri=220e-6,
        num_pulses=24,
    )
    geometry_c = ArrayGeometry(
        num_x=6,
        num_y=6,
        dx=0.5 * waveform_c.wavelength,
        dy=0.5 * waveform_c.wavelength,
    )
    targets_c = [
        Target(7_000.0, -5.0, np.deg2rad(-10.0), np.deg2rad(1.0), 1.0 + 0.0j),
        Target(14_000.0, 12.0, np.deg2rad(22.0), np.deg2rad(-6.0), 0.85 + 0.05j),
    ]
    scenes.append(("Scene C (6x6, 24 pulses)", waveform_c, geometry_c, targets_c))

    return scenes


def plot_scene(
    title: str,
    waveform: Waveform,
    geometry: ArrayGeometry,
    targets: list[Target],
) -> None:
    az_deg = np.arange(-60.0, 60.1, 1.0)
    el_deg = np.arange(-10.0, 20.1, 1.0)

    rx, fast_time, _ = simulate_scene(waveform, geometry, targets)
    range_cube = range_compress(rx, waveform)
    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)
    doppler_cube, rd_power, rd_db = compute_rd(range_cube, waveform)
    doppler_axis = doppler_axis_hz(waveform.num_pulses, waveform.pri)
    velocity_axis = doppler_axis * waveform.wavelength / 2.0

    doa_power, doa_db, steering = compute_doa(
        range_cube, range_axis, geometry, waveform, targets, az_deg, el_deg
    )
    _, ra_power, ra_db, range_axis_dec = compute_ra(
        doppler_cube, range_axis, steering, az_deg, el_deg
    )

    _, rd_detections = ca_cfar_2d(
        rd_power, guard_cells=(1, 0), training_cells=(6, 1), pfa=1e-7
    )
    _, doa_detections = ca_cfar_2d(
        doa_power, guard_cells=(1, 1), training_cells=(4, 4), pfa=1e-4
    )
    _, ra_detections = ca_cfar_2d(
        ra_power, guard_cells=(2, 1), training_cells=(8, 4), pfa=1e-5
    )

    rd_db_cfar = np.where(rd_detections, rd_db, np.nan)
    doa_db_cfar = np.where(doa_detections, doa_db, np.nan)
    ra_db_cfar = np.where(ra_detections, ra_db, np.nan)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(title)

    extent_rd = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0] / 1e3,
        range_axis[-1] / 1e3,
    ]
    im_rd = axes[0, 0].imshow(
        rd_db,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="jet",
    )
    axes[0, 0].set_title("RD (no CFAR)")
    axes[0, 0].set_xlabel("Velocity (m/s)")
    axes[0, 0].set_ylabel("Range (km)")
    for target in targets:
        axes[0, 0].scatter(
            target.velocity_m_s,
            target.range_m / 1e3,
            marker="x",
            color="white",
            s=40,
        )
    fig.colorbar(im_rd, ax=axes[0, 0], label="Power (dB)")

    im_rd_cfar = axes[0, 1].imshow(
        rd_db_cfar,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="jet",
    )
    axes[0, 1].set_title("RD (CFAR)")
    axes[0, 1].set_xlabel("Velocity (m/s)")
    axes[0, 1].set_ylabel("Range (km)")
    fig.colorbar(im_rd_cfar, ax=axes[0, 1], label="Power (dB)")

    extent_ra = [
        az_deg[0],
        az_deg[-1],
        range_axis_dec[0] / 1e3,
        range_axis_dec[-1] / 1e3,
    ]
    im_ra = axes[1, 0].imshow(
        ra_db,
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap="jet",
    )
    axes[1, 0].set_title("RA (no CFAR)")
    axes[1, 0].set_xlabel("Azimuth (deg)")
    axes[1, 0].set_ylabel("Range (km)")
    for target in targets:
        axes[1, 0].scatter(
            np.rad2deg(target.azimuth_rad),
            target.range_m / 1e3,
            marker="x",
            color="white",
            s=40,
        )
    fig.colorbar(im_ra, ax=axes[1, 0], label="Power (dB)")

    im_ra_cfar = axes[1, 1].imshow(
        ra_db_cfar,
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap="jet",
    )
    axes[1, 1].set_title("RA (CFAR)")
    axes[1, 1].set_xlabel("Azimuth (deg)")
    axes[1, 1].set_ylabel("Range (km)")
    fig.colorbar(im_ra_cfar, ax=axes[1, 1], label="Power (dB)")

    extent_doa = [az_deg[0], az_deg[-1], el_deg[0], el_deg[-1]]
    im_doa = axes[2, 0].imshow(
        doa_db,
        origin="lower",
        aspect="auto",
        extent=extent_doa,
        cmap="jet",
    )
    axes[2, 0].set_title("Az-El MUSIC (no CFAR)")
    axes[2, 0].set_xlabel("Azimuth (deg)")
    axes[2, 0].set_ylabel("Elevation (deg)")
    for target in targets:
        axes[2, 0].scatter(
            np.rad2deg(target.azimuth_rad),
            np.rad2deg(target.elevation_rad),
            marker="x",
            color="white",
            s=40,
        )
    fig.colorbar(im_doa, ax=axes[2, 0], label="Spectrum (dB)")

    im_doa_cfar = axes[2, 1].imshow(
        doa_db_cfar,
        origin="lower",
        aspect="auto",
        extent=extent_doa,
        cmap="jet",
    )
    axes[2, 1].set_title("Az-El MUSIC (CFAR)")
    axes[2, 1].set_xlabel("Azimuth (deg)")
    axes[2, 1].set_ylabel("Elevation (deg)")
    fig.colorbar(im_doa_cfar, ax=axes[2, 1], label="Spectrum (dB)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])


def main() -> None:
    scenes = build_scenes()
    for title, waveform, geometry, targets in scenes:
        plot_scene(title, waveform, geometry, targets)
    plt.show()


if __name__ == "__main__":
    main()
