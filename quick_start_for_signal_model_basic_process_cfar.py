from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

from basic_radar_signal_processing.processing import (
    build_steering_matrix,
    ca_cfar_2d,
    doppler_axis_hz,
    doppler_process,
    music_spectrum_from_snapshots,
    range_axis_m,
    range_compress,
)
from utils import build_demo_scene, simulate_rx


PARULA_DATA = [
    [0.2081, 0.1663, 0.5292],
    [0.2116, 0.1898, 0.5777],
    [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771],
    [0.1959, 0.2645, 0.7279],
    [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303],
    [0.0591, 0.3598, 0.8683],
    [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828],
    [0.0165, 0.4266, 0.8786],
    [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641],
    [0.0629, 0.4737, 0.8554],
    [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384],
    [0.0793, 0.5200, 0.8312],
    [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240],
    [0.0488, 0.5772, 0.8228],
    [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135],
    [0.0239, 0.6287, 0.8038],
    [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768],
    [0.0267, 0.6642, 0.7607],
    [0.0384, 0.6743, 0.7436],
    [0.0587, 0.6838, 0.7254],
    [0.0843, 0.6928, 0.7062],
    [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646],
    [0.1801, 0.7177, 0.6424],
    [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954],
    [0.3022, 0.7376, 0.5712],
    [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244],
    [0.4420, 0.7481, 0.5033],
    [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661],
    [0.5711, 0.7485, 0.4494],
    [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188],
    [0.6834, 0.7435, 0.4044],
    [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768],
    [0.7858, 0.7356, 0.3633],
    [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360],
    [0.8824, 0.7274, 0.3217],
    [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886],
    [0.9739, 0.7314, 0.2666],
    [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164],
    [0.9955, 0.7861, 0.1967],
    [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633],
    [0.9697, 0.8481, 0.1475],
    [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132],
    [0.9598, 0.9218, 0.0948],
    [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
]
PARULA = LinearSegmentedColormap.from_list("parula", PARULA_DATA)


def main() -> None:
    rd_cmap = PARULA
    doa_cmap = PARULA
    ra_cmap = PARULA

    waveform, geometry, targets = build_demo_scene()
    rx, fast_time, _ = simulate_rx(waveform, geometry, targets)

    range_cube = range_compress(rx, waveform)
    doppler_cube = doppler_process(range_cube, window="hann")

    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)
    doppler_axis = doppler_axis_hz(waveform.num_pulses, waveform.pri)
    velocity_axis = doppler_axis * waveform.wavelength / 2.0

    rd_power = np.sum(np.abs(doppler_cube) ** 2, axis=(0, 1))
    rd_db = 10.0 * np.log10(rd_power + 1e-12)
    _, rd_detections = ca_cfar_2d(
        rd_power,
        guard_cells=(1, 0),
        training_cells=(6, 1),
        pfa=1e-7,
    )
    rd_db_cfar = np.where(rd_detections, rd_db, np.nan)

    az_deg = np.arange(-60.0, 60.1, 1.0)
    el_deg = np.arange(-10.0, 20.1, 1.0)
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
    _, doa_detections = ca_cfar_2d(
        doa_power,
        guard_cells=(1, 1),
        training_cells=(4, 4),
        pfa=1e-4,
    )
    doa_db_cfar = np.where(doa_detections, doa_db, np.nan)

    range_decimation = 4
    doppler_cube_dec = doppler_cube[:, :, ::range_decimation, :]
    range_axis_dec = range_axis[::range_decimation]
    num_elements = geometry.num_x * geometry.num_y
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
    _, ra_detections = ca_cfar_2d(
        ra_power,
        guard_cells=(2, 1),
        training_cells=(8, 4),
        pfa=1e-5,
    )
    ra_db_cfar = np.where(ra_detections, ra_db, np.nan)

    rd_vmax = np.nanmax(rd_db)
    rd_vmin = rd_vmax - 40.0
    fig_rd, axes_rd = plt.subplots(1, 2, figsize=(12, 5))
    extent_rd = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0] / 1e3,
        range_axis[-1] / 1e3,
    ]
    im_rd = axes_rd[0].imshow(
        rd_db,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap=rd_cmap,
        vmin=rd_vmin,
        vmax=rd_vmax,
    )
    axes_rd[0].set_title("RD Map (no CFAR)")
    axes_rd[0].set_xlabel("Velocity (m/s)")
    axes_rd[0].set_ylabel("Range (km)")
    for tgt in targets:
        axes_rd[0].scatter(
            tgt.velocity_m_s,
            tgt.range_m / 1e3,
            marker="x",
            color="white",
            s=40,
        )
    fig_rd.colorbar(im_rd, ax=axes_rd[0], label="Power (dB)")

    im_rd_cfar = axes_rd[1].imshow(
        rd_db_cfar,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap=rd_cmap,
        vmin=rd_vmin,
        vmax=rd_vmax,
    )
    axes_rd[1].set_title("RD Map (CFAR)")
    axes_rd[1].set_xlabel("Velocity (m/s)")
    axes_rd[1].set_ylabel("Range (km)")
    fig_rd.colorbar(im_rd_cfar, ax=axes_rd[1], label="Power (dB)")

    doa_vmax = np.nanmax(doa_db)
    doa_vmin = doa_vmax - 30.0
    fig_doa, axes_doa = plt.subplots(1, 2, figsize=(12, 5))
    extent_doa = [az_deg[0], az_deg[-1], el_deg[0], el_deg[-1]]
    im_doa = axes_doa[0].imshow(
        doa_db,
        origin="lower",
        aspect="auto",
        extent=extent_doa,
        cmap=doa_cmap,
        vmin=doa_vmin,
        vmax=doa_vmax,
    )
    for target in targets:
        axes_doa[0].scatter(
            np.rad2deg(target.azimuth_rad),
            np.rad2deg(target.elevation_rad),
            marker="x",
            color="white",
            s=60,
        )
    axes_doa[0].set_title("Az-El MUSIC (no CFAR)")
    axes_doa[0].set_xlabel("Azimuth (deg)")
    axes_doa[0].set_ylabel("Elevation (deg)")
    fig_doa.colorbar(im_doa, ax=axes_doa[0], label="Spectrum (dB)")

    im_doa_cfar = axes_doa[1].imshow(
        doa_db_cfar,
        origin="lower",
        aspect="auto",
        extent=extent_doa,
        cmap=doa_cmap,
        vmin=doa_vmin,
        vmax=doa_vmax,
    )
    axes_doa[1].set_title("Az-El MUSIC (CFAR)")
    axes_doa[1].set_xlabel("Azimuth (deg)")
    axes_doa[1].set_ylabel("Elevation (deg)")
    fig_doa.colorbar(im_doa_cfar, ax=axes_doa[1], label="Spectrum (dB)")

    az_grid, el_grid = np.meshgrid(az_deg, el_deg)
    fig_doa_3d = plt.figure(figsize=(10, 6))
    ax_doa_3d: Axes3D = fig_doa_3d.add_subplot(111, projection="3d")
    ax_doa_3d.plot_surface(
        az_grid,
        el_grid,
        doa_db,
        cmap=doa_cmap,
        linewidth=0.0,
        antialiased=True,
        alpha=0.9,
    )
    ax_doa_3d.set_xlabel("Azimuth (deg)")
    ax_doa_3d.set_ylabel("Elevation (deg)")
    ax_doa_3d.set_zlabel("Spectrum (dB)")
    ax_doa_3d.set_title("Az-El MUSIC Surface (no CFAR)")

    det_el, det_az = np.where(doa_detections)
    fig_doa_3d_cfar = plt.figure(figsize=(10, 6))
    ax_doa_3d_cfar: Axes3D = fig_doa_3d_cfar.add_subplot(111, projection="3d")
    ax_doa_3d_cfar.plot_surface(
        az_grid,
        el_grid,
        doa_db,
        cmap=doa_cmap,
        linewidth=0.0,
        antialiased=True,
        alpha=0.35,
    )
    if det_el.size:
        ax_doa_3d_cfar.scatter(
            az_deg[det_az],
            el_deg[det_el],
            doa_db[det_el, det_az],
            s=25,
            c="tab:orange",
            label="CFAR",
        )
        ax_doa_3d_cfar.legend(loc="upper right")
    ax_doa_3d_cfar.set_xlabel("Azimuth (deg)")
    ax_doa_3d_cfar.set_ylabel("Elevation (deg)")
    ax_doa_3d_cfar.set_zlabel("Spectrum (dB)")
    ax_doa_3d_cfar.set_title("Az-El MUSIC Surface (CFAR)")

    ra_vmax = np.nanmax(ra_db)
    ra_vmin = ra_vmax - 40.0
    fig_ra, axes_ra = plt.subplots(1, 2, figsize=(12, 5))
    extent_ra = [
        az_deg[0],
        az_deg[-1],
        range_axis_dec[0] / 1e3,
        range_axis_dec[-1] / 1e3,
    ]
    im_ra = axes_ra[0].imshow(
        ra_db,
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap=ra_cmap,
        vmin=ra_vmin,
        vmax=ra_vmax,
    )
    axes_ra[0].set_title("RA Map (no CFAR)")
    axes_ra[0].set_xlabel("Azimuth (deg)")
    axes_ra[0].set_ylabel("Range (km)")
    for tgt in targets:
        axes_ra[0].scatter(
            np.rad2deg(tgt.azimuth_rad),
            tgt.range_m / 1e3,
            marker="x",
            color="white",
            s=40,
        )
    fig_ra.colorbar(im_ra, ax=axes_ra[0], label="Power (dB)")

    im_ra_cfar = axes_ra[1].imshow(
        ra_db_cfar,
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap=ra_cmap,
        vmin=ra_vmin,
        vmax=ra_vmax,
    )
    axes_ra[1].set_title("RA Map (CFAR)")
    axes_ra[1].set_xlabel("Azimuth (deg)")
    axes_ra[1].set_ylabel("Range (km)")
    fig_ra.colorbar(im_ra_cfar, ax=axes_ra[1], label="Power (dB)")

    plt.show()


if __name__ == "__main__":
    main()
