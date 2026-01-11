from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from basic_radar_signal_processing.processing import (
    build_steering_matrix,
    ca_cfar_2d,
    ca_cfar_3d,
    doppler_axis_hz,
    doppler_process,
    range_axis_m,
    range_compress,
)
from utils import build_demo_scene, simulate_rx


def main() -> None:
    waveform, geometry, targets = build_demo_scene()
    rx, fast_time, _ = simulate_rx(waveform, geometry, targets)

    range_cube = range_compress(rx, waveform)
    doppler_cube = doppler_process(range_cube, window="hann")

    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)
    doppler_axis = doppler_axis_hz(waveform.num_pulses, waveform.pri)
    velocity_axis = doppler_axis * waveform.wavelength / 2.0

    az_deg = np.arange(-60.0, 60.1, 2.0)
    el_deg = np.arange(-6.0, 6.1, 2.0)
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    steering = build_steering_matrix(geometry, waveform.wavelength, az_rad, el_rad)
    num_elements = geometry.num_x * geometry.num_y
    num_range = doppler_cube.shape[2]
    num_doppler = doppler_cube.shape[3]

    snapshots = doppler_cube.reshape(num_elements, -1)
    responses = steering.conj() @ snapshots
    responses = responses.reshape(el_deg.size, az_deg.size, num_range, num_doppler)
    power = np.abs(responses) ** 2
    rad_cube = np.max(power, axis=0)
    rad_cube = np.transpose(rad_cube, (1, 0, 2))

    range_decimation = 4
    rad_cube = rad_cube[::range_decimation, :, :]
    range_axis = range_axis[::range_decimation]

    ra_map = np.max(rad_cube, axis=2)
    ra_db = 10.0 * np.log10(ra_map / np.max(ra_map) + 1e-12)

    _, ra_detections = ca_cfar_2d(
        ra_map,
        guard_cells=(2, 1),
        training_cells=(8, 4),
        pfa=1e-5,
    )
    _, rad_detections = ca_cfar_3d(
        rad_cube,
        guard_cells=(2, 1, 1),
        training_cells=(8, 3, 3),
        pfa=1e-6,
    )

    fig_ra, ax_ra = plt.subplots(figsize=(9, 6))
    extent_ra = [az_deg[0], az_deg[-1], range_axis[0] / 1e3, range_axis[-1] / 1e3]
    im_ra = ax_ra.imshow(
        ra_db,
        origin="lower",
        aspect="auto",
        extent=extent_ra,
        cmap="magma",
    )
    ax_ra.set_xlabel("Azimuth (deg)")
    ax_ra.set_ylabel("Range (km)")
    ax_ra.set_title("Range-Azimuth Map (max over elevation, doppler)")
    fig_ra.colorbar(im_ra, ax=ax_ra, label="Power (dB)")

    det_r, det_az = np.where(ra_detections)
    if det_r.size:
        ax_ra.scatter(
            az_deg[det_az],
            range_axis[det_r] / 1e3,
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
            label="CFAR",
        )
    for target in targets:
        ax_ra.scatter(
            np.rad2deg(target.azimuth_rad),
            target.range_m / 1e3,
            marker="x",
            color="white",
            s=50,
        )
    if det_r.size:
        ax_ra.legend(loc="upper right")

    det_r, det_az, det_d = np.where(rad_detections)
    fig_rad = plt.figure(figsize=(10, 6))
    ax_rad: Axes3D = fig_rad.add_subplot(111, projection="3d")
    if det_r.size:
        ax_rad.scatter(
            az_deg[det_az],
            range_axis[det_r] / 1e3,
            velocity_axis[det_d],
            s=8,
            c="tab:orange",
            alpha=0.7,
            label="CFAR",
        )
    for target in targets:
        ax_rad.scatter(
            np.rad2deg(target.azimuth_rad),
            target.range_m / 1e3,
            target.velocity_m_s,
            marker="x",
            color="black",
            s=50,
        )
    ax_rad.set_xlabel("Azimuth (deg)")
    ax_rad.set_ylabel("Range (km)")
    ax_rad.set_zlabel("Velocity (m/s)")
    ax_rad.set_title("RAD CFAR Detections")
    if det_r.size:
        ax_rad.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
