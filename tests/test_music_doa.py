from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import (
    build_steering_matrix,
    ca_cfar_2d,
    music_spectrum_from_snapshots,
    range_axis_m,
    range_compress,
)
from utils import build_demo_scene, simulate_rx


def main() -> None:
    waveform, geometry, targets = build_demo_scene()
    rx, fast_time, _ = simulate_rx(waveform, geometry, targets)

    range_cube = range_compress(rx, waveform)
    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)

    target = targets[0]
    range_idx = int(np.argmin(np.abs(range_axis - target.range_m)))
    snapshots = range_cube[:, :, range_idx, :].reshape(
        geometry.num_x * geometry.num_y, -1
    )

    az_deg = np.arange(-60.0, 60.1, 2.0)
    el_deg = np.arange(-10.0, 20.1, 2.0)
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    steering = build_steering_matrix(geometry, waveform.wavelength, az_rad, el_rad)
    spectrum = music_spectrum_from_snapshots(
        snapshots,
        steering,
        num_sources=1,
        diagonal_loading=1e-3,
    )
    spectrum_map = spectrum.reshape(el_deg.size, az_deg.size)
    spectrum_db = 10.0 * np.log10(spectrum_map / np.max(spectrum_map) + 1e-12)

    _, detections = ca_cfar_2d(
        spectrum_map,
        guard_cells=(1, 1),
        training_cells=(4, 4),
        pfa=1e-4,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [az_deg[0], az_deg[-1], el_deg[0], el_deg[-1]]
    im = ax.imshow(
        spectrum_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    ax.scatter(
        np.rad2deg(target.azimuth_rad),
        np.rad2deg(target.elevation_rad),
        marker="x",
        color="white",
        s=60,
        label="Target",
    )
    det_el, det_az = np.where(detections)
    if det_el.size:
        ax.scatter(
            az_deg[det_az],
            el_deg[det_el],
            s=20,
            marker="o",
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="CFAR",
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("MUSIC Az-El Spectrum (range bin of target 1)")
    fig.colorbar(im, ax=ax, label="Spectrum (dB)")
    plt.show()


if __name__ == "__main__":
    main()
