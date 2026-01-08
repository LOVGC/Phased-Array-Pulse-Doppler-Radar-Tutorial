from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import (
    bartlett_beamform,
    build_steering_matrix,
    doppler_process,
    range_axis_m,
    range_compress,
)
from utils import build_demo_scene, simulate_rx


def main() -> None:
    waveform, geometry, targets = build_demo_scene()
    rx, fast_time, _ = simulate_rx(waveform, geometry, targets)

    compressed = range_compress(rx, waveform)
    doppler_cube = doppler_process(compressed, window="hann")

    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)
    target = targets[0]
    range_idx = int(np.argmin(np.abs(range_axis - target.range_m)))

    doppler_power = np.sum(np.abs(doppler_cube[:, :, range_idx, :]) ** 2, axis=(0, 1))
    doppler_idx = int(np.argmax(doppler_power))
    snapshot = doppler_cube[:, :, range_idx, doppler_idx].reshape(-1)

    az_deg = np.arange(-60.0, 60.1, 2.0)
    el_deg = np.arange(-20.0, 20.1, 2.0)
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    steering = build_steering_matrix(geometry, waveform.wavelength, az_rad, el_rad)
    response = bartlett_beamform(snapshot, steering, el_rad.size, az_rad.size)
    response_db = 20.0 * np.log10(np.abs(response) / np.max(np.abs(response)) + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [az_deg[0], az_deg[-1], el_deg[0], el_deg[-1]]
    im = ax.imshow(
        response_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Bartlett DBF (single range/Doppler bin)")
    fig.colorbar(im, ax=ax, label="Response (dB)")
    plt.show()


if __name__ == "__main__":
    main()
