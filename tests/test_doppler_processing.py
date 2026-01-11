from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import (
    ca_cfar_2d,
    doppler_axis_hz,
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
    doppler_axis = doppler_axis_hz(waveform.num_pulses, waveform.pri)
    velocity_axis = doppler_axis * waveform.wavelength / 2.0

    rd_power = np.sum(np.abs(doppler_cube) ** 2, axis=(0, 1))
    rd_db = 10.0 * np.log10(rd_power + 1e-12)
    _, detections = ca_cfar_2d(
        rd_power,
        guard_cells=(2, 1),
        training_cells=(12, 4),
        pfa=1e-5,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    extent = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0] / 1e3,
        range_axis[-1] / 1e3,
    ]
    im = ax.imshow(
        rd_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (km)")
    ax.set_title("Range-Doppler Map")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    det_r, det_d = np.where(detections)
    if det_r.size:
        ax.scatter(
            velocity_axis[det_d],
            range_axis[det_r] / 1e3,
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
            label="CFAR",
        )
        ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
