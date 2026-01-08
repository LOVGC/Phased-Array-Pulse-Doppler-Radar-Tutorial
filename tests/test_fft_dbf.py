from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import (
    fft_beamform_2d,
    range_axis_m,
    range_compress,
    spatial_frequency_axes,
    doppler_process,
)
from tests.utils import build_demo_scene, simulate_rx


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

    doppler_slice = doppler_cube[:, :, range_idx : range_idx + 1, doppler_idx : doppler_idx + 1]
    spectrum = fft_beamform_2d(
        doppler_slice, window="hann", fft_size_x=64, fft_size_y=64
    )
    spectrum_2d = spectrum[:, :, 0, 0]

    u_x, u_y = spatial_frequency_axes(
        geometry.num_x,
        geometry.num_y,
        geometry.dx,
        geometry.dy,
        waveform.wavelength,
        fft_size_x=64,
        fft_size_y=64,
    )

    power_db = 20.0 * np.log10(np.abs(spectrum_2d) / np.max(np.abs(spectrum_2d)) + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [u_x[0], u_x[-1], u_y[0], u_y[-1]]
    im = ax.imshow(
        power_db.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    ax.set_xlabel("Direction cosine u_x")
    ax.set_ylabel("Direction cosine u_y")
    ax.set_title("2D FFT Beamforming (single range/Doppler bin)")
    fig.colorbar(im, ax=ax, label="Response (dB)")
    plt.show()


if __name__ == "__main__":
    main()
