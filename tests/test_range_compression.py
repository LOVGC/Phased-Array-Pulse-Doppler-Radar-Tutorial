from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from basic_radar_signal_processing.processing import range_axis_m, range_compress
from utils import build_demo_scene, simulate_rx


def main() -> None:
    waveform, geometry, targets = build_demo_scene()
    rx, fast_time, _ = simulate_rx(waveform, geometry, targets)

    compressed = range_compress(rx, waveform)
    range_axis = range_axis_m(fast_time.size, waveform.sample_rate)

    raw = rx[0, 0, :, 0]
    comp = compressed[0, 0, :, 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(range_axis / 1e3, np.abs(raw))
    axes[0].set_xlabel("Range (km)")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Raw pulse (element 0,0)")

    axes[1].plot(range_axis / 1e3, np.abs(comp))
    axes[1].set_xlabel("Range (km)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Range-compressed pulse (element 0,0)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
