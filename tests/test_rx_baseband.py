import numpy as np
import matplotlib.pyplot as plt

from radar_sim.signal_model import ArrayGeometry, NoiseConfig, RadarSimulator, Target, Waveform


def main() -> None:
    waveform = Waveform(
        pulse_width=20e-6,
        bandwidth=5e6,
        carrier_frequency=10e9,
        sample_rate=40e6,
        pri=200e-6,
        num_pulses=8,
    )
    geometry = ArrayGeometry(
        num_x=2,
        num_y=2,
        dx=0.5 * waveform.wavelength,
        dy=0.5 * waveform.wavelength,
    )
    targets = [
        Target(
            range_m=8_000.0,
            velocity_m_s=30.0,
            azimuth_rad=np.deg2rad(15.0),
            elevation_rad=np.deg2rad(5.0),
            amplitude=1.0 + 0.0j,
        ),
        Target(
            range_m=4_500.0,
            velocity_m_s=-20.0,
            azimuth_rad=np.deg2rad(-10.0),
            elevation_rad=np.deg2rad(0.0),
            amplitude=0.7 + 0.2j,
        ),
    ]

    simulator = RadarSimulator(geometry, waveform)
    rx, fast_time, slow_time = simulator.rx_baseband(
        targets, noise=NoiseConfig(std=0.02, seed=0)
    )

    num_fast = fast_time.size
    num_pulses = slow_time.size
    time_cpi = np.arange(num_fast * num_pulses) / waveform.sample_rate # 这个就是又把时间轴拉直了。

    num_elements = geometry.num_x * geometry.num_y
    fig, axes = plt.subplots(
        num_elements, 2, figsize=(12, 3 * num_elements), squeeze=False
    )

    element_index = 0
    for p in range(geometry.num_x):
        for q in range(geometry.num_y):
            signal = rx[p, q].T.reshape(-1)

            axes[element_index, 0].plot(time_cpi * 1e3, np.abs(signal))
            axes[element_index, 0].set_xlabel("Time (ms)")
            axes[element_index, 0].set_ylabel("Magnitude")
            axes[element_index, 0].set_title(f"Element ({p}, {q}) time domain")

            nfft = int(2 ** np.ceil(np.log2(signal.size)))
            spectrum = np.fft.fftshift(np.fft.fft(signal, n=nfft))
            freqs = np.fft.fftshift(
                np.fft.fftfreq(nfft, d=1.0 / waveform.sample_rate)
            )
            axes[element_index, 1].plot(
                freqs / 1e6, 20 * np.log10(np.abs(spectrum) + 1e-12)
            )
            axes[element_index, 1].set_xlabel("Frequency (MHz)")
            axes[element_index, 1].set_ylabel("Magnitude (dB)")
            axes[element_index, 1].set_title(f"Element ({p}, {q}) spectrum")

            element_index += 1

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
