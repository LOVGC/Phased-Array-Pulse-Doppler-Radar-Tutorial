import numpy as np
import matplotlib.pyplot as plt

from radar_sim.signal_model import Waveform, tx_baseband


def main() -> None:
    waveform = Waveform(
        pulse_width=20e-6,
        bandwidth=10e6,
        carrier_frequency=10e9,
        sample_rate=40e6,
        pri=200e-6,
        num_pulses=1,
    )

    fast_time = np.arange(waveform.fast_time_samples) / waveform.sample_rate
    tx = tx_baseband(fast_time, waveform)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(fast_time * 1e6, np.real(tx), label="I")
    axes[0].plot(fast_time * 1e6, np.imag(tx), label="Q", alpha=0.8)
    axes[0].set_xlabel("Time (us)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("TX baseband in one PRI")
    axes[0].legend()

    nfft = int(2 ** np.ceil(np.log2(tx.size)))
    spectrum = np.fft.fftshift(np.fft.fft(tx, n=nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / waveform.sample_rate))

    axes[1].plot(freqs / 1e6, 20 * np.log10(np.abs(spectrum) + 1e-12))
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("Spectrum")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
