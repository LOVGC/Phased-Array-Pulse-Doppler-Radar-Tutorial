from __future__ import annotations

import numpy as np
from scipy import constants

from radar_sim.signal_model import ArrayGeometry, Waveform, lfm_pulse


def next_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return int(2 ** np.ceil(np.log2(value)))


def matched_filter_taps(waveform: Waveform) -> np.ndarray:
    num_ref = int(np.round(waveform.pulse_width * waveform.sample_rate))
    if num_ref < 1:
        raise ValueError("Pulse width too small for the current sample rate.")
    t_ref = np.arange(num_ref) / waveform.sample_rate
    reference = lfm_pulse(t_ref, waveform.pulse_width, waveform.bandwidth)
    return np.conj(reference[::-1])


def range_compress(rx: np.ndarray, waveform: Waveform) -> np.ndarray:
    if rx.ndim != 4:
        raise ValueError("rx must have shape [num_x, num_y, fast_time, slow_time].")
    num_fast = rx.shape[2]
    taps = matched_filter_taps(waveform)
    nfft = next_pow2(num_fast + taps.size - 1)
    taps_f = np.fft.fft(taps, n=nfft)
    rx_f = np.fft.fft(rx, n=nfft, axis=2)
    compressed_full = np.fft.ifft(rx_f * taps_f[None, None, :, None], axis=2)
    start = taps.size - 1
    return compressed_full[:, :, start : start + num_fast, :]


def doppler_process(range_cube: np.ndarray, window: str | None = "hann") -> np.ndarray:
    if range_cube.ndim != 4:
        raise ValueError(
            "range_cube must have shape [num_x, num_y, range, slow_time]."
        )
    num_pulses = range_cube.shape[3]
    if window is None:
        window_values = np.ones(num_pulses)
    elif window.lower() == "hann":
        window_values = np.hanning(num_pulses)
    elif window.lower() == "hamming":
        window_values = np.hamming(num_pulses)
    else:
        raise ValueError(f"Unsupported window: {window}")

    windowed = range_cube * window_values[None, None, None, :]
    doppler = np.fft.fftshift(np.fft.fft(windowed, axis=3), axes=3)
    return doppler


def doppler_axis_hz(num_pulses: int, pri: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(num_pulses, d=pri))


def range_axis_m(num_fast: int, sample_rate: float) -> np.ndarray:
    fast_time = np.arange(num_fast) / sample_rate
    return constants.c * fast_time / 2.0


def build_steering_matrix(
    geometry: ArrayGeometry, wavelength: float, az_rad: np.ndarray, el_rad: np.ndarray
) -> np.ndarray:
    az = az_rad[None, :]
    el = el_rad[:, None]
    cos_el = np.cos(el)
    dir_x = cos_el * np.cos(az)
    dir_y = cos_el * np.sin(az)
    dir_z = np.broadcast_to(np.sin(el), dir_x.shape)
    directions = np.stack([dir_x, dir_y, dir_z], axis=-1)

    directions_flat = directions.reshape(-1, 3)
    positions = geometry.element_positions().reshape(-1, 3)
    phase = (2.0 * np.pi / wavelength) * (directions_flat @ positions.T)
    return np.exp(1j * phase)


def bartlett_beamform(
    snapshot: np.ndarray, steering: np.ndarray, num_el: int, num_az: int
) -> np.ndarray:
    if snapshot.ndim != 1:
        raise ValueError("snapshot must be a 1D array of array elements.")
    response = steering.conj() @ snapshot
    return response.reshape(num_el, num_az)


def fft_beamform_2d(
    doppler_cube: np.ndarray,
    window: str | None = "hann",
    fft_size_x: int | None = None,
    fft_size_y: int | None = None,
) -> np.ndarray:
    if doppler_cube.ndim != 4:
        raise ValueError(
            "doppler_cube must have shape [num_x, num_y, range, doppler]."
        )
    num_x, num_y = doppler_cube.shape[:2]
    if window is None:
        window_x = np.ones(num_x)
        window_y = np.ones(num_y)
    elif window.lower() == "hann":
        window_x = np.hanning(num_x)
        window_y = np.hanning(num_y)
    elif window.lower() == "hamming":
        window_x = np.hamming(num_x)
        window_y = np.hamming(num_y)
    else:
        raise ValueError(f"Unsupported window: {window}")

    win2d = window_x[:, None] * window_y[None, :]
    weighted = doppler_cube * win2d[:, :, None, None]
    size_x = fft_size_x or num_x
    size_y = fft_size_y or num_y
    spectrum = np.fft.fftshift(
        np.fft.fft2(weighted, s=(size_x, size_y), axes=(0, 1)),
        axes=(0, 1),
    )
    return spectrum


def spatial_frequency_axes(
    num_x: int,
    num_y: int,
    dx: float,
    dy: float,
    wavelength: float,
    fft_size_x: int | None = None,
    fft_size_y: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    size_x = fft_size_x or num_x
    size_y = fft_size_y or num_y
    freq_x = np.fft.fftshift(np.fft.fftfreq(size_x, d=1.0))
    freq_y = np.fft.fftshift(np.fft.fftfreq(size_y, d=1.0))
    u_x = (wavelength / dx) * freq_x
    u_y = (wavelength / dy) * freq_y
    return u_x, u_y
