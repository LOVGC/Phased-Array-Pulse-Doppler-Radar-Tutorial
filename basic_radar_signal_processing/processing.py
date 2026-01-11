from __future__ import annotations

import numpy as np
from scipy import constants, signal

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
    return np.conj(reference[::-1]) # shape (num_ref, )


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


def estimate_covariance(
    snapshots: np.ndarray, diagonal_loading: float = 0.0
) -> np.ndarray:
    if snapshots.ndim == 1:
        snapshots = snapshots[:, None]
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape [num_elements, num_snapshots].")
    num_snapshots = snapshots.shape[1]
    if num_snapshots < 1:
        raise ValueError("snapshots must contain at least one snapshot.")
    covariance = snapshots @ snapshots.conj().T / num_snapshots
    if diagonal_loading > 0.0:
        loading = diagonal_loading * np.trace(covariance).real / covariance.shape[0]
        covariance = covariance + loading * np.eye(covariance.shape[0])
    return covariance


def music_spectrum(
    covariance: np.ndarray,
    steering: np.ndarray,
    num_sources: int,
    floor: float = 1e-12,
) -> np.ndarray:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    num_elements = covariance.shape[0]
    if steering.ndim != 2 or steering.shape[1] != num_elements:
        raise ValueError("steering must have shape [num_angles, num_elements].")
    if not (1 <= num_sources < num_elements):
        raise ValueError("num_sources must be between 1 and num_elements - 1.")

    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)
    noise_eigvecs = eigvecs[:, order[: num_elements - num_sources]]
    proj = noise_eigvecs.conj().T @ steering.T
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    return 1.0 / np.maximum(denom, floor)


def music_spectrum_from_snapshots(
    snapshots: np.ndarray,
    steering: np.ndarray,
    num_sources: int,
    diagonal_loading: float = 0.0,
    floor: float = 1e-12,
) -> np.ndarray:
    covariance = estimate_covariance(
        snapshots, diagonal_loading=diagonal_loading
    )
    return music_spectrum(covariance, steering, num_sources, floor=floor)


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


def _ca_cfar_nd(
    power: np.ndarray,
    guard_cells: tuple[int, ...],
    training_cells: tuple[int, ...],
    pfa: float,
) -> tuple[np.ndarray, np.ndarray]:
    if power.ndim != len(guard_cells) or power.ndim != len(training_cells):
        raise ValueError("guard_cells and training_cells must match power dimensions.")
    if any(value < 0 for value in guard_cells + training_cells):
        raise ValueError("guard_cells and training_cells must be non-negative.")
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must be in (0, 1).")

    half_sizes = [g + t for g, t in zip(guard_cells, training_cells)]
    kernel_big = np.ones(tuple(2 * size + 1 for size in half_sizes))
    kernel_guard = np.ones(tuple(2 * size + 1 for size in guard_cells))
    num_training = int(kernel_big.size - kernel_guard.size)
    if num_training <= 0:
        raise ValueError("Training window must contain at least one cell.")

    alpha = num_training * (pfa ** (-1.0 / num_training) - 1.0)
    sum_big = signal.convolve(power, kernel_big, mode="same")
    sum_guard = signal.convolve(power, kernel_guard, mode="same")
    training_sum = sum_big - sum_guard
    threshold = alpha * training_sum / num_training

    slices = []
    for half in half_sizes:
        if half == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(half, -half))
    valid = np.zeros_like(power, dtype=bool)
    valid[tuple(slices)] = True
    threshold = np.where(valid, threshold, np.nan)
    detections = (power > threshold) & valid
    return threshold, detections


def ca_cfar_2d(
    power_map: np.ndarray,
    guard_cells: tuple[int, int],
    training_cells: tuple[int, int],
    pfa: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if power_map.ndim != 2:
        raise ValueError("power_map must be 2D.")
    return _ca_cfar_nd(power_map, guard_cells, training_cells, pfa)


def ca_cfar_3d(
    power_cube: np.ndarray,
    guard_cells: tuple[int, int, int],
    training_cells: tuple[int, int, int],
    pfa: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if power_cube.ndim != 3:
        raise ValueError("power_cube must be 3D.")
    return _ca_cfar_nd(power_cube, guard_cells, training_cells, pfa)
