# 代码风格要
- setup 合适的 abstractions, i.e. classes (methods), functions. 争取做到 "don't repeat yourself"。代码是让人看的，需要逻辑清楚。
- 变量命名：要 meaningful, 要让人类 easy to read and easy to follow its logic.

# Package 使用
- 使用 numpy, scipy 作为科学计算工具包
- 使用 matplotlib 来作为画图工具包



# Project codebase summary
- Project uses uv to manage dependencies and run scripts (see `pyproject.toml`).
- Package layout: `radar_sim/` holds the simulation code, `basic_radar_signal_processing/` holds processing utilities, `tests/` holds runnable demo scripts.
- `radar_sim/signal_model.py`:
  - Core functions: `unit_direction`, `lfm_pulse`, `tx_baseband`.
  - Data classes: `ArrayGeometry`, `Waveform`, `Target`, `NoiseConfig`, `RadarConfig`.
  - `RadarSimulator` builds element positions, exposes time axes, and simulates RX baseband.
  - `rx_baseband` returns `X[p, q, fast_time, slow_time]` plus time axes, with optional noise.
  - On simulation start, it prints a radar performance summary (range/velocity resolution, PRF, CPI).
- `radar_sim/__init__.py` re-exports public classes and helpers.
- `basic_radar_signal_processing/processing.py`:
  - Range compression helpers: `matched_filter_taps`, `range_compress`, `range_axis_m`.
  - Doppler processing: `doppler_process`, `doppler_axis_hz`.
  - Angle processing: `build_steering_matrix`, `bartlett_beamform`, `fft_beamform_2d`, `spatial_frequency_axes`.
- `basic_radar_signal_processing/__init__.py` re-exports processing utilities.
- Demo scripts:
  - `tests/test_tx_baseband.py`: plots a single-PRI TX baseband time/spectrum.
  - `tests/test_rx_baseband.py`: plots each element's CPI time/spectrum (with noise).
  - `tests/test_range_compression.py`: compares raw vs range-compressed pulse.
  - `tests/test_doppler_processing.py`: plots the range-Doppler map.
  - `tests/test_bartlett_dbf.py`: Bartlett DBF angle response for a chosen range/Doppler bin.
  - `tests/test_fft_dbf.py`: 2D FFT beamforming in direction-cosine space.
  - `quick_start_for_basic_radar_signal_processing.py`: end-to-end walkthrough with plots.
- Run demos with uv:
  - `uv run python tests/test_tx_baseband.py`
  - `uv run python tests/test_rx_baseband.py`
  - `uv run python tests/test_range_compression.py`
  - `uv run python tests/test_doppler_processing.py`
  - `uv run python tests/test_bartlett_dbf.py`
  - `uv run python tests/test_fft_dbf.py`
  - `uv run python quick_start_for_basic_radar_signal_processing.py`
