# 代码风格要
- setup 合适的 abstractions, i.e. classes (methods), functions. 争取做到 "don't repeat yourself"。代码是让人看的，需要逻辑清楚。
- 变量命名：要 meaningful, 要让人类 easy to read and easy to follow its logic.

# Package 使用
- 使用 numpy, scipy 作为科学计算工具包
- 使用 matplotlib 来作为画图工具包

# radar signal model knowledge base

关于目前知道的 radar signal model 以及 basic radar signal porcessing 的知识在./knowledge_base 这个文件夹中。你需要先 go through 一下这里面的知识。

# Project codebase summary
- Purpose: simulate a 2D phased-array pulse-Doppler radar (RX baseband), then perform basic signal processing to estimate range, velocity, and angle; include MUSIC DOA and CFAR detection on RD/RA/RAD outputs.
- Architecture: `radar_sim/` models waveform/array/targets and generates `X[p,q,fast_time,slow_time]`; `basic_radar_signal_processing/` provides range/Doppler/angle processing plus DOA and CFAR; `utils/` supplies shared demo scenes; `tests/` holds runnable plot-based demos.
- `radar_sim/signal_model.py`:
  - Core functions: `unit_direction`, `lfm_pulse`, `tx_baseband`.
  - Data classes: `ArrayGeometry`, `Waveform`, `Target`, `NoiseConfig`, `RadarConfig`.
  - `RadarSimulator` builds element positions, exposes time axes, and simulates RX baseband.
  - `rx_baseband` returns `X[p, q, fast_time, slow_time]` plus time axes, with optional noise.
  - Prints radar performance summary (range/velocity/angle resolution, PRF, CPI).
- `radar_sim/__init__.py` re-exports public classes and helpers.
- `basic_radar_signal_processing/processing.py`:
  - Range compression: `matched_filter_taps`, `range_compress`, `range_axis_m`.
  - Doppler processing: `doppler_process`, `doppler_axis_hz`.
  - Angle processing: `build_steering_matrix`, `bartlett_beamform`, `fft_beamform_2d`, `spatial_frequency_axes`.
  - DOA utilities: `estimate_covariance`, `music_spectrum`, `music_spectrum_from_snapshots`.
  - CFAR utilities: `ca_cfar_2d`, `ca_cfar_3d`.
- `basic_radar_signal_processing/__init__.py` re-exports processing utilities.
- `utils/demo_scene.py` provides `build_demo_scene` and `simulate_rx` for shared test scenarios.
- Demo scripts (plots):
  - `tests/test_tx_baseband.py`: single-PRI TX baseband time/spectrum.
  - `tests/test_rx_baseband.py`: element CPI time/spectrum (with noise).
  - `tests/test_range_compression.py`: raw vs range-compressed pulse.
  - `tests/test_doppler_processing.py`: range-Doppler map with CFAR overlay.
  - `tests/test_bartlett_dbf.py`: Bartlett DBF response for a range/Doppler bin.
  - `tests/test_fft_dbf.py`: 2D FFT beamforming in direction-cosine space.
  - `tests/test_music_doa.py`: MUSIC az-el spectrum with CFAR overlay.
  - `tests/test_ra_rad_cfar.py`: RA map and RAD CFAR detections.
  - `tests/test_multi_scene_cfar.py`: multiple scenes with RD/RA/Az-El pre/post CFAR.
  - `quick_start_for_basic_radar_signal_processing.py`: end-to-end walkthrough with plots.
  - `quick_start_for_signal_model_basic_process_cfar.py`: end-to-end with RD/RA/Az-El pre/post CFAR.
- How to run (uv):
  - `uv run python quick_start_for_basic_radar_signal_processing.py`
  - `uv run python quick_start_for_signal_model_basic_process_cfar.py`
  - `uv run python tests/test_doppler_processing.py`
  - `uv run python tests/test_music_doa.py`
  - `uv run python tests/test_ra_rad_cfar.py`
  - `uv run python tests/test_multi_scene_cfar.py`
