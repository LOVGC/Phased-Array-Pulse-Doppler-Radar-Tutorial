# 代码风格要
- setup 合适的 abstractions, i.e. classes (methods), functions. 争取做到 "don't repeat yourself"。代码是让人看的，需要逻辑清楚。
- 变量命名：要 meaningful, 要让人类 easy to read and easy to follow its logic.

# Package 使用
- 使用 numpy, scipy 作为科学计算工具包
- 使用 matplotlib 来作为画图工具包



# Project codebase summary
- Project uses uv to manage dependencies and run scripts (see `pyproject.toml`).
- Package layout: `radar_sim/` holds the simulation code, `tests/` holds runnable demo scripts.
- `radar_sim/signal_model.py`:
  - Core functions: `unit_direction`, `lfm_pulse`, `tx_baseband`.
  - Data classes: `ArrayGeometry`, `Waveform`, `Target`, `NoiseConfig`, `RadarConfig`.
  - `RadarSimulator` builds element positions, exposes time axes, and simulates RX baseband.
  - `rx_baseband` returns `X[p, q, fast_time, slow_time]` plus time axes, with optional noise.
  - On simulation start, it prints a radar performance summary (range/velocity resolution, PRF, CPI).
- `radar_sim/__init__.py` re-exports public classes and helpers.
- Demo scripts:
  - `tests/test_tx_baseband.py`: plots a single-PRI TX baseband time/spectrum.
  - `tests/test_rx_baseband.py`: plots each element's CPI time/spectrum (with noise).
- Run demos with uv:
  - `uv run python tests/test_tx_baseband.py`
  - `uv run python tests/test_rx_baseband.py`
