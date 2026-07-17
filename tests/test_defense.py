from __future__ import annotations

from pathlib import Path

import numpy as np

from psorad.defense.purify import fft_lowpass, local_median, purify, region_aware_mask


class TestPurify:
    def test_local_median_shape(self) -> None:
        img = np.random.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)
        result = local_median(img, window=3)
        assert result.shape == (32, 32, 3)

    def test_local_median_smooth(self) -> None:
        """Constant image: local median should be unchanged."""
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        result = local_median(img, window=3)
        assert np.allclose(result, 0.5)

    def test_region_aware_mask_no_anomaly(self) -> None:
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        mask = region_aware_mask(img, window=3, threshold=0.08)
        assert not mask.any()

    def test_region_aware_mask_detects_anomaly(self) -> None:
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        img[5:8, 5:8] = 1.0  # Inject bright patch
        mask = region_aware_mask(img, window=3, threshold=0.08)
        assert mask.any()
        # The anomaly region should be covered
        assert mask[5:8, 5:8].any()

    def test_fft_lowpass_shape(self) -> None:
        img = np.random.uniform(0.0, 1.0, size=(16, 16, 3)).astype(np.float32)
        result = fft_lowpass(img, cutoff=0.25)
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.float32

    def test_fft_lowpass_smooths_high_freq(self) -> None:
        """Checkerboard (pure high-freq): lowpass should smooth it."""
        img = np.zeros((16, 16, 3), dtype=np.float32)
        img[::2, ::2] = 1.0
        result = fft_lowpass(img, cutoff=0.1)
        # Result should be smoother (less variation)
        assert result.std() < img.std()

    def test_purify_basic(self) -> None:
        """Smooth background + few injected pixels: only injected pixels change."""
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        img[3:6, 3:6] = 1.0  # 3x3 anomaly
        result = purify(img, window=3, threshold=0.08, dilation=1, low_freq_cutoff=0.25, low_freq="fft")
        # Anomaly pixels should be reduced
        anomaly_region = result[3:6, 3:6]
        assert np.mean(anomaly_region) < 0.9  # significantly reduced
        # Non-anomaly region should be nearly unchanged
        bg = result[10:12, 10:12]
        assert np.allclose(bg, 0.5, atol=0.05)

    def test_purify_no_anomaly(self) -> None:
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        result = purify(img, window=3, threshold=0.08)
        assert np.allclose(result, 0.5, atol=0.01)

    def test_purify_output_range(self) -> None:
        img = np.random.uniform(0.0, 1.0, size=(16, 16, 3)).astype(np.float32)
        result = purify(img, window=3, threshold=0.08)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_purify_median_mode(self) -> None:
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        img[3:6, 3:6] = 1.0
        result = purify(img, window=3, threshold=0.08, low_freq="median")
        assert result.shape == (16, 16, 3)


class TestLoadDefenseConfig:
    def test_load_from_toml(self, tmp_path: Path) -> None:
        from psorad.config import load_defense_config

        toml_path = tmp_path / "test_defense.toml"
        toml_path.write_text(
            """
[Defense]
method = "freq_region_purify"
backbone = "resnet50"
checkpoint = "model.pt"
batch_report = "report.json"

[Defense.purify]
window = 5
threshold = 0.1
dilation = 2
low_freq_cutoff = 0.3
low_freq = "median"

[Output]
output_dir = "output/test_defense"
report_name = "test_report.json"
""",
            encoding="utf-8",
        )

        cfg = load_defense_config(str(toml_path))
        assert cfg.backbone == "resnet50"
        assert cfg.checkpoint == "model.pt"
        assert cfg.batch_report == "report.json"
        assert cfg.window == 5
        assert cfg.threshold == 0.1
        assert cfg.dilation == 2
        assert cfg.low_freq_cutoff == 0.3
        assert cfg.low_freq == "median"
        assert str(cfg.output_dir) == "output/test_defense"
        assert cfg.report_name == "test_report.json"
