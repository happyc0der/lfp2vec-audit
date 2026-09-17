"""Tests for the Stage 3 model, dataset and trainer.

Everything here runs offline on synthetic data. The model tests need the audio checkpoint, so
they skip when it is not cached rather than downloading 360 MB inside continuous integration.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lfpaudit import REGIONS
from lfpaudit.models.dataset import ChunkDataset, balanced_sample, class_counts
from lfpaudit.models.train import FineTuneConfig, _linear_warmup_decay

torch = pytest.importorskip("torch")


def _fake_take(fs: float = 1250.0, n_samples: int = 3750):
    rng = np.random.default_rng(0)
    table = rng.normal(size=(20, n_samples))

    def take(ids):
        return table[np.asarray(ids, dtype=np.int64)]

    return take


class TestDataset:
    def test_items_are_model_ready(self):
        dataset = ChunkDataset(_fake_take(), np.arange(6), np.zeros(6, dtype=int), fs=1250.0)
        waveform, label = dataset[0]
        # Three seconds at the model's rate, not the stored rate.
        assert waveform.shape == (48000,)
        assert waveform.dtype == torch.float32
        assert label == 0

    def test_items_are_normalised(self):
        dataset = ChunkDataset(_fake_take(), np.arange(6), np.zeros(6, dtype=int), fs=1250.0)
        waveform = dataset[2][0].numpy()
        assert abs(waveform.mean()) < 1e-4
        assert abs(waveform.std() - 1.0) < 1e-2

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="chunks for"):
            ChunkDataset(_fake_take(), np.arange(6), np.zeros(3, dtype=int), fs=1250.0)

    def test_lowpass_is_applied(self):
        fs, n = 1250.0, 3750
        t = np.arange(n) / fs
        loud = np.tile(np.sin(2 * np.pi * 300 * t), (4, 1))

        plain = ChunkDataset(lambda i: loud[np.asarray(i)], np.arange(4), np.zeros(4, int), fs=fs)
        filtered = ChunkDataset(
            lambda i: loud[np.asarray(i)], np.arange(4), np.zeros(4, int), fs=fs, lowpass_hz=100.0
        )
        # A 300 Hz tone survives normalisation untouched but not a 100 Hz low-pass.
        assert np.abs(filtered[0][0].numpy()).mean() < np.abs(plain[0][0].numpy()).mean()


class TestBalancedSample:
    def test_caps_each_class(self):
        labels = np.array([0] * 100 + [3] * 100 + [4] * 5)
        chosen = balanced_sample(labels, np.arange(len(labels)), max_per_class=10, seed=0)
        counts = np.bincount(labels[chosen], minlength=5)
        assert counts[0] == 10
        assert counts[3] == 10
        # A class with fewer chunks than the cap contributes everything it has.
        assert counts[4] == 5

    def test_is_deterministic(self):
        labels = np.repeat([0, 3], 50)
        ids = np.arange(100)
        first = balanced_sample(labels, ids, 10, seed=5)
        second = balanced_sample(labels, ids, 10, seed=5)
        np.testing.assert_array_equal(first, second)

    def test_returns_sorted_positions_without_replacement(self):
        labels = np.repeat([0, 3, 4], 30)
        chosen = balanced_sample(labels, np.arange(90), 12, seed=1)
        assert len(set(chosen.tolist())) == len(chosen)
        assert list(chosen) == sorted(chosen)

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="labels for"):
            balanced_sample(np.zeros(5, int), np.arange(3), 2)

    def test_class_counts_are_readable(self):
        counts = class_counts(np.array([0, 0, 3, 4]), list(REGIONS))
        assert counts == {"CA1": 2, "DG": 1, "VIS": 1}


class TestSchedule:
    def test_warmup_rises_then_decays(self):
        total, warmup = 100, 10
        assert _linear_warmup_decay(0, total, warmup) == 0.0
        assert _linear_warmup_decay(5, total, warmup) == pytest.approx(0.5)
        assert _linear_warmup_decay(10, total, warmup) == pytest.approx(1.0)
        assert _linear_warmup_decay(55, total, warmup) == pytest.approx(0.5)
        assert _linear_warmup_decay(100, total, warmup) == pytest.approx(0.0)

    def test_never_goes_negative_past_the_end(self):
        assert _linear_warmup_decay(200, 100, 10) == 0.0


class TestConfig:
    def test_effective_batch_matches_the_paper(self):
        config = FineTuneConfig()
        assert config.effective_batch == 32
        assert config.learning_rate == 3e-5
        assert config.warmup_ratio == 0.1
        assert config.epochs == 10


@pytest.mark.slow
class TestModel:
    """Needs the audio checkpoint; skipped when it is not already cached."""

    @staticmethod
    def _build():
        from lfpaudit.models.lfp2vec_lite import build_model

        try:
            return build_model()
        except Exception as error:  # noqa: BLE001 - offline or uncached
            pytest.skip(f"audio checkpoint unavailable: {error}")

    def test_head_matches_the_described_architecture(self):
        model, info = self._build()
        # Project to 256, mean-pool, classify: what the paper describes.
        assert model.projector.out_features == 256
        assert model.classifier.out_features == len(REGIONS)
        assert info.num_labels == len(REGIONS)

    def test_feature_encoder_is_frozen_by_default(self):
        model, info = self._build()
        assert info.frozen_feature_encoder
        conv = model.wav2vec2.feature_extractor
        assert not any(p.requires_grad for p in conv.parameters())
        assert info.trainable_parameters < info.total_parameters

    def test_pooled_embeddings_have_the_head_width(self):
        from lfpaudit.models.lfp2vec_lite import pooled_embeddings

        model, _ = self._build()
        model.eval()
        pooled = pooled_embeddings(model, torch.zeros(2, 16000))
        assert pooled.shape == (2, 256)

    def test_smoke_drives_the_loss_down(self, tmp_path):
        from lfpaudit.models.train import smoke

        result = smoke(device="cpu", steps=12)
        assert result["last_loss"] < result["first_loss"]
        json.dumps(result)  # must be serialisable into a manifest
