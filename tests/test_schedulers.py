import numpy as np
import pytest
from numpycnn import LRScheduler, CosineAnnealingLR, WarmupScheduler, ExponentialLR


class TestLRScheduler:
    def test_patient_reduces_on_plateau(self):
        s = LRScheduler(initial_lr=0.01, lr_decay_factor=0.5, patience=2, patient=True)
        for e in range(5):
            s(e, val_loss=1.0)
        assert s.lr < 0.01

    def test_step_reduces_at_interval(self):
        s = LRScheduler(initial_lr=0.01, lr_decay_factor=0.5, step_size=2, patient=False)
        s(0, val_loss=1.0)
        s(1, val_loss=1.0)
        assert s.lr < 0.01

    def test_min_lr(self):
        s = LRScheduler(initial_lr=0.01, lr_decay_factor=0.01, patience=0, min_lr=1e-5, patient=True)
        for e in range(100):
            s(e, val_loss=float('inf'))
        assert s.lr >= 1e-5


class TestCosineAnnealingLR:
    def test_starts_at_initial(self):
        s = CosineAnnealingLR(initial_lr=0.01, T_max=20)
        assert s(0) == pytest.approx(0.01, abs=1e-6)

    def test_reaches_min_at_end(self):
        s = CosineAnnealingLR(initial_lr=0.01, T_max=20, min_lr=0.0)
        lr = s(20)
        assert lr == pytest.approx(0.0, abs=1e-6)

    def test_monotonically_decreasing(self):
        s = CosineAnnealingLR(initial_lr=0.01, T_max=20)
        lrs = [s(e) for e in range(21)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


class TestWarmupScheduler:
    def test_starts_low(self):
        base = CosineAnnealingLR(initial_lr=0.01, T_max=20)
        s = WarmupScheduler(base, warmup_epochs=5, warmup_start_lr=1e-6)
        assert s(0) < 0.001

    def test_reaches_initial_after_warmup(self):
        base = CosineAnnealingLR(initial_lr=0.01, T_max=20)
        s = WarmupScheduler(base, warmup_epochs=5, warmup_start_lr=0.0)
        lr = s(5)
        assert lr == pytest.approx(0.01, abs=1e-3)


class TestExponentialLR:
    def test_decays(self):
        s = ExponentialLR(initial_lr=0.01, decay_rate=0.9)
        lrs = [s(e) for e in range(10)]
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1]

    def test_min_lr(self):
        s = ExponentialLR(initial_lr=0.01, decay_rate=0.5, min_lr=0.001)
        for e in range(100):
            s(e)
        assert s.lr >= 0.001
