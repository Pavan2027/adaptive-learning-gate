"""
Unit tests for alg.training.stream_trainer (Phase 6 Continual Training).

Covers:
- Verifying IGNORE action causes zero model parameter changes.
- Verifying LEARN action updates model parameters.
- Verifying Conventional mode updates unconditionally on every batch.
- Verifying Defer Queue re-evaluation respects max_defer_retries (no infinite loops).
- End-to-end smoke test on a continuous stream.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from alg.data.stream import SampleMetadata, StreamSample, StreamSimulator
from alg.gating import (
    AdaptiveLearningGate,
    GateAction,
    GateDecision,
    ReasonCode,
    RuleBasedGatePolicy,
)
from alg.models.mlp import ModularMLP
from alg.signals import PrototypeMemory, SignalExtractor
from alg.signals.types import SignalScores
from alg.training.stream_trainer import StreamMode, StreamTrainer


class DummyPolicyAlwaysAction:
    """Mock policy returning a pre-fixed action for deterministic testing."""

    def __init__(self, action: GateAction) -> None:
        self.action = action

    def evaluate(self, signals: SignalScores, **kwargs: Any) -> GateDecision:
        return GateDecision(
            sample_id="dummy",
            stream_index=0,
            action=self.action,
            utility=0.9 if self.action == GateAction.LEARN else 0.1,
            benefit=0.5,
            harm=0.5,
            reason=ReasonCode.HIGH_UTILITY if self.action == GateAction.LEARN else ReasonCode.LOW_RELIABILITY,
            reason_detail="Mock test decision",
            signals=signals,
        )


def _create_mock_simulator(
    num_samples: int = 16,
    in_features: int = 10,
    num_classes: int = 2,
) -> StreamSimulator:
    """Construct a lightweight mock StreamSimulator with fake tensors."""
    class DummyDataset:
        def __len__(self) -> int:
            return num_samples

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return torch.randn(in_features), idx % num_classes

    base_ds = DummyDataset()
    sim = StreamSimulator(
        base_dataset=base_ds,
        stream_size=num_samples,
        label_noise_rate=0.25,
        feature_corruption_rate=0.0,
        num_classes=num_classes,
        seed=42,
    )
    return sim


class TestStreamTrainerUpdateSemantics:
    def test_ignore_causes_zero_parameter_changes(self) -> None:
        """Verify that when the gate chooses IGNORE, no backward pass or weight update occurs."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        initial_params = [p.clone().detach() for p in model.parameters()]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gate = AdaptiveLearningGate(policy=DummyPolicyAlwaysAction(GateAction.IGNORE))

        memory = PrototypeMemory(num_classes=2, feature_dim=5, device="cpu")
        extractor = SignalExtractor(model=model, prototype_memory=memory, device="cpu")

        trainer = StreamTrainer(
            model=model,
            optimizer=optimizer,
            gate=gate,
            signal_extractor=extractor,
            device="cpu",
            use_amp=False,
        )

        sim = _create_mock_simulator(num_samples=8, in_features=10)
        res = trainer.train_stream(simulator=sim, batch_size=4, mode=StreamMode.GATED)

        assert res.total_update_steps == 0
        assert res.samples_learned == 0
        assert res.samples_ignored == 8

        # Verify all parameters remain strictly unchanged
        for p_before, p_after in zip(initial_params, model.parameters()):
            assert torch.equal(p_before, p_after)

    def test_learn_causes_parameter_changes(self) -> None:
        """Verify that when the gate chooses LEARN, optimizer steps execute and weights update."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        initial_params = [p.clone().detach() for p in model.parameters()]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        gate = AdaptiveLearningGate(policy=DummyPolicyAlwaysAction(GateAction.LEARN))

        memory = PrototypeMemory(num_classes=2, feature_dim=5, device="cpu")
        extractor = SignalExtractor(model=model, prototype_memory=memory, device="cpu")

        trainer = StreamTrainer(
            model=model,
            optimizer=optimizer,
            gate=gate,
            signal_extractor=extractor,
            device="cpu",
            use_amp=False,
        )

        sim = _create_mock_simulator(num_samples=8, in_features=10)
        res = trainer.train_stream(simulator=sim, batch_size=4, mode=StreamMode.GATED)

        assert res.total_update_steps > 0
        assert res.samples_learned == 8
        assert res.samples_ignored == 0

        # Verify at least one parameter changed
        weights_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(initial_params, model.parameters())
        )
        assert weights_changed

    def test_conventional_mode_updates_every_batch(self) -> None:
        """Verify conventional continual mode runs gradient steps on every batch without gating."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = StreamTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            use_amp=False,
        )

        sim = _create_mock_simulator(num_samples=16, in_features=10)
        res = trainer.train_stream(simulator=sim, batch_size=4, mode=StreamMode.CONVENTIONAL)

        assert res.mode == StreamMode.CONVENTIONAL
        assert res.total_samples == 16
        assert res.total_update_steps == 4  # 16 samples / batch 4 = 4 update steps
        assert res.samples_learned == 16
        assert res.compute_savings_pct == 0.0


class TestDeferQueueRecheckSemantics:
    def test_max_defer_retries_prevents_infinite_loop(self) -> None:
        """Verify that samples remaining in DEFER across multiple rechecks are retired after max_retries."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Policy always defers
        gate = AdaptiveLearningGate(
            policy=DummyPolicyAlwaysAction(GateAction.DEFER),
            defer_queue_capacity=50,
        )
        memory = PrototypeMemory(num_classes=2, feature_dim=5, device="cpu")
        extractor = SignalExtractor(model=model, prototype_memory=memory, device="cpu")

        trainer = StreamTrainer(
            model=model,
            optimizer=optimizer,
            gate=gate,
            signal_extractor=extractor,
            device="cpu",
            use_amp=False,
            recheck_interval=1,     # Recheck every batch
            recheck_batch_size=8,
            max_defer_retries=1,    # Allow at most 1 requeue
        )

        sim = _create_mock_simulator(num_samples=16, in_features=10)
        res = trainer.train_stream(simulator=sim, batch_size=4, mode=StreamMode.GATED)

        # Since policy always defers, after 1 retry they must be retired (ignored)
        assert res.samples_deferred == 16
        assert res.rechecked_ignored > 0
        assert res.total_update_steps == 0
        # Defer queue should not blow up indefinitely
        assert len(gate.defer_queue) <= 16


class TestStreamSmokeTest:
    def test_end_to_end_stream_execution(self) -> None:
        """End-to-end smoke test on small synthetic MLP stream with accuracy evaluation."""
        torch.manual_seed(42)
        model = ModularMLP(input_dim=10, hidden_dims=[16], num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        policy = RuleBasedGatePolicy(learn_threshold=0.50, ignore_threshold=0.30)
        gate = AdaptiveLearningGate(policy=policy, defer_queue_capacity=20)
        memory = PrototypeMemory(num_classes=2, feature_dim=16, device="cpu")
        extractor = SignalExtractor(model=model, prototype_memory=memory, device="cpu")

        trainer = StreamTrainer(
            model=model,
            optimizer=optimizer,
            gate=gate,
            signal_extractor=extractor,
            prototype_memory=memory,
            device="cpu",
            use_amp=False,
            recheck_interval=2,
            recheck_batch_size=4,
        )

        sim = _create_mock_simulator(num_samples=20, in_features=10)
        eval_ds = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
        eval_loader = DataLoader(eval_ds, batch_size=5)

        res = trainer.train_stream(
            simulator=sim,
            batch_size=4,
            mode=StreamMode.GATED,
            eval_loader=eval_loader,
            eval_interval=10,
        )

        assert res.total_samples == 20
        assert res.total_update_steps >= 0
        assert 0.0 <= res.final_test_accuracy <= 1.0
        assert len(res.accuracy_trajectory) >= 1
