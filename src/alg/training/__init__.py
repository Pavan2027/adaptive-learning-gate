"""
Training loops and baseline trainers for the Adaptive Learning Gate.

Exports:
- BaselineTrainer: Conventional update-on-every-sample trainer.
- save_checkpoint: Checkpoint saving helper.
- load_checkpoint: Checkpoint restoring helper.
"""

from alg.training.baseline import BaselineTrainer, load_checkpoint, save_checkpoint
from alg.training.stream_trainer import StreamMode, StreamStepRecord, StreamTrainer, StreamTrainingResult

__all__ = [
    "BaselineTrainer",
    "StreamTrainer",
    "StreamMode",
    "StreamStepRecord",
    "StreamTrainingResult",
    "save_checkpoint",
    "load_checkpoint",
]
