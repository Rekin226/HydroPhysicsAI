"""Models: the contestants benchmarked against the gray-box baseline.

All models implement the :class:`GroundwaterModel` interface (see base.py): ``fit`` on
the calibration period, then ``simulate`` a free-running hindcast over the full record
so they are scored in the SAME simulation mode as the gray-box.

  - gru.py  GlobalGRU      reference deep-learning model (pure PyTorch). Working
                          template: trains across all wells, runs on cuda/mps/cpu.
  - ude.py  PhysicsUDE     physics-informed Universal Differential Equation skeleton
                          (parameter hypernetwork + differentiable ODE). The core new
                          method; TODO markers show what to implement on the GPU.

PhysicsNeMo (NVIDIA): the UDE/operator port targets PhysicsNeMo for GPU training and
held-out-well generalization. See the "NVIDIA GPU path" section in the README.

torch is an optional dependency; importing these classes without torch raises a clear
error. The foundation (data/metrics/baselines/eval) does not need torch.
"""

from .base import GroundwaterModel

__all__ = ["GroundwaterModel"]
