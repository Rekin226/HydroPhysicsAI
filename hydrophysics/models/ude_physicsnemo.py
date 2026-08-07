"""PhysicsNeMoUDE: the PhysicsUDE ported onto NVIDIA PhysicsNeMo.

This is the real NVIDIA-framework version of the flagship operator. The physics
(attribute-conditioned hypernetwork -> ODE parameters, stable semi-implicit rollout,
physics-residual collocation loss) is identical to ``PhysicsUDE`` and is reused by
subclassing it. The one thing that changes is the *trainable component*: the
hypernetwork is now a ``physicsnemo.Module`` instead of a bare ``torch.nn.Module``.

Why that matters (what PhysicsNeMo actually buys here):
  - The model carries ``ModelMetaData`` declaring its capabilities (AMP / CUDA-graph /
    bf16 readiness), which the PhysicsNeMo training utilities introspect.
  - It serializes to a single versioned ``.mdlus`` checkpoint via ``.save()`` /
    ``Module.from_checkpoint()`` -- architecture + weights together, no separate config
    needed to reload (``save_checkpoint`` / ``load_checkpoint`` below).
  - It plugs into PhysicsNeMo's ecosystem (distributed helpers, mixed precision,
    optimized layers) without rewriting the model.

The class satisfies the same ``GroundwaterModel`` interface, so ``train.py`` and the
benchmark wire it up unchanged (``--model ude_nemo``). It trains and is scored exactly
like ``PhysicsUDE`` -- the numbers are the same architecture, now running on NVIDIA's
physics-ML framework.
"""

from __future__ import annotations

from .ude import PhysicsUDE, _require_torch

try:
    import torch  # noqa: F401  (re-exported indirectly; needed for the Module subclass)
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    from physicsnemo import ModelMetaData
    from physicsnemo import Module as NeMoModule
    _HAS_PHYSICSNEMO = True
except Exception:  # pragma: no cover - optional heavy dependency
    _HAS_PHYSICSNEMO = False


def _require_physicsnemo() -> None:
    if not _HAS_PHYSICSNEMO:
        raise ImportError(
            "PhysicsNeMoUDE needs nvidia-physicsnemo. "
            "Install with `pip install nvidia-physicsnemo` (or `uv pip install ...`)."
        )


class PhysicsNeMoUDE(PhysicsUDE):
    """PhysicsUDE whose hypernetwork is a native ``physicsnemo.Module``.

    Identical physics and training to :class:`PhysicsUDE`; only the network container
    differs, so it gains PhysicsNeMo metadata + ``.mdlus`` checkpointing while producing
    the same simulation scores.
    """

    name = "physics_ude_nemo"

    def __init__(self, *args, **kwargs):
        _require_physicsnemo()
        super().__init__(*args, **kwargs)

    def _build(self, n_static: int) -> None:
        # Match the parent's output width exactly: PhysicsUDE._n_params() is
        # len(PARAM_NAMES) only when rain_memory is empty; with K memory timescales it is
        # len(PARAM_NAMES)-1+K. Hardcoding len(PARAM_NAMES) silently mis-maps (or
        # IndexErrors) the moment --model ude_nemo is combined with --rain-memory.
        self.hypernet = _HyperNetNeMo(n_static, self.hidden, self._n_params()).to(self.device)

    # --- PhysicsNeMo checkpointing -------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Serialize the trained hypernetwork to a single ``.mdlus`` file.

        PhysicsNeMo stores architecture + weights + metadata together, so the model
        reloads with :meth:`load_checkpoint` (or ``physicsnemo.Module.from_checkpoint``)
        without re-specifying any hyperparameters.
        """
        if self.hypernet is None:
            raise RuntimeError("call fit() before save_checkpoint()")
        self.hypernet.save(path)

    def load_checkpoint(self, path: str) -> PhysicsNeMoUDE:
        """Reload a hypernetwork saved by :meth:`save_checkpoint`."""
        _require_physicsnemo()
        self.hypernet = NeMoModule.from_checkpoint(path).to(self.device)
        return self


if _HAS_TORCH and _HAS_PHYSICSNEMO:
    from torch import nn

    class _HyperNetNeMo(NeMoModule):
        """Static well attributes -> ODE parameters, as a PhysicsNeMo Module.

        Same architecture as ``ude._HyperNet`` (2 hidden SiLU layers), but wrapped as a
        ``physicsnemo.Module`` so it carries capability metadata and serializes to the
        portable ``.mdlus`` checkpoint format.
        """

        def __init__(self, n_in: int, hidden: int, n_params: int):
            # Capability metadata is what the PhysicsNeMo training utilities introspect,
            # and it is opt-in: StaticCaptureTraining silently disables CUDA graphs unless
            # cuda_graphs=True is declared here. The declarations are truthful for this
            # model -- the architecture is pure Linear/SiLU (so bf16/AMP are safe) and the
            # whole training step has static shapes with no data-dependent control flow
            # and no host synchronization, which is exactly the graph-capture precondition.
            super().__init__(meta=ModelMetaData(amp=True, bf16=True, auto_grad=True,
                                                cuda_graphs=True))
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, n_params),
            )

        def forward(self, x):
            return self.net(x)
else:  # pragma: no cover - keeps the name importable for error messages / typing
    _HyperNetNeMo = None  # type: ignore[assignment]


__all__ = ["PhysicsNeMoUDE", "_require_physicsnemo"]


# Re-exported so callers can `from .ude_physicsnemo import _require_torch`-style guard.
_ = _require_torch
