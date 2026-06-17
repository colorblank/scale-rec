from __future__ import annotations

"""Compatibility re-export for structured model outputs."""

from ..core.model_output import (  # noqa: F401
    BINARY_LOGIT,
    PROBABILITY,
    REGRESSION,
    SCORE,
    ModelOutput,
    OutputKind,
    OutputTensor,
    ensure_model_output,
)
