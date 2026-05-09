"""Stub: FWHT removed (experimentally destructive, §8.1.11)."""

from __future__ import annotations

import torch


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Identity — FWHT transform removed."""
    return x


def ifwht(x: torch.Tensor) -> torch.Tensor:
    """Identity — FWHT inverse transform removed."""
    return x
