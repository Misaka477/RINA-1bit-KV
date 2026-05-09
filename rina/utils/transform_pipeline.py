"""Stub: TransformPipeline removed.  Only resolve_transform_mode remains."""

from __future__ import annotations


class TransformContext:
    """Empty placeholder — transform pipeline removed."""
    mode: str = "none"
    tile_size: int = 0
    transform_decisions: list = None

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "none")
        self.tile_size = kwargs.get("tile_size", 0)
        self.transform_decisions = kwargs.get("transform_decisions", None)


class TransformPipeline:
    """Empty placeholder — transform pipeline removed."""
    mode = "none"
    tile_size = 0
    _context = TransformContext()

    def __init__(self, **kwargs):
        pass


def resolve_transform_mode(mode):
    """Always returns 'none' — all transforms removed."""
    return "none"
