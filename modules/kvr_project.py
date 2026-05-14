"""
KVR project module — handles per-layer QKV + RoPE + int4/int2 pack in C++.
"""
import os, torch
from torch.utils.cpp_extension import load

_project_module = None

def _get_project_module():
    global _project_module
    if _project_module is not None:
        return _project_module
    cu_path = os.path.join(os.path.dirname(__file__), "kvr_project.cu")
    _project_module = load(name="kvr_project", sources=[cu_path], verbose=False)
    return _project_module
