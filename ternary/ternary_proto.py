"""
ternary_proto.py — CANN + SSM minimal testbed
"""
import numpy as np

D = 64; V = 16; HORIZON = 32; SEED = 42
np.random.seed(SEED)

attractors = np.random.randn(V, D).astype(np.float32)
attractors /= np.linalg.norm(attractors, axis=1, keepdims=True)
sigma2 = 0.5

def energy(s):
    diffs = s[np.newaxis, :] - attractors
    return -np.sum(np.exp(-np.sum(diffs**2, axis=1) / sigma2))

def grad_energy(s):
    diffs = s[np.newaxis, :] - attractors
    dist2 = np.sum(diffs**2, axis=1)
    weights = np.exp(-dist2 / sigma2)[:, np.newaxis]
    return np.sum(2 * (-diffs) * weights / sigma2, axis=0)

def converge(s, steps=30, lr=0.15):
    for _ in range(steps):
        s = s - lr * grad_energy(s)
    return s

P = np.random.randn(V, D).astype(np.float32) * 0.1

def norm(x):
    return x / (np.linalg.norm(x) + 1e-8)

def write(seq, lam=0.9):
    s = np.zeros(D, dtype=np.float32)
    for t in seq:
        s = lam * s + P[t]
    return s

def decode(s):
    d = np.linalg.norm(s[np.newaxis, :] - attractors, axis=1)
    return np.argmin(d)

A, B, Q = 2, 5, 12
print(f"D={D} V={V} H={HORIZON}")

seq = [A] + [B] * (HORIZON - 1)
s = write(seq)
print(f"State closest: {decode(norm(s))} (expect {B})")

c = converge(norm(s) + P[Q])
print(f"Converged: {decode(c)} (expect {A}) {'OK' if decode(c) == A else 'FAIL'}")

for depth in range(1, 11):
    seq = [B]*depth + [A] + [B]*(HORIZON-depth-1)
    s = write(seq)
    c = converge(norm(s) + P[Q])
    d_a = np.linalg.norm(c - attractors[A])
    d_b = np.linalg.norm(c - attractors[B])
    print(f"depth={depth:2d} {'OK' if d_a<d_b else 'FAIL'} d_a={d_a:.3f} d_b={d_b:.3f}")
