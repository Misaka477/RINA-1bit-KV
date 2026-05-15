"""
ternary_v2.py — CANN + SSM with HiPPO matrix + cosine attractors.
Fix #1: HiPPO state update (LegS) instead of cumulative sum.
Fix #2: attractors = normalized input projections.
Fix #3: cosine similarity energy instead of Euclidean.
"""
import numpy as np

D = 64; V = 16; HORIZON = 64; SEED = 42
np.random.seed(SEED)

# ── HiPPO-LegS (diagonal, ZOH discretization, always stable) ──
def hippo_diag(N, dt=0.01):
    """Diagonal HiPPO with exact ZOH discretization."""
    A_disc = np.zeros(N, dtype=np.float32)  # exp(-(i+1)*dt)
    B_disc = np.zeros(N, dtype=np.float32)  # (1-exp(-(i+1)*dt))/(i+1)*B_i
    for i in range(N):
        a_cont = -(i + 1)
        A_disc[i] = np.exp(a_cont * dt)
        B_i = (2 * i + 1) ** 0.5
        B_disc[i] = (1 - np.exp(a_cont * dt)) / (-a_cont) * B_i
    return A_disc, B_disc

A_diag, B_diag = hippo_diag(D)
# Full state: first half is HiPPO, second half is leaky integration
A_full = np.ones(D, dtype=np.float32)
A_full[:D//2] = A_diag[:D//2]
A_full[D//2:] = 0.9  # leaky for extra dimensions
B_full = np.ones(D, dtype=np.float32) * 0.1
B_full[:D//2] = B_diag[:D//2]

# ── Input projections (also serve as attractor targets) ──
P = np.random.randn(V, D).astype(np.float32)
# Normalize P rows so attractors have comparable scale
P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8) * 0.5
attractors = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

# ── SSM state update with HiPPO (ZOH, always stable) ──
def write(seq):
    s = np.zeros(D, dtype=np.float32)
    for t in seq:
        s = A_full * s + B_full * P[t]
    return s

# ── Cosine energy (smooth attractor landscape) ──
tau = 0.5
def energy(s):
    cos_sim = s @ attractors.T
    w = np.exp(cos_sim / tau)
    w = w / (w.sum() + 1e-8)
    return -np.sum(w * cos_sim)

def grad_energy(s):
    cos_sim = s @ attractors.T
    w = np.exp(cos_sim / tau)
    w = w / (w.sum() + 1e-8)
    grad = -np.sum(w[:, np.newaxis] * attractors / tau, axis=0)
    return grad

def converge(s, steps=30, lr=0.1):
    for _ in range(steps):
        g = grad_energy(s)
        g_norm = np.linalg.norm(g) + 1e-8
        s = s - lr * g / g_norm * 0.1  # norm-clipped gradient
    return s

# ── Decode: closest attractor ──
def decode(s):
    sim = s @ attractors.T
    return np.argmax(sim)

def norm(x):
    return x / (np.linalg.norm(x) + 1e-8)

# ── Test ──
A, B, Q = 2, 5, 12
print(f"HiPPO-CANN: D={D} V={V} H={HORIZON}")
print(f"A={A} B={B} Q={Q}")

seq = [A] + [B] * (HORIZON - 1)
s = write(seq)
s_n = norm(s)
print(f"\nPost-write state norm: {np.linalg.norm(s):.3f}")
print(f"Nearest attractor: {decode(s_n)} (expect {B})")

c = converge(s_n + norm(P[Q]), steps=40, lr=0.5)
print(f"Converged: {decode(c)} (expect {A}) {'OK' if decode(c)==A else 'FAIL'}")

# ── Depth sweep ──
print(f"\nRecall sweep (A at depth, {HORIZON}-token sequence):")
for depth in [1, 2, 4, 8, 16, 32, 48, HORIZON-1]:
    seq = [B]*depth + [A] + [B]*(HORIZON-depth-1)
    s = write(seq)
    c = converge(norm(s) + norm(P[Q]), steps=40, lr=0.5)
    sim_a = c @ attractors[A]
    sim_b = c @ attractors[B]
    near = 'A' if sim_a > sim_b else 'B'
    print(f"  depth={depth:3d} nearest={near} sim_A={sim_a:.3f} sim_B={sim_b:.3f}")
