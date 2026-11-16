"""Checks that the one-step-at-a-time implementations in bench_overhead.py give the same alert onsets as the frozen
vectorized code (core.filter_logodds + core.run_policy). Usage: python exploratory/test_streaming.py"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evialert import core
import bench_overhead as B

ok = True
for seed in range(5):
    rng = np.random.default_rng(seed); T = 3000
    P = rng.dirichlet(np.ones(4) * 0.5, size=T); U = rng.random(T) > 0.2
    Tt, prior, a, b, g = 1.8, 0.5, 0.0025, 0.05, 1.0
    ell = core.evidence(core.temperature_scale(P, Tt), [0, 1], prior, clip=None)[None, :]
    for th in (1.0, 3.0, 6.0):
        ref, _ = core.run_policy(core.filter_logodds(ell, U[None, :], a, b, g), U[None, :], th, accumulate=False)
        s = B.FilterStream(Tt, prior, a, b, g, th)
        ok &= np.array_equal(ref[0, 0], np.array([s.step(int(u), p.tolist()) for u, p in zip(U, P)], bool))
    for h in (4.0, 8.0, 16.0):
        ref, _ = core.run_policy(ell, U[None, :], h, accumulate=True)
        s = B.CusumStream(Tt, prior, a, b, g, h)
        ok &= np.array_equal(ref[0, 0], np.array([s.step(int(u), p.tolist()) for u, p in zip(U, P)], bool))
print('streaming implementations identical to frozen code:', bool(ok))
assert ok
