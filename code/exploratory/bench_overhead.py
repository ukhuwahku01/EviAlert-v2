"""Runtime overhead of the alerting layer (added after the confirmatory test): per-step cost of the online part for one
stream processed one step at a time (as in deployment), for the tempered filter and the CUSUM accumulator, and the
offline cost of person-level certification. Usage: python exploratory/bench_overhead.py [out_json]"""
import json, math, os, platform, sys, time, tracemalloc
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from evialert import core


class FilterStream:
    """Online part of Algorithm 1 for one stream (tempered filter, sufficiency gate, refractory period)."""
    def __init__(self, T, prior, a, b, gamma, theta, W=6, rho=0.5, R=24):
        self.T, self.lp, self.a, self.b, self.g, self.th = T, math.log(prior / (1 - prior)), a, b, gamma, theta
        self.W, self.rho, self.R = W, rho, R
        self.lam = a / (a + b); self.win = [0] * W; self.i = 0; self.n = 0; self.r = 0

    def step(self, u, p):
        lam_bar = self.lam * (1 - self.b) + (1 - self.lam) * self.a
        L = math.log(lam_bar / (1 - lam_bar))
        if u:
            z = [math.log(x) / self.T for x in p]; m = max(z); e = [math.exp(v - m) for v in z]
            q = min(max((e[0] + e[1]) / sum(e), 1e-6), 1 - 1e-6)          # target = first two classes
            L += self.g * (math.log(q / (1 - q)) - self.lp)
        L = min(max(L, -50.0), 50.0)
        self.lam = 1 / (1 + math.exp(-L))
        self.n += u - self.win[self.i]; self.win[self.i] = u; self.i = (self.i + 1) % self.W
        if self.n / self.W < self.rho:
            onset = 0
        else:
            onset = 1 if (L >= self.th and self.r == 0) else 0
        self.r = self.R if onset else max(self.r - 1, 0)
        return onset


class CusumStream(FilterStream):
    """Same layer with the CUSUM e-detector accumulator (Eq. 5): no decay, held on unusable frames, reset after onset."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); self.S = 0.0

    def step(self, u, p):
        if u:
            z = [math.log(x) / self.T for x in p]; m = max(z); e = [math.exp(v - m) for v in z]
            q = min(max((e[0] + e[1]) / sum(e), 1e-6), 1 - 1e-6)
            self.S = max(0.0, self.S + math.log(q / (1 - q)) - self.lp)
        self.n += u - self.win[self.i]; self.win[self.i] = u; self.i = (self.i + 1) % self.W
        onset = 0
        if self.n / self.W >= self.rho and self.S >= self.th and self.r == 0:
            onset = 1; self.S = 0.0
        self.r = self.R if onset else max(self.r - 1, 0)
        return onset


def bench_stream(cls, steps=200_000, reps=5, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(4), size=steps).tolist(); U = (rng.random(steps) > 0.1).astype(int).tolist()
    best = float('inf')
    for _ in range(reps):
        s = cls(1.8, 0.5, 0.0025, 0.05, 1.0, math.log(720))
        t0 = time.perf_counter()
        for u, p in zip(U, P):
            s.step(u, p)
        best = min(best, (time.perf_counter() - t0) / steps)
    tracemalloc.start(); s = cls(1.8, 0.5, 0.0025, 0.05, 1.0, math.log(720)); s.step(1, P[0])
    mem = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return best * 1e6, mem


def bench_cert(n_persons=14, k=150, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    thr = np.linspace(-6, 18, k); rates = np.sort(rng.gamma(2.0, 0.6, size=(k, n_persons)), axis=0)[::-1]
    t0 = time.perf_counter()
    for _ in range(reps):
        core.certify_threshold(thr, rates, 1.0)
    return (time.perf_counter() - t0) / reps * 1e3


if __name__ == '__main__':
    out = {'cpu': platform.processor() or platform.machine(), 'python': platform.python_version()}
    try:
        out['cpu_model'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    except Exception:
        pass
    out['filter_us_per_step'], out['filter_peak_bytes'] = bench_stream(FilterStream)
    out['cusum_us_per_step'], out['cusum_peak_bytes'] = bench_stream(CusumStream)
    out['certification_ms'] = bench_cert()
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], 'w'), indent=1)
