"""EviAlert as a reusable software component.

This module wraps the frozen research code (``core.py``) behind a small, documented interface:

* ``fit_frame_calibration`` and ``fit_hazards`` estimate the parameters from calibration data;
* ``certify`` verifies the per-person false-alert budget and returns a ``Certificate``;
* ``ReleaseGate`` turns a certificate into a release decision;
* ``AlertStream`` runs the online part for one monitored person, one step at a time;
* ``replay`` runs a configuration over recorded streams (vectorized, identical onsets).

It adds no new statistics: every number is computed by the functions in ``core.py``, which are unchanged
(their SHA-256 fingerprints are listed in PREREGISTRATION.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from . import core

NO_ALERT, WATCHLIST, HIGH_ALERT, INSUFFICIENT_EVIDENCE = 'No Alert', 'Watchlist', 'High Alert', 'Insufficient Evidence'
ACCUMULATORS = ('filter', 'cusum')


@dataclass(frozen=True)
class AlertConfig:
    """Everything the online part needs. ``threshold`` is theta (filter, log-odds) or h (CUSUM)."""
    target: tuple                     # indices of the target classes D
    temperature: float                # T*
    prior_cal: float                  # share of target frames in the calibration data
    threshold: float
    accumulator: str = 'filter'       # 'filter' (tempered two-state filter) or 'cusum' (CUSUM e-detector)
    a: float = 0.0025                 # onset hazard (filter only)
    b: float = 0.05                   # exit hazard (filter only)
    gamma: float = 1.0                # tempering exponent (filter only)
    window: int = 6                   # W, steps
    rho: float = 0.5                  # sufficiency level
    refractory: int = 24              # R, steps
    watch_gap: float = 2.0            # Watchlist below the High Alert threshold (filter only)

    def __post_init__(self):
        if self.accumulator not in ACCUMULATORS:
            raise ValueError(f'accumulator must be one of {ACCUMULATORS}')
        if not 0 < self.prior_cal < 1:
            raise ValueError('prior_cal must lie in (0, 1)')
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        if self.accumulator == 'filter' and not (0 < self.a < 1 and 0 < self.b < 1 and 0 < self.gamma <= 1):
            raise ValueError('filter needs 0 < a, b < 1 and 0 < gamma <= 1')
        if self.window < 1 or not 0 <= self.rho <= 1 or self.refractory < 0:
            raise ValueError('invalid window, rho or refractory')


@dataclass(frozen=True)
class Certificate:
    """Result of person-level certification (Proposition 2)."""
    threshold: float
    certified: bool
    n_persons: int
    n_min: int
    budget: float
    alpha: float
    delta: float

    def statement(self) -> str:
        if not self.certified:
            return (f'NOT CERTIFIABLE: {self.n_persons} calibration persons; at least {self.n_min} are needed to promise '
                    f'that at most {self.alpha:.0%} of new persons exceed {self.budget:g} false alerts per hour '
                    f'(confidence {1 - self.delta:.0%}), or no threshold met the budget.')
        return (f'CERTIFIED: with confidence {1 - self.delta:.0%}, at most {self.alpha:.0%} of new persons exceed '
                f'{self.budget:g} false alerts per hour at threshold {self.threshold:g} '
                f'({self.n_persons} calibration persons).')


def fit_frame_calibration(P_cal: np.ndarray, y_cal: np.ndarray, target: Sequence[int]) -> tuple[float, float]:
    """Temperature T* (NLL on calibration frames) and the calibration prior of the target classes."""
    T = core.fit_temperature(np.asarray(P_cal, float), np.asarray(y_cal, int))
    prior = float(np.isin(np.asarray(y_cal), list(target)).mean())
    return float(T), min(max(prior, 1e-6), 1 - 1e-6)


def fit_hazards(state_seqs: Iterable[Sequence[bool]]) -> tuple[float, float]:
    """Onset and exit hazards of alert-worthy episodes from calibration persons' target-state sequences."""
    return core.fit_transitions(list(state_seqs))


def certify(thresholds: Sequence[float], person_rates: np.ndarray, budget: float = 1.0,
            alpha: float = 0.2, delta: float = 0.1, higher_is_conservative: bool = True) -> Certificate:
    """Certify a threshold. ``person_rates[j, i]`` is the false-alert rate per hour of calibration person i at
    ``thresholds[j]``. Thresholds are tested in fixed sequence from the most conservative one (the highest, unless
    ``higher_is_conservative`` is False)."""
    thr = np.asarray(thresholds, float); R = np.asarray(person_rates, float)
    if R.ndim != 2 or R.shape[0] != len(thr):
        raise ValueError('person_rates must have shape (len(thresholds), n_persons)')
    if not 0 < alpha < 1 or not 0 < delta < 1 or budget < 0:
        raise ValueError('invalid alpha, delta or budget')
    sign = 1.0 if higher_is_conservative else -1.0
    h, ok, n_min = core.certify_threshold(sign * thr, R, budget, alpha=alpha, delta=delta)
    return Certificate(float(sign * h), bool(ok), int(R.shape[1]), int(n_min), float(budget), float(alpha), float(delta))


@dataclass
class ReleaseGate:
    """Blocks a release whose false-alert budget is not certified."""
    log: list = field(default_factory=list)

    def check(self, certificate: Certificate) -> bool:
        self.log.append(certificate.statement())
        return certificate.certified

    def enforce(self, certificate: Certificate) -> None:
        if not self.check(certificate):
            raise RuntimeError(certificate.statement())


class AlertStream:
    """Online part for one monitored person (Algorithm 1). Constant memory; one call per step."""

    def __init__(self, config: AlertConfig):
        self.c = config
        self._lp = math.log(config.prior_cal / (1 - config.prior_cal))
        self._lam = config.a / (config.a + config.b)
        self._S = 0.0
        self._win = [0] * config.window; self._i = 0; self._n = 0; self._r = 0
        self.score = 0.0

    def _evidence(self, probs: Sequence[float]) -> float:
        z = [math.log(min(max(float(p), 1e-12), 1.0)) / self.c.temperature for p in probs]   # as core.temperature_scale
        m = max(z); e = [math.exp(v - m) for v in z]
        q = sum(e[k] for k in self.c.target) / sum(e)
        q = min(max(q, 1e-6), 1 - 1e-6)
        return math.log(q / (1 - q)) - self._lp

    def step(self, usable: bool, probs: Sequence[float] | None = None) -> tuple[str, bool]:
        """Process one step. ``probs`` is the classifier's probability vector (ignored when not usable).
        Returns (status, onset); onset is True when a High Alert is pushed to the recipient."""
        c = self.c; u = 1 if usable else 0
        if u and probs is None:
            raise ValueError('a usable step needs a probability vector')
        if c.accumulator == 'filter':
            lam_bar = self._lam * (1 - c.b) + (1 - self._lam) * c.a
            L = math.log(lam_bar / (1 - lam_bar))
            if u:
                L += c.gamma * self._evidence(probs)
            L = min(max(L, -50.0), 50.0)
            self._lam = 1 / (1 + math.exp(-L)); score = L
        else:
            if u:
                self._S = max(0.0, self._S + self._evidence(probs))
            score = self._S
        self.score = score
        self._n += u - self._win[self._i]; self._win[self._i] = u; self._i = (self._i + 1) % c.window
        if self._n / c.window < c.rho:
            onset, status = False, INSUFFICIENT_EVIDENCE
        else:
            onset = score >= c.threshold and self._r == 0
            if score >= c.threshold or self._r > 0:
                status = HIGH_ALERT
            elif c.accumulator == 'filter' and score >= c.threshold - c.watch_gap:
                status = WATCHLIST
            else:
                status = NO_ALERT
        if onset and c.accumulator == 'cusum':
            self._S = 0.0
        self._r = c.refractory if onset else max(self._r - 1, 0)
        return status, bool(onset)


def replay(config: AlertConfig, P: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Alert onsets (n_streams, T) of a configuration over recorded streams, computed with the frozen vectorized
    code. ``P`` has shape (n_streams, T, K) and ``usable`` (n_streams, T)."""
    P = np.asarray(P, float); U = np.asarray(usable, bool)
    if P.ndim != 3 or U.shape != P.shape[:2]:
        raise ValueError('P must have shape (n_streams, T, K) and usable (n_streams, T)')
    ell = core.evidence(core.temperature_scale(P, config.temperature), list(config.target), config.prior_cal, clip=None)
    if config.accumulator == 'filter':
        L = core.filter_logodds(ell, U, config.a, config.b, config.gamma)
        on, _ = core.run_policy(L, U, config.threshold, window=config.window, rho=config.rho,
                                refractory=config.refractory, accumulate=False)
    else:
        on, _ = core.run_policy(ell, U, config.threshold, window=config.window, rho=config.rho,
                                refractory=config.refractory, accumulate=True)
    return on[0]
