"""Unit tests for the EviAlert component interface (evialert/layer.py).
Run from the code folder: python -m unittest discover -s tests -v   (standard library only; pytest also collects them)"""
import math
import unittest
import numpy as np
from scipy import stats
from evialert.layer import (AlertConfig, AlertStream, Certificate, ReleaseGate, certify, fit_frame_calibration,
                            fit_hazards, replay, HIGH_ALERT, INSUFFICIENT_EVIDENCE, NO_ALERT, WATCHLIST)


def _streams(seed, n=3, T=2000, K=4, p_unusable=0.2, conc=0.5):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(K) * conc, size=(n, T)), rng.random((n, T)) > p_unusable


def _run(cfg, P, U):
    on = np.zeros(U.shape, bool); status = np.empty(U.shape, object); scores = np.zeros(U.shape)
    for i in range(U.shape[0]):
        s = AlertStream(cfg)
        for t in range(U.shape[1]):
            status[i, t], on[i, t] = s.step(bool(U[i, t]), P[i, t])
            scores[i, t] = s.score
    return on, status, scores


class TestAlertStream(unittest.TestCase):
    def test_stream_matches_frozen_vectorized_code(self):
        for acc, thr in (('filter', 1.0), ('filter', 3.0), ('filter', 6.0), ('cusum', 4.0), ('cusum', 12.0)):
            for seed in range(3):
                with self.subTest(acc=acc, thr=thr, seed=seed):
                    P, U = _streams(seed)
                    cfg = AlertConfig(target=(0, 1), temperature=1.7, prior_cal=0.45, threshold=thr, accumulator=acc)
                    on, _, _ = _run(cfg, P, U)
                    self.assertTrue(np.array_equal(on, replay(cfg, P, U)))

    def test_proposition1_no_new_alert_without_new_evidence(self):
        for seed in range(5):
            P, U = _streams(seed, p_unusable=0.4)
            cfg = AlertConfig(target=(0, 1), temperature=1.0, prior_cal=0.5, threshold=2.0, a=0.01, b=0.05)
            self.assertGreater(cfg.threshold, math.log(cfg.a / cfg.b))
            _, _, sc = _run(cfg, P, U)
            prev = np.concatenate([np.full((sc.shape[0], 1), math.log(cfg.a / cfg.b)), sc[:, :-1]], 1)
            crossing = (prev < cfg.threshold) & (sc >= cfg.threshold)
            self.assertFalse(np.any(crossing & ~U))

    def test_insufficient_evidence_blocks_decisions(self):
        P, U = _streams(3, p_unusable=0.6)
        cfg = AlertConfig(target=(0, 1), temperature=1.0, prior_cal=0.5, threshold=-10.0)   # would fire whenever allowed
        on, st, _ = _run(cfg, P, U)
        insuff = st == INSUFFICIENT_EVIDENCE
        self.assertTrue(insuff.any())
        self.assertFalse(np.any(on & insuff))
        for i in range(U.shape[0]):
            for t in range(cfg.window - 1, U.shape[1]):
                frac = U[i, t - cfg.window + 1:t + 1].sum() / cfg.window
                self.assertEqual(frac < cfg.rho, insuff[i, t])

    def test_refractory_period_separates_onsets(self):
        P, U = _streams(4, p_unusable=0.0)
        for acc, thr in (('filter', -10.0), ('cusum', 0.0)):
            cfg = AlertConfig(target=(0, 1), temperature=1.0, prior_cal=0.5, threshold=thr, accumulator=acc)
            on, _, _ = _run(cfg, P, U)
            for i in range(on.shape[0]):
                idx = np.flatnonzero(on[i])
                self.assertGreater(len(idx), 1)
                self.assertTrue(np.all(np.diff(idx) >= cfg.refractory))

    def test_statuses_are_tiered(self):
        P, U = _streams(5, p_unusable=0.05)
        cfg = AlertConfig(target=(0, 1), temperature=1.0, prior_cal=0.5, threshold=4.0)
        _, st, sc = _run(cfg, P, U)
        self.assertTrue({NO_ALERT, WATCHLIST, HIGH_ALERT} <= set(st.ravel()))
        w = st == WATCHLIST
        self.assertTrue(np.all((sc[w] >= cfg.threshold - cfg.watch_gap) & (sc[w] < cfg.threshold)))

    def test_usable_step_needs_probabilities(self):
        s = AlertStream(AlertConfig(target=(0,), temperature=1.0, prior_cal=0.5, threshold=1.0))
        with self.assertRaises(ValueError):
            s.step(True, None)
        self.assertEqual(s.step(False)[0], INSUFFICIENT_EVIDENCE)

    def test_replay_rejects_bad_shapes(self):
        cfg = AlertConfig(target=(0,), temperature=1.0, prior_cal=0.5, threshold=1.0)
        with self.assertRaises(ValueError):
            replay(cfg, np.ones((10, 4)), np.ones(10, bool))


class TestCertification(unittest.TestCase):
    def test_needs_enough_persons(self):
        thr = np.linspace(-6, 18, 50)
        cert = certify(thr, np.zeros((50, 10)), budget=1.0, alpha=0.2, delta=0.1)   # nobody exceeds the budget ...
        self.assertEqual(cert.n_min, 11)
        self.assertFalse(cert.certified)                                            # ... but 10 persons cannot certify
        self.assertEqual(cert.threshold, thr.max())                                 # most conservative threshold
        cert = certify(thr, np.zeros((50, 11)), budget=1.0)
        self.assertTrue(cert.certified)
        self.assertEqual(cert.threshold, thr.min())

    def test_fixed_sequence_with_binomial_p_values(self):
        thr = np.arange(10.0); n = 30; rates = np.zeros((10, n))
        for j in range(10):                          # lower thresholds: more persons above the budget
            rates[j, : (9 - j) * 2] = 2.0
        cert = certify(thr, rates, budget=1.0, alpha=0.2, delta=0.1)
        p = stats.binom.cdf((rates > 1.0).sum(1), n, 0.2)
        last = 9
        for j in range(9, -1, -1):
            if p[j] > 0.1:
                break
            last = j
        self.assertTrue(cert.certified)
        self.assertEqual(cert.threshold, thr[last])

    def test_lower_is_conservative_option(self):
        thr = np.arange(10.0); rates = np.tile(np.linspace(0, 3, 10)[:, None], (1, 20))   # higher thr -> more alerts
        cert = certify(thr, rates, budget=1.0, higher_is_conservative=False)
        self.assertTrue(cert.certified)
        self.assertTrue(np.all(rates[thr <= cert.threshold] <= 1.0))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            certify([1, 2], np.zeros((3, 5)))
        with self.assertRaises(ValueError):
            certify([1, 2], np.zeros((2, 5)), alpha=1.5)

    def test_release_gate(self):
        gate = ReleaseGate()
        good = Certificate(3.0, True, 14, 11, 1.0, 0.2, 0.1); bad = Certificate(18.0, False, 7, 11, 1.0, 0.2, 0.1)
        self.assertTrue(gate.check(good))
        self.assertFalse(gate.check(bad))
        with self.assertRaisesRegex(RuntimeError, 'NOT CERTIFIABLE'):
            gate.enforce(bad)
        gate.enforce(good)
        self.assertEqual(len(gate.log), 4)
        self.assertIn('CERTIFIED', good.statement())


class TestFittingAndConfig(unittest.TestCase):
    def test_fitting_helpers(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 2000); P = np.full((2000, 4), 0.02); P[np.arange(2000), y] = 0.94
        P[:400] = P[:400][:, ::-1]                                   # 20% confident errors -> overconfident
        T, prior = fit_frame_calibration(P, y, target=(0, 1))
        self.assertGreater(T, 1)
        self.assertAlmostEqual(prior, np.isin(y, [0, 1]).mean())
        a, b = fit_hazards([np.r_[np.zeros(90, bool), np.ones(10, bool), np.zeros(10, bool)] for _ in range(20)])
        self.assertTrue(0 < a < 0.05 and 0.05 < b < 0.2)

    def test_invalid_configuration_is_rejected(self):
        base = dict(target=(0, 1), temperature=1.0, prior_cal=0.5, threshold=1.0)
        for kw in (dict(accumulator='kalman'), dict(prior_cal=1.0), dict(temperature=0.0), dict(gamma=1.5), dict(window=0)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                AlertConfig(**{**base, **kw})


if __name__ == '__main__':
    unittest.main()
