"""Additional baseline (added in response to the editor's screening): the CUSUM e-detector of Shin, Ramdas & Rinaldo
(2024) on calibrated, unclipped likelihood ratios without drift, on the infant test splits (seeds 101-120).
Frozen modules are used unchanged (EviAlert-C with clip=None and kappa=0). Besides the calibrated threshold rules,
the theoretical threshold log(720) for an average run length of 720 steps (1 h) is evaluated. Usage: python exploratory/infant_edetector.py <model>"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from evialert import experiments as EX
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lopo_curves import E_GRID, VILLE_THRESHOLD
EX.GRID_CUSUM = E_GRID          # wider grid incl. log(720) exactly (frozen module patched at run time only)
U = os.environ.get('INFANTFER_RESULTS', os.path.join(ROOT, '..', 'results'))
model = sys.argv[1]
P, y, g = EX.load_oof(U + '/runs', U + '/clean_index.csv', model)
res = []; t0 = time.time(); thr = VILLE_THRESHOLD
for s in range(101, 121):
    fr, out, meta = EX.evaluate_split(P, y, g, seed=s, methods=['EviAlert-C'], budget=1.0, keep_curves=True,
                                      kw=dict(clip=None, kappa=0.0))
    o = out['EviAlert-C']; c = o['curve']; grid = np.array(c['grid'])
    j = int(np.where(grid >= thr - 1e-12)[0][0])
    o['ville'] = dict(h=float(grid[j]), fa=float(c['fa'][j]), recall=float(c['recall'][j]), compliant=bool(c['fa'][j] <= 1.0))
    res.append(dict(methods={'eCUSUM': o}, meta=meta))
json.dump(res, open(os.path.join(ROOT, 'results', f'explore_edetector_{model}.json'), 'w'))
print(model, 'done in', round(time.time() - t0), 's', flush=True)
