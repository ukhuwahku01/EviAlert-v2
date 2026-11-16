"""Summary of the e-detector comparison (comparison of accumulators). Exploratory, added after the confirmatory test.
Usage: python exploratory/summarize_edetector.py [out_json]"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results')
MODELS = ['resnet50', 'vit_b16', 'convnext_t', 'swin_t', 'effnet_b0']
out = {'infant': {}, 'wesad': {}, 'nurse': {}}

# infant sessions: 5 backbones x 20 splits
E, F = [], []
for m in MODELS:
    E += [r['methods']['eCUSUM'] for r in json.load(open(f'{R}/explore_edetector_{m}.json'))]
    F += [r['methods']['EviAlert-F'] for r in json.load(open(f'{R}/main_v2_{m}.json'))]
inf = {}
inf['ville_fa_mean'] = float(np.mean([e['ville']['fa'] for e in E]))
inf['ville_fa_min'] = float(np.min([e['ville']['fa'] for e in E])); inf['ville_fa_max'] = float(np.max([e['ville']['fa'] for e in E]))
inf['ville_within'] = float(np.mean([e['ville']['compliant'] for e in E]))
inf['ville_recall'] = float(np.mean([e['ville']['recall'] for e in E]))
for x in ('0.5', '1.0', '2.0'):
    a = np.array([e['recall@' + x] for e in E]); b = np.array([f['recall@' + x] for f in F])
    inf['R@' + x] = dict(edet=float(a.mean()), evialert=float(b.mean()), diff=float((b - a).mean()),
                         evialert_higher=int((b > a).sum()), lower=int((b < a).sum()), p=float(wilcoxon(b, a).pvalue))
for rule in ('naive', 'poisson', 'gboot'):
    inf['within_' + rule] = float(np.mean([e[rule]['compliant'] for e in E]))
g = [e['gboot'] for e in E]
inf['gboot'] = dict(feasible=float(np.mean([x['feasible'] for x in g])), fa=float(np.mean([x['fa'] for x in g])),
                    recall=float(np.mean([x['recall'] for x in g])), stale_per150h=float(np.mean([x['stale'] for x in g])))
inf['within_gboot_evialert'] = float(np.mean([f['gboot']['compliant'] for f in F]))
inf['gboot_evialert'] = dict(fa=float(np.mean([f['gboot']['fa'] for f in F])), recall=float(np.mean([f['gboot']['recall'] for f in F])),
                             stale_per150h=float(np.mean([f['gboot']['stale'] for f in F])))
inf['per_backbone_R@1'] = {m: float(np.mean([e['recall@1.0'] for e in E[20 * i:20 * i + 20]])) for i, m in enumerate(MODELS)}
out['infant'] = inf

# real streams: leave-one-person-out
for which in ('wesad', 'nurse'):
    D = json.load(open(f'{R}/explore_edetector_{which}.json')); X = json.load(open(f'{R}/explore_{which}.json'))
    for m in D:
        e = D[m]['eCUSUM']; f = X[m]['EviAlert-F']
        o = dict(ville=e['ville'], ltt=e['ltt'], crc=e['crc'], naive=e['naive'], insuff=e['insuff'],
                 evialert_ltt=f['ltt'], evialert_crc=f['crc'], evialert_naive=f['naive'])
        ks = sorted(set(e['person']) & set(f['person']))
        for x in ('0.5', '1.0', '2.0'):
            a = np.array([e['person'][k]['recall@' + x] for k in ks]); b = np.array([f['person'][k]['recall@' + x] for k in ks])
            try: p = float(wilcoxon(a, b).pvalue)
            except ValueError: p = float('nan')
            o['person_R@' + x] = dict(edet=float(a.mean()), evialert=float(b.mean()), edet_higher=int((a > b).sum()),
                                      lower=int((a < b).sum()), p_unadjusted=p, n=len(ks))
        out[which][m] = o
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else f'{R}/explore_edetector_summary.json', 'w'), indent=1)
print(json.dumps(out['infant'], indent=1))
for w in ('wesad', 'nurse'):
    for m, o in out[w].items():
        print(w, m, 'ville', {k: round(v, 3) for k, v in o['ville'].items()})
        print('   ltt e/F', {k: round(v, 3) for k, v in o['ltt'].items()}, {k: round(v, 3) for k, v in o['evialert_ltt'].items()})
        print('   crc e/F', {k: round(v, 3) for k, v in o['crc'].items()}, {k: round(v, 3) for k, v in o['evialert_crc'].items()})
        print('   R@1', o['person_R@1.0'])
