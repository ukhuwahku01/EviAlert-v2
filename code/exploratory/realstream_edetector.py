"""Additional baseline on the real streams (leave-one-person-out): CUSUM e-detector (Shin, Ramdas & Rinaldo, 2024)
on calibrated likelihood ratios, with its theoretical threshold log(720) and with the calibration-based rules.
Usage: python exploratory/realstream_edetector.py wesad|nurse <features_dir> <out_json>"""
import os, sys, json, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'wesad')); sys.path.insert(0, os.path.join(ROOT, 'nurse')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lopo_curves as LC

if __name__ == '__main__':
    which, feat, outp = sys.argv[1], sys.argv[2], sys.argv[3]; out = {}
    for model in ['LR', 'RF', 'HGB']:
        t0 = time.time()
        if which == 'wesad':
            import run_wesad as RW, run_wesad_v2 as RW2
            S = RW.load(feat); P = RW.loso_probs(S, model); data = RW2.build(S, P, 0.10); tgt = RW.D; kw = {}
        else:
            import run_nurse as RN
            S = RN.load(feat); P = RN.lopo_probs(S, model); data = RN.build(S, P); tgt = RN.TARGET; kw = dict(grace=RN.GRACE)
        folds = LC.lopo_raw(data, ['eCUSUM'], tgt, **kw)
        out[model] = LC.analyse(folds, ['eCUSUM'])
        print(which, model, 'done', round(time.time() - t0), 's', flush=True)
        json.dump(out, open(outp, 'w'), default=float)
