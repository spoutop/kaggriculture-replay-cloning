import argparse, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
TRACKA = os.environ.get('BES_TRACKA') or next(p for p in (os.path.join(os.path.dirname(HERE), 'trackA'), os.path.dirname(HERE)) if os.path.exists(os.path.join(p, 'np_ensemble.py')))
sys.path.insert(0, TRACKA)
import numpy as np
from np_ensemble import EnsNpPolicy
from lean_engine import LeanGame
from a5_select import _opp_policy

ap = argparse.ArgumentParser(); ap.add_argument('--bundle', required=True); ap.add_argument('--spec', required=True); ap.add_argument('--corr', default='')
ap.add_argument('--seeds', default='32309050:32309051'); ap.add_argument('--opp', default='npz:incumbent/refv2.npz'); ap.add_argument('--out', default='')
ap.add_argument('--book', default='', help='reference = clonegap ref_book.BookPolicy(EnsNpPolicy, book) (independent of the bundle BookWrap)')
ap.add_argument('--late-spec', default='', help='phase switch reference: late ensemble spec (members/weights; same --corr)'); ap.add_argument('--switch', type=int, default=0)
ap.add_argument('--detswitch', default='', help='R11 reference: np_ensemble.DetectorSwitch(spec, alt spec, detector)'); ap.add_argument('--alt-spec', default='')
ap.add_argument('--ref-kw', default='{}', help='JSON kwargs for the analyst BookPolicy reference, e.g. {"soft":0.05,"opp_detect":true,"hybrid":true}')
ap.add_argument('--book-spec', default='', help="reference = gpuread es_gpu.NpBook(EnsNpPolicy, '<npz>?tmax=..&kmin=..&fm=..&day=..&pmin=..') (the evaluated variant semantics)")
ap.add_argument('--base-bundle', default='', help='the plain E3 bundle: timed on the SAME observations (interleaved), for the timing bar')
a = ap.parse_args()
from kaggle_environments.agent import get_last_callable, read_file
main_py = os.path.join(os.path.abspath(a.bundle), 'main.py')
spec = json.load(open(a.spec)); mem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(a.spec), m) for m in spec['members']]
corr = json.load(open(a.corr)) if a.corr else None
lo, hi = map(int, a.seeds.split(':')); res = []; bad = 0; times = []; btimes = []
base_py = os.path.join(os.path.abspath(a.base_bundle), 'main.py') if a.base_bundle else None
for seed in range(lo, hi):
    for seat in (0, 1):
        agent = get_last_callable(read_file(main_py, main_py), path=main_py)
        bagent = get_last_callable(read_file(base_py, base_py), path=base_py) if base_py else None
        ref = EnsNpPolicy(mem, spec.get('weights'), corr); opp = _opp_policy(a.opp, os.environ.get('R1_DIR'))
        if a.detswitch:
            from np_ensemble import DetectorSwitch, load_detector
            al = json.load(open(a.alt_spec)); amem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(a.alt_spec), m) for m in al['members']]
            ref = DetectorSwitch(ref, EnsNpPolicy(amem, al.get('weights'), corr), load_detector(a.detswitch))
        if a.switch:
            sys.path.insert(0, os.path.join(HERE, 'gpuread')); import es_gpu as _EG
            ls = json.load(open(a.late_spec)); lmem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(a.late_spec), m) for m in ls['members']]
            ref = _EG.SwitchPol(a.switch, ref, EnsNpPolicy(lmem, ls.get('weights'), corr))
        if a.book:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clonegap', 'book')); from ref_book import BookPolicy
            ref = BookPolicy(ref, a.book, keys=('full',), **json.loads(a.ref_kw))
        if a.book_spec:
            sys.path.insert(0, os.path.join(HERE, 'gpuread')); import es_gpu as _EG
            ref = _EG.NpBook(ref, a.book_spec)
        g = LeanGame(seed); cfg = g.env.configuration; n = mism = 0; first = None
        while not g.done:
            oa, ob = g.obs(seat), g.obs(1 - seat)
            t0 = time.perf_counter(); ab_ = agent(oa, cfg); times.append(time.perf_counter() - t0)
            if bagent is not None: t0 = time.perf_counter(); bagent(oa, cfg); btimes.append(time.perf_counter() - t0)
            ar = ref(oa, cfg); n += 1
            if json.dumps(ab_, sort_keys=True) != json.dumps(ar, sort_keys=True):
                mism += 1
                if first is None: first = dict(step=int(oa.get('step', 0)), bundle=ab_, ref=ar)
            x = [None, None]; x[seat] = ab_; x[1 - seat] = opp(ob, cfg); g.step(x[0], x[1])
        bad += mism; errs = agent.__globals__.get('_ERRORS')
        res.append(dict(seed=seed, seat=seat, steps=n, mismatches=mism, first=first, rewards=g.rewards, agent_errors=errs))
        print(f'seed {seed} seat {seat}: {n} steps, bundle vs evaluated policy mismatches {mism}, agent errors {errs}, rewards {g.rewards}', flush=True)
tstat = lambda v: dict(turns=len(v), worst=round(float(np.max(v)), 1), p99=round(float(np.percentile(v, 99)), 1), mean=round(float(np.mean(v)), 1)) if len(v) else None
out = dict(bundle=os.path.abspath(a.bundle), corr=a.corr, rows=res, PASS=bad == 0 and all(r['agent_errors'] == 0 for r in res),
           timing_ms=tstat(np.array(times) * 1000), base_timing_ms=tstat(np.array(btimes) * 1000), base_bundle=a.base_bundle,
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
print('BUNDLE PARITY', 'PASS' if out['PASS'] else 'FAIL', 'corrected', out['timing_ms'], 'plain E3', out['base_timing_ms'], flush=True)
if a.out: json.dump(out, open(a.out, 'w'), indent=1, default=str)
