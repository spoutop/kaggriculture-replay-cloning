import argparse, hashlib, json, shutil, sys, tarfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
MAIN = '''import {mod}_np_ensemble as ens_mod

_POLICY = ens_mod.EnsNpPolicy(ens_mod.member_paths(), ens_mod.member_weights())
_ERRORS = 0

def agent(observation, configuration=None):
    global _ERRORS
    try:
        return _POLICY(observation, configuration)
    except Exception:
        _ERRORS += 1
        if _ERRORS <= 3:
            import sys as _s, traceback as _tb
            print('ENS agent error #' + str(_ERRORS), file=_s.stderr); _tb.print_exc(file=_s.stderr)
        return {{'farmer': ['PASS'], 'hands': [], 'market': []}}
'''
MAIN_CORR = MAIN.replace('_POLICY = ens_mod.EnsNpPolicy(ens_mod.member_paths(), ens_mod.member_weights())',
                         '_POLICY = ens_mod.EnsNpPolicy(ens_mod.member_paths(), ens_mod.member_weights(), ens_mod.member_correction())')
assert MAIN_CORR != MAIN
BOOK_LINE = '\n_POLICY = ens_mod.BookWrap(_POLICY, ens_mod.book_path(), ens_mod.book_params())\n_ERRORS = 0\n'
BOOK_CODE = "class BookWrap:\n    def __init__(self, base, path, params=None):\n        import numpy as _n, json as _j\n        z = _n.load(path); self.base = base; self.tab = dict(zip(z['full_keys'].tolist(), z['full_vals'].tolist())); self.acts = [_j.loads(x) for x in z['actions']]\n        self.p = dict(tmax=10 ** 9, kmin=0, fm=0, day=0, pmin=0.0, opp=0, hyb=0); self.p.update(params or {})\n        if self.p['hyb']: self.p['opp'] = 1\n        self.cnt = dict(zip(z['full_keys'].tolist(), z['full_counts'].tolist())) if 'full_counts' in z.files else None\n        self.hits = 0; self.off = False; self.det = False; self.opp_m = {}\n\n    def key(self, obs):\n        import hashlib as _hl, numpy as _n\n        gr, un, xy, n, gl = E.encode(obs); um, mm, qm = E.masks(obs); m = _hl.blake2b(digest_size=8)\n        for p_ in (gr.astype(_n.float16), un.astype(_n.float16), xy.astype(_n.int8), gl.astype(_n.float16), mm.astype(bool), _n.minimum(qm, 127).astype(_n.int8), um.astype(bool)):\n            m.update(_n.ascontiguousarray(p_).tobytes())\n        return int(_n.frombuffer(m.digest(), _n.uint64)[0])\n\n    def plausible(self, u, kinds, ps):\n        import numpy as _n\n        ul, kl = self.base.last_ul, self.base.last_kl\n        for i, a in enumerate(list(u)[:len(ul)]):\n            if _n.exp(ul[i, int(a)]) < ps: return False\n        anyk = 1.0 - _n.prod(1.0 - _n.exp(kl), axis=0)\n        for k in kinds:\n            if int(k) != 0 and anyk[int(k)] < ps: return False\n        return True\n\n    def __call__(self, obs, cfg=None):\n        act = self.base(obs, cfg)\n        try:\n            p = self.p; step = int(obs.get('step', 0) or 0); me = int(obs['player']); ps = p['pmin']\n            if step < 1: self.off = False; self.det = False; self.opp_m = {}\n            if p['opp']:\n                if step in (1, 2): self.opp_m[step] = float(obs['farms'][1 - me]['money'])\n                if step < 2: return act\n                if step == 2: self.det = self.opp_m.get(1, 0) >= 2940 and 1000 <= self.opp_m.get(2, 0) <= 1150\n                if p['hyb']: ps = 0.0 if self.det else p['pmin']\n                elif not self.det: return act\n            if p['day'] and step // 24 >= p['day']: self.off = True\n            if self.off or step >= p['tmax']: return act\n            k = self.key(obs); ai = self.tab.get(k)\n            if ai is not None and p['kmin'] > 0 and (self.cnt is None or self.cnt.get(k, 0) < p['kmin']): ai = None\n            if ai is None:\n                if p['fm'] and step > 0: self.off = True\n                return act\n            u, kinds, qs = self.acts[ai]\n            if ps > 0 and not self.plausible(u, kinds, ps): return act\n            self.hits += 1\n            return C.decode_action(list(u), [(k_, q + 1) for k_, q in zip(kinds, qs)], n_hands=len(obs['farms'][me]['hands']))\n        except Exception:\n            return act\n\ndef book_path():\n    return str(_here() / 'book.npz')\n\ndef book_params():\n    import json as _j, os as _o\n    f = _here() / 'book_params.json'\n    return _j.load(open(f)) if _o.path.exists(str(f)) else {}\n"
SWITCH_LINE = ('\n_POLICY = ens_mod.SwitchWrap(_POLICY, ens_mod.EnsNpPolicy(ens_mod.late_paths(), ens_mod.late_weights(), '
               'ens_mod.member_correction() if hasattr(ens_mod, "member_correction") else None), ens_mod.switch_step())\n_ERRORS = 0\n')
SWITCH_CODE = 'class SwitchWrap:\n    def __init__(self, early, late, n):\n        self.early, self.late, self.n, self.cur = early, late, int(n), early\n\n    def __call__(self, obs, cfg=None):\n        t = int(obs.get(\'step\', 0) or 0); self.cur = self.early if t < self.n else self.late; return self.cur(obs, cfg)\n\n    last_u = property(lambda self: getattr(self.cur, \'last_u\', None))\n    last_ul = property(lambda self: getattr(self.cur, \'last_ul\', None))\n    last_kl = property(lambda self: getattr(self.cur, \'last_kl\', None))\n\ndef late_paths():\n    import json as _j\n    return [str(_here() / m) for m in _j.load(open(_here() / "ensemble.json"))["switch"]["members"]]\n\ndef late_weights():\n    import json as _j\n    return _j.load(open(_here() / "ensemble.json"))["switch"]["weights"]\n\ndef switch_step():\n    import json as _j\n    return int(_j.load(open(_here() / "ensemble.json"))["switch"]["step"])\n'
DET_LINE = ('\n_POLICY = ens_mod.DetectorSwitch(_POLICY, ens_mod.EnsNpPolicy(ens_mod.alt_paths(), ens_mod.alt_weights(), '
            'ens_mod.member_correction() if hasattr(ens_mod, "member_correction") else None), ens_mod.load_detector(ens_mod.detector_path()))\n_ERRORS = 0\n')
DET_CODE = ('def alt_paths():\n    import json as _j\n    return [str(_here() / m) for m in _j.load(open(_here() / "ensemble.json"))["detswitch"]["members"]]\n\ndef alt_weights():\n    import json as _j\n    return _j.load(open(_here() / "ensemble.json"))["detswitch"]["weights"]\n\ndef detector_path():\n    return str(_here() / "detector.json")\n')
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('npz', nargs='+'); ap.add_argument('--weights', default='')
ap.add_argument('--correction', default='', help='B-ES correction JSON (np_ensemble.Correction); default: none')
ap.add_argument('--book', default='', help='REF BOOK npz (clonegap/book/ref_book.py, FULL keys); default: none (bundle built exactly as before)')
ap.add_argument('--switch', type=int, default=0, help='phase switch step (HP R9): the --late ensemble plays from this step on; 0 = off')
ap.add_argument('--late', nargs='*', default=[], help='late-phase member npz files (same correction as the early ensemble)'); ap.add_argument('--late-weights', default='')
ap.add_argument('--detswitch', default='', help='R11 detector json: policy b = the --alt ensemble when the detector fires at step K (np_ensemble.DetectorSwitch)')
ap.add_argument('--alt', nargs='*', default=[], help='policy-b member npz files (same correction)'); ap.add_argument('--alt-weights', default='')
ap.add_argument('--book-params', default='', help="variant, gpuread syntax 'tmax=48&kmin=2&fm=1&day=2&pmin=0.2' -> book_params.json (default: plain book)")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
w = [float(x) for x in a.weights.split(',')] if a.weights else [1.0] * len(a.npz); assert len(w) == len(a.npz)
h = hashlib.sha256(out.name.encode())
for f in a.npz: h.update(hashlib.sha256(open(f, 'rb').read()).digest())
if a.correction: h.update(hashlib.sha256(open(a.correction, 'rb').read()).digest())
if a.book: h.update(hashlib.sha256(open(a.book, 'rb').read()).digest())
assert bool(a.switch) == bool(a.late), '--switch and --late go together'
assert bool(a.detswitch) == bool(a.alt), '--detswitch and --alt go together'; assert not (a.switch and a.detswitch), 'one switch type per bundle'
AW = [float(x) for x in a.alt_weights.split(',')] if a.alt_weights else [1.0] * len(a.alt); assert len(AW) == len(a.alt)
if a.detswitch:
    h.update(hashlib.sha256(open(a.detswitch, 'rb').read()).digest())
    for f in a.alt: h.update(hashlib.sha256(open(f, 'rb').read()).digest())
LW = [float(x) for x in a.late_weights.split(',')] if a.late_weights else [1.0] * len(a.late); assert len(LW) == len(a.late)
if a.switch:
    h.update(f'switch{a.switch}'.encode())
    for f in a.late: h.update(hashlib.sha256(open(f, 'rb').read()).digest())
BP = {}
for kv in filter(None, a.book_params.split('&')):
    k_, _, v_ = kv.partition('='); assert k_ in ('tmax', 'kmin', 'fm', 'day', 'pmin', 'opp', 'hyb'), k_; BP[k_] = float(v_) if k_ == 'pmin' else int(v_)
assert not BP or a.book, '--book-params needs --book'
if BP: h.update(json.dumps(BP, sort_keys=True).encode())
MOD = 'ens_' + h.hexdigest()[:8]
rep = lambda t: (t.replace('import codec as C, encoder as E', f'import {MOD}_codec as C, {MOD}_encoder as E')
                  .replace('import codec as C', f'import {MOD}_codec as C').replace('import encoder as E', f'import {MOD}_encoder as E')
                  .replace('from np_policy import NpPolicy', f'from {MOD}_np_policy import NpPolicy'))
for f in ('np_policy.py', 'codec.py', 'encoder.py', 'np_ensemble.py'):
    txt = rep((HERE / f).read_text())
    assert 'import codec' not in txt and 'import encoder' not in txt and 'from np_policy' not in txt, f
    if f == 'np_ensemble.py':
        txt += ('def _here():\n    from pathlib import Path as _P\n    return _P(__file__).resolve().parent\n\ndef member_paths():\n    import json as _j\n    d = _j.load(open(_here() / "ensemble.json"))\n    return [str(_here() / m) for m in d["members"]]\n\ndef member_weights():\n    import json as _j\n    return _j.load(open(_here() / "ensemble.json"))["weights"]\n')
        if a.switch:
            txt += SWITCH_CODE
        if a.detswitch:
            assert 'class DetectorSwitch' in txt, 'np_ensemble.py lacks the ENS-A51 DetectorSwitch block'; txt += DET_CODE
        if a.book:
            txt += BOOK_CODE
        if a.correction:
            txt += ('def member_correction():\n    import json as _j\n    c = _j.load(open(_here() / "ensemble.json")).get("correction")\n    return None if c is None else str(_here() / c)\n')
    (out / f'{MOD}_{f}').write_text(txt)
_ug = [('rule_unit_group' in __import__('numpy').load(f).files) for f in a.npz]
assert len(set(_ug)) == 1, f'rule_unit_group must be the same in every member: {_ug}'
assert not _ug[0] or all('rule_cons_tnat' in __import__('numpy').load(f).files for f in a.npz), 'grouped unit decoding needs rule_cons_tnat'
members = []
for i, f in enumerate(a.npz):
    shutil.copy(f, out / f'weights_{i}.npz'); members.append(f'weights_{i}.npz')
ens_json = dict(members=members, weights=w, sources=[str(Path(f).resolve()) for f in a.npz])
alt_members = []
if a.detswitch:
    sys.path.insert(0, str(HERE)); from np_ensemble import load_detector as _ld; _ld(a.detswitch)
    for i, f in enumerate(a.alt):
        shutil.copy(f, out / f'weights_alt_{i}.npz'); alt_members.append(f'weights_alt_{i}.npz')
    shutil.copy(a.detswitch, out / 'detector.json')
    ens_json['detswitch'] = dict(members=alt_members, weights=AW, sources=[str(Path(f).resolve()) for f in a.alt], detector_source=str(Path(a.detswitch).resolve()))
late_members = []
if a.switch:
    _lg = [('rule_unit_group' in __import__('numpy').load(f).files) for f in a.late]; assert len(set(_lg)) == 1, f'late rule_unit_group differs: {_lg}'
    for i, f in enumerate(a.late):
        shutil.copy(f, out / f'weights_late_{i}.npz'); late_members.append(f'weights_late_{i}.npz')
    ens_json['switch'] = dict(step=a.switch, members=late_members, weights=LW, sources=[str(Path(f).resolve()) for f in a.late])
if a.correction:
    import numpy as _np
    sys.path.insert(0, str(HERE)); from np_ensemble import Correction as _Corr
    _c = _Corr(a.correction); assert _c.active, 'the correction is all zero: build without --correction'
    shutil.copy(a.correction, out / 'correction.json'); ens_json['correction'] = 'correction.json'; ens_json['correction_source'] = str(Path(a.correction).resolve())
json.dump(ens_json, open(out / 'ensemble.json', 'w'), indent=1)
_mt = (MAIN_CORR if a.correction else MAIN).format(mod=MOD, k=len(a.npz))
if a.switch:
    _mt = _mt.replace('\n_ERRORS = 0\n', SWITCH_LINE, 1); assert 'SwitchWrap' in _mt
if a.detswitch:
    _mt = _mt.replace('\n_ERRORS = 0\n', DET_LINE, 1); assert 'DetectorSwitch' in _mt
if a.book:
    shutil.copy(a.book, out / 'book.npz'); ens_json['book'] = 'book.npz'; ens_json['book_source'] = str(Path(a.book).resolve())
    if BP: json.dump(BP, open(out / 'book_params.json', 'w'), sort_keys=True); ens_json['book_params'] = BP
    json.dump(ens_json, open(out / 'ensemble.json', 'w'), indent=1)
    _mt = _mt.replace('\n_ERRORS = 0\n', BOOK_LINE, 1); assert 'BookWrap' in _mt
(out / 'main.py').write_text(_mt)
_ns = {}; sys.path.append(str(out)); exec(compile((out / 'main.py').read_text(), 'main.py', 'exec'), _ns); sys.path.remove(str(out))
_last = [v for v in _ns.values() if callable(v)][-1]; assert getattr(_last, '__name__', '') == 'agent', _last
names = ['main.py', f'{MOD}_np_ensemble.py', f'{MOD}_np_policy.py', f'{MOD}_codec.py', f'{MOD}_encoder.py', 'ensemble.json'] + members + (['correction.json'] if a.correction else []) + (['book.npz'] if a.book else []) + (['book_params.json'] if BP else []) + late_members + alt_members + (['detector.json'] if a.detswitch else [])
sha = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in names}
json.dump(dict(built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), members=[str(f) for f in a.npz], weights=w, sha256=sha,
               **({'correction': str(Path(a.correction).resolve())} if a.correction else {})), open(out / 'BUNDLE.json', 'w'), indent=1)
with tarfile.open(str(out) + '.tar.gz', 'w:gz') as tf:
    for n in names: tf.add(out / n, arcname=n)
print('bundle', out, 'tar', str(out) + '.tar.gz', round(Path(str(out) + '.tar.gz').stat().st_size / 1e6, 1), 'MB', 'module', MOD)
