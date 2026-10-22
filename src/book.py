import argparse, glob, hashlib, json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'rank1_20260923', 'trackA')
TA = os.environ.get('BOOK_TA') or os.path.abspath(os.path.join(HERE, '..', '..', 'trackA'))
if TA not in sys.path: sys.path.insert(0, TA)
OPP_GLOB = list(range(11, 17))

def _h(parts):
    m = hashlib.blake2b(digest_size=8)
    for p in parts: m.update(np.ascontiguousarray(p).tobytes())
    return np.frombuffer(m.digest(), np.uint64)[0]

def keys_of(grid, units, xy, glob_, mmask, qmax, umask):
    full = _h([grid, units, xy, glob_, mmask, qmax, umask])
    g2 = glob_.copy(); g2[OPP_GLOB] = 0
    own = _h([grid[0], units, xy, g2, mmask, qmax, umask])
    return full, own

def encode_for_key(obs):
    import encoder as E
    gr, un, xy, n, gl = E.encode(obs); um, mm, qm = E.masks(obs)
    return keys_of(gr.astype(np.float16), un.astype(np.float16), xy.astype(np.int8), gl.astype(np.float16), mm.astype(bool), np.minimum(qm, 127).astype(np.int8), um.astype(bool))

def build(a):
    sys.path.insert(0, os.path.dirname(HERE))
    src = []
    if not a.dirs:
        submap = json.load(open(os.path.join(os.path.dirname(HERE), 'stem_sub_v3d.json')))
        src = [(f, '56541087') for f in sorted(glob.glob(os.path.join(os.path.dirname(HERE), 'data_newest', '*.npz')))]
        extra = sorted(glob.glob(os.path.join(TA, 'data', 'ref_v3d', '*.npz'))) + sorted(glob.glob(os.path.join(os.path.dirname(HERE), 'data_hotop', '*.npz')))
        for f in extra: src.append((f, submap.get(os.path.basename(f)[:-4], '56527482' if 'hotop' in f else '0')))
    if a.wide:
        subs = {}
        for f_ in glob.glob(os.environ.get('KGR_ROOT', '.') + '/replays/**/EPISODES_*.json', recursive=True) + glob.glob(os.path.join(TA, '..', 'ladder', 'ep', 'EP_*.json')):
            try: d_ = json.load(open(f_))
            except Exception: continue
            for e_ in (d_.get('episodes', []) if isinstance(d_, dict) else []):
                for ag in e_.get('agents', []) or []: subs[(int(e_['id']), ag.get('index', 0) or 0)] = ag.get('submissionId') or 0
        seen = {}
        for dd in ['ref_v3d', 'ref_v3d_new', 'ref_v3c', 'ref_v3c_new', 'ref_v3c_old', 'ref_v3b', 'ref_live', 'ref_0923new', 'ref_holdout', 'ref']:
            for f_ in glob.glob(os.path.join(TA, 'data', dd, '*.npz')): seen.setdefault(os.path.basename(f_), f_)
        for f_ in glob.glob(os.path.join(os.path.dirname(HERE), 'data_newest', '*.npz')) + glob.glob(os.path.join(os.path.dirname(HERE), 'data_hotop', '*.npz')): seen.setdefault(os.path.basename(f_), f_)
        src = []
        for stem, f_ in seen.items():
            e_, s_ = stem[:-4].split('_'); src.append((f_, str(subs.get((int(e_), int(s_)), 0))))
    if a.dirs:
        sm = json.load(open(a.submap)) if a.submap else {}; seen = {}
        for dd in a.dirs.split(','):
            for f_ in glob.glob(os.path.join(dd, '*.npz')): seen.setdefault(os.path.basename(f_), f_)
        src = [(f_, str(sm.get(stem[:-4], 0))) for stem, f_ in seen.items()]
    if a.newest_only: src = [s for s in src if s[1] == '56541087']
    src.sort(key=lambda x: (-int(x[1]), -int(os.path.basename(x[0]).split('_')[0])))
    excl = set(json.load(open(a.exclude))) if a.exclude else set()
    books = {'full': {}, 'own': {}}; amb = {'full': set(), 'own': set()}; first_ver = {'full': {}, 'own': {}}; cnt = {'full': collections.Counter(), 'own': collections.Counter()}
    acts = {}; act_list = []; n_rows = 0
    for f, ver in src:
        eid = int(os.path.basename(f).split('_')[0])
        if eid in excl: continue
        z = np.load(f); d = {k: z[k] for k in ('grid', 'units', 'xy', 'glob', 'mmask', 'qmax', 'umask', 'n_units', 'a_unit', 'a_kind', 'a_q')}; T = min(a.tmax, len(d['glob']))
        for t in range(T):
            kf, ko = keys_of(d['grid'][t], d['units'][t], d['xy'][t], d['glob'][t], d['mmask'][t], d['qmax'][t], d['umask'][t])
            n = int(d['n_units'][t]); av = (tuple(int(x) for x in d['a_unit'][t][:n]), tuple(int(x) for x in d['a_kind'][t]), tuple(int(x) for x in d['a_q'][t]))
            ai = acts.get(av)
            if ai is None: ai = acts[av] = len(act_list); act_list.append(av)
            for nm, k in (('full', kf), ('own', ko)):
                if k in amb[nm]: continue
                b = books[nm]
                if k not in b: b[k] = ai; first_ver[nm][k] = ver; cnt[nm][k] = 1
                elif b[k] == ai: cnt[nm][k] += 1
                elif b[k] != ai:
                    if first_ver[nm][k] == ver or not a.newest_wins: amb[nm].add(k); del b[k]

            n_rows += 1
    out = {}
    for nm in ('full', 'own'):
        ks = np.array(sorted(books[nm]), dtype=np.uint64); out[f'{nm}_keys'] = ks; out[f'{nm}_vals'] = np.array([books[nm][k] for k in ks], np.int32); out[f'{nm}_counts'] = np.array([cnt[nm][k] for k in ks.tolist()], np.int32)
        out[f'{nm}_ambiguous'] = np.array(sorted(amb[nm]), dtype=np.uint64)
    out['actions'] = np.array([json.dumps(x) for x in act_list])
    used = sorted(set(int(os.path.basename(f).split('_')[0]) for f, _ in src) - excl)
    out['source_eids'] = np.array(used, np.int64)
    out['meta'] = np.array(json.dumps(dict(tmax=a.tmax, rows=n_rows, sources=len(src), newest_only=a.newest_only, excluded=len(excl), newest_wins=a.newest_wins, wide=a.wide, exclude_file=a.exclude)))
    np.savez_compressed(a.out, **out)
    print(a.out, 'rows', n_rows, {nm: (len(books[nm]), len(amb[nm])) for nm in books}, 'actions', len(act_list), 'MB', round(os.path.getsize(a.out) / 1e6, 2))

class BookPolicy:
    def __init__(self, base, path, keys=('full',), cutoff=10 ** 9, min_count=1, sticky=False, opp_detect=False, soft=0.0, hybrid=False):
        self.soft = float(soft); self.hybrid = bool(hybrid); self.soft0 = float(soft)
        z = np.load(path); self.base = base; self.keys = keys; self.cutoff = cutoff; self.sticky = sticky; self.opp_detect = opp_detect; self.off = False; self.opp_m = {}
        cn = {nm: (z[f'{nm}_counts'] if f'{nm}_counts' in z.files else np.ones(len(z[f'{nm}_keys']), np.int32)) for nm in keys}
        self.tab = {nm: {k: v for k, v, c in zip(z[f'{nm}_keys'].tolist(), z[f'{nm}_vals'].tolist(), cn[nm].tolist()) if c >= min_count} for nm in keys}
        self.actions = [json.loads(s) for s in z['actions']]; self.stats = collections.Counter(); self.hits_by_day = collections.Counter(); self.steps_by_day = collections.Counter()
        self.corr = getattr(base, 'corr', None)

    def __call__(self, obs, cfg=None):
        import codec as C
        act = self.base(obs, cfg)
        step = int(obs.get('step', 0)); day = step // 24; self.steps_by_day[day] += 1
        me = int(obs['player'])
        if self.opp_detect:
            if step in (1, 2): self.opp_m[step] = float(obs['farms'][1 - me]['money'])
            if step < 2: self.stats['detect_wait'] += 1; return act
            if step == 2:
                self.det = self.opp_m.get(1, 0) >= 2940 and 1000 <= self.opp_m.get(2, 0) <= 1150; self.stats['detect_on' if self.det else 'detect_off'] += 1
                if self.hybrid: self.soft = 0.0 if self.det else self.soft0
            if not getattr(self, 'det', False) and not self.hybrid: return act
        if self.off or step >= self.cutoff: self.stats['off'] += 1; return act
        kf, ko = encode_for_key(obs); ai = None
        for nm, k in (('full', kf), ('own', ko)):
            if nm in self.tab and int(k) in self.tab[nm]: ai = self.tab[nm][int(k)]; self.stats[nm] += 1; break
        if ai is None:
            self.stats['miss'] += 1
            if self.sticky and step > 0: self.off = True
            return act
        u, kinds, qs = self.actions[ai]
        if self.soft > 0 and not self._plausible(obs, u, kinds):
            self.stats['soft_reject'] += 1; return act
        self.hits_by_day[day] += 1
        slots = [(k, q + 1) for k, q in zip(kinds, qs)]
        return C.decode_action(list(u), slots, n_hands=len(obs['farms'][me]['hands']))

def _ens_logprobs(ens, obs):
    import encoder as E
    from np_ensemble import _lsm
    grid, units, xy, n, glob_ = E.encode(obs); um, mm, qmax = E.masks(obs); tn = ens.tnat; feats = []
    for m in ens.members:
        hist = m._hist.buf and E.hist_block(m._hist.buf) if hasattr(m, '_hist') and m.ghist else None
        feats.append(m.features(grid, glob_, hist))
    ul = sum(w * _lsm(np.where(um[:n], m.unit_logits(x, g, units, xy, n, None) * tn, -1e9)) for w, m, (x, g) in zip(ens.w, ens.members, feats))
    kl = sum(w * _lsm(np.where(mm[None], m.kind_logits(g) * tn, -1e9)) for w, m, (x, g) in zip(ens.w, ens.members, feats))
    return ul, kl

def _plausible_impl(self, obs, u, kinds):
    ens = self.base
    if not hasattr(ens, 'members'): return True
    ul, kl = _ens_logprobs(ens, obs); p = self.soft
    for i, a in enumerate(u[:len(ul)]):
        if np.exp(ul[i, int(a)]) < p: return False
    pk = np.exp(kl); anyk = 1.0 - np.prod(1.0 - pk, axis=0)
    for k in kinds:
        if int(k) != 0 and anyk[int(k)] < p: return False
    return True

BookPolicy._plausible = _plausible_impl

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd')
    b = sp.add_parser('build'); b.add_argument('--out', required=True); b.add_argument('--tmax', type=int, default=240); b.add_argument('--newest-only', type=int, default=0)
    b.add_argument('--exclude', default='', help='JSON list of episode ids to keep OUT of the book (e.g. the GT9 panel, for held-out coverage)'); b.add_argument('--newest-wins', type=int, default=1); b.add_argument('--wide', type=int, default=0); b.add_argument('--dirs', default=''); b.add_argument('--submap', default='')
    a = ap.parse_args()
    if a.cmd == 'build': build(a)
