import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
TRACKA = os.environ.get('BES_TRACKA') or next(p for p in (os.path.join(os.path.dirname(HERE), 'trackA'), os.path.dirname(HERE)) if os.path.exists(os.path.join(p, 'np_ensemble.py')))
if TRACKA not in sys.path: sys.path.insert(0, TRACKA)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('A5_ROSTER', 'roster/agents')
import numpy as np
import codec as C
import encoder as E
from np_ensemble import EnsNpPolicy, _lsm, spec_correction
from np_policy import NpPolicy, grouped_argmax

B0_DIR = os.environ.get('BTRAIN_B0_DIR', 'b0')

def game_class(engine):
    if engine == 'lean':
        from lean_engine import LeanGame
        return LeanGame
    assert engine == 'cpp', engine
    assert os.path.exists(os.path.join(B0_DIR, 'READY')), f'{B0_DIR}/READY missing: the C++ engine is not certified'
    if B0_DIR not in sys.path: sys.path.insert(0, B0_DIR)
    from cpp_game import CppGame
    return CppGame

def load_ens_spec(path, correction='spec'):
    spec = json.load(open(path)); mem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(path), m) for m in spec['members']]
    return mem, spec.get('weights'), (spec_correction(spec, path) if correction == 'spec' else correction)

class GNet:
    def __init__(self, npm, dev):
        import torch, model as MOD
        z = npm.w
        assert not npm.uhist and not npm.gextra and not npm.ghist and not npm.glead, 'GNet: uhist / clock / history / lead inputs not supported'
        net = MOD.Net(ch=int(z['stem.weight'].shape[0]), blocks=npm.blocks); sd = net.state_dict()
        for k in sd:
            if k != 'coords': sd[k] = torch.from_numpy(np.ascontiguousarray(z[k], dtype=np.float32))
        net.load_state_dict(sd); net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        with torch.no_grad(): net.temb.weight[0] = torch.from_numpy(npm.tvec.astype(np.float32))
        self.net = net.to(dev); self.dev = dev; self.torch = torch

    def forward(self, G, GL, U, XY):
        torch = self.torch; net = self.net; B = G.shape[0]
        with torch.no_grad():
            x, g = net.trunk(G, GL)
            idx = (XY[..., 1] * 10 + XY[..., 0]).long()
            feat = x.flatten(2).gather(2, idx[:, None, :].expand(-1, x.shape[1], -1)).transpose(1, 2)
            order = (torch.arange(C.MAX_UNITS, device=self.dev, dtype=torch.float32) / C.MAX_UNITS)[None, :, None].expand(B, -1, -1)
            ue = net.uenc(torch.cat([U, order], -1))
            ul = net.uhead(torch.cat([feat, ue, g[:, None, :].expand(-1, C.MAX_UNITS, -1)], -1))
            kl = net.khead(torch.cat([g[:, None, :].expand(-1, C.MAX_SLOTS, -1), net.slot.weight[None].expand(B, -1, -1)], -1))
        return g, ul, kl

    def qty(self, g, kinds):
        torch = self.torch; net = self.net; B = g.shape[0]
        with torch.no_grad():
            return net.qhead(torch.cat([g[:, None, :].expand(-1, C.MAX_SLOTS, -1), net.slot.weight[None].expand(B, -1, -1), net.kemb(kinds)], -1))

class NetPolicy:
    def __init__(self, kind, ref, keys):
        self.kind, self.ref, self.keys = kind, ref, keys
        if kind == 'ens':
            self.w = ref.w; self.tn = ref.tnat; self.tau = ref.tau; self.overflow = ref.overflow; self.ugroup = ref.ugroup; self.corr = ref.corr
            assert not ref.uhist, 'uhist ensembles not supported on the GPU path'
        else:
            W = ref.w
            assert not ref.uhist and 'rule_hazard' not in W and 'sched_ub' not in W and 'sched_kb' not in W and not ref.dedup and ref.mode == 'argmax'
            self.ugroup = 'rule_unit_group' in W and float(W['rule_unit_group']) > 0
            self.tau = float(W['rule_consolidate']) if 'rule_consolidate' in W else 0.0
            self.tn = float(W['rule_cons_tnat']) if 'rule_cons_tnat' in W else 1.0
            self.overflow = 'rule_overflow' in W and float(W['rule_overflow']) > 0; self.corr = None

_TAKES_OBS = {}

def _takes_obs(fn):
    key = (type(getattr(fn, '__self__', None)), getattr(fn, '__name__', ''))
    if key not in _TAKES_OBS:
        import inspect
        try: _TAKES_OBS[key] = 'obs' in inspect.signature(fn).parameters
        except (TypeError, ValueError): _TAKES_OBS[key] = False
    return _TAKES_OBS[key]

def _consolidate(kl_nat, kinds, mm, tau):
    z = kl_nat.astype(np.float64); z = z - z.max(-1, keepdims=True); p_ = np.exp(z); p_ = p_ / p_.sum(-1, keepdims=True)
    any_ = 1.0 - np.prod(1.0 - p_, axis=0); any_[C.MK_NONE] = 0.0; kinds = np.array(kinds).copy()
    for kind in np.argsort(-any_, kind='stable').tolist():
        if any_[kind] > tau and mm[kind] and kind not in kinds:
            real = np.nonzero(kinds != C.MK_NONE)[0]; nxt = int(real.max()) + 1 if len(real) else 0
            if nxt < len(kinds): kinds[nxt] = kind
    return kinds

def decode_stage1(P, obs, n, um, mm, ULs, KLs, shadow=False):
    seeds = obs['private'].get('seeds', {})
    if P.kind == 'ens':
        tn = P.tn
        ul = sum(w * _lsm(np.where(um[:n], UL[:n] * tn, -1e9)) for w, UL in zip(P.w, ULs))
        u = C.fix_plant_budget((grouped_argmax(ul, 1.0) if P.ugroup else ul.argmax(-1)).tolist(), seeds)
        u0 = u; cr = P.corr; ph = None
        if cr is not None:
            ph = cr.phase(obs); cr.stats['steps'] += 1
            u = cr.units(ul, um[:n], u, seeds, ph, obs) if _takes_obs(cr.units) else cr.units(ul, um[:n], u, seeds, ph)
        kl = sum(w * _lsm(np.where(mm[None], KL * tn, -1e9)) for w, KL in zip(P.w, KLs))
        kl0 = kl
        if cr is not None: kl = cr.kinds(kl, ph, obs) if _takes_obs(cr.kinds) else cr.kinds(kl, ph)
        kinds = kl.argmax(-1)
        if P.tau > 0: kinds = _consolidate(kl, kinds, mm, P.tau)
        st = dict(u=u, kinds=kinds, ph=ph)
        if shadow and cr is not None:
            k0 = kl0.argmax(-1)
            if P.tau > 0: k0 = _consolidate(kl0, k0, mm, P.tau)
            st['u0'] = u0; st['kinds0'] = k0
        return st
    ul = np.where(um[:n], ULs[0][:n], -1e9)
    u = C.fix_plant_budget((grouped_argmax(ul, P.tn) if P.ugroup else ul.argmax(-1)).tolist(), seeds)
    kl = np.where(mm[None], KLs[0], -1e9); kinds = kl.argmax(-1)
    if P.tau > 0: kinds = _consolidate(kl.astype(np.float64) * P.tn, kinds, mm, P.tau)
    return dict(u=u, kinds=kinds, ph=None)

def decode_stage2(P, obs, st, qmax, QLs):
    kinds = np.asarray(st['kinds'])
    qm = np.maximum(qmax[kinds], 1); ok = np.arange(C.MAX_Q)[None] < qm[:, None]
    if P.kind == 'ens':
        ql = sum(w * _lsm(np.where(ok, QL * P.tn, -1e9)) for w, QL in zip(P.w, QLs))
        qs = ql.argmax(-1) + 1
        if P.corr is not None: qs = P.corr.qty(kinds, qs, qmax, st['ph'])
    else:
        ql = np.where(ok, QLs[0], -1e9); qs = ql.argmax(-1) + 1
    act = C.decode_action(st['u'], list(zip(kinds.tolist(), qs.tolist())), n_hands=len(obs['farms'][int(obs['player'])]['hands']))
    if P.overflow: act = C.overflow_sale(obs, act)
    if P.kind == 'ens' and P.corr is not None and getattr(P.corr, 'k3_on', False): act = P.corr.k3_apply(obs, act)
    if P.kind == 'ens' and P.corr is not None and getattr(P.corr, 'lt1_on', False): act = P.corr.lt1_apply(act)
    if P.kind == 'ens' and P.corr is not None and getattr(P.corr, 'lt1b_on', False): act = P.corr.lt1b_apply(act)
    return act

class Evaluator:
    def __init__(self, base_spec, device='cuda', engine='lean', r1_dir=None, shadow=False, fastenc=True):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)
        self.torch = torch; self.dev = torch.device(device); self.engine = engine; self.G = game_class(engine)
        self.base_spec = base_spec; self.r1_dir = r1_dir; self.shadow = shadow
        self.fastenc = bool(fastenc)
        self.nets = {}; self.pols = {}; self.cand_corr = {}; self._md5 = {}
        self.pols['E3'] = self._make_ens(base_spec, None)

    def _key(self, p):
        rp = os.path.realpath(p)
        if rp not in self._md5:
            import hashlib
            self._md5[rp] = hashlib.md5(open(rp, 'rb').read()).hexdigest()
        return self._md5[rp]

    def _net(self, npm, path):
        k = self._key(path)
        if k not in self.nets: self.nets[k] = GNet(npm, self.dev)
        return k

    def _make_ens(self, spec_path, correction):
        mem, wts, _ = load_ens_spec(spec_path, correction=None)
        ref = EnsNpPolicy(mem, wts, correction)
        keys = [self._net(m, p) for m, p in zip(ref.members, mem)]
        return NetPolicy('ens', ref, keys)

    def add_cand(self, name, correction):
        from np_ensemble import load_correction
        self.cand_corr[name] = correction
        base = self.pols['E3']; Q = NetPolicy.__new__(NetPolicy); Q.__dict__.update(base.__dict__); Q.corr = load_correction(correction)
        self.pols['cand:' + name] = Q

    def set_cands(self, cands):
        for k in [k for k in self.pols if k.startswith('cand:')]: del self.pols[k]
        self.cand_corr = {}
        for nm, c in cands.items(): self.add_cand(nm, c)

    def policy(self, key):
        if key in self.pols: return self.pols[key]
        if key.startswith('ens:'):
            mem, wts, corr = load_ens_spec(key[4:])
            ref = EnsNpPolicy(mem, wts, corr); keys = [self._net(m, p) for m, p in zip(ref.members, mem)]
            self.pols[key] = NetPolicy('ens', ref, keys); return self.pols[key]
        if key.startswith('npz:'):
            ref = NpPolicy(key[4:], mode='argmax', temp=1.0, team=0, salt=0)
            self.pols[key] = NetPolicy('np', ref, [self._net(ref, key[4:])]); return self.pols[key]
        return None

    def _game_policy(self, key):
        P = self.policy(key)
        if P is None:
            from a5_select import _opp_policy
            return _opp_policy(key, self.r1_dir)
        if key.startswith('cand:') and P.corr is not None:
            from np_ensemble import Correction
            Q = NetPolicy.__new__(NetPolicy); Q.__dict__.update(P.__dict__); Q.corr = Correction(P.corr.spec); return Q
        return P

    def play(self, cells, trace=False):
        torch = self.torch
        games = [self.G(int(c['seed'])) for c in cells]; cfg = games[0].env.configuration
        seatpol = []
        for c in cells:
            s = int(c['seat']); seatpol.append({s: self._game_policy(c['a']), 1 - s: self._game_policy(c['b'])})
        acrc = [[] for _ in cells] if trace else None
        diff = np.zeros(len(cells), np.int64)
        t0 = time.time(); tm = self.timing = dict(enc=0.0, cpu_agents=0.0, gpu=0.0, dec1=0.0, gpu_q=0.0, dec2=0.0, engine=0.0, steps=0)
        if self.fastenc:
            import bes_encfast as EF
            tcache = EF.TileCache()
        else:
            tcache = None
        while not all(g.done for g in games):
            live = [i for i, g in enumerate(games) if not g.done]
            acts = {i: [None, None] for i in live}
            reqs = []
            tm['steps'] += 1
            if tcache is not None: tcache.clear()
            for i in live:
                for seat in (0, 1):
                    P = seatpol[i][seat]; o = games[i].obs(seat)
                    if isinstance(P, NetPolicy):
                        t1 = time.time(); reqs.append((i, seat, P, o, EF.encode(o, tcache) if tcache is not None else E.encode(o), E.masks(o))); tm['enc'] += time.time() - t1
                    else:
                        t1 = time.time(); acts[i][seat] = P(o, cfg); tm['cpu_agents'] += time.time() - t1
            if reqs:
                by_net = {}
                for r, (i, seat, P, o, enc, msk) in enumerate(reqs):
                    for j, k in enumerate(P.keys): by_net.setdefault(k, []).append((r, j))
                t1 = time.time()
                outs = {}
                gcache = {}
                GA = torch.from_numpy(np.stack([q[4][0] for q in reqs])).to(self.dev); GLA = torch.from_numpy(np.stack([q[4][4] for q in reqs])).to(self.dev)
                UA = torch.from_numpy(np.stack([q[4][1] for q in reqs])).to(self.dev); XYA = torch.from_numpy(np.stack([q[4][2] for q in reqs])).to(self.dev)
                for k, lst in by_net.items():
                    rs = [r for r, _ in lst]
                    if len(rs) == len(reqs) and rs == list(range(len(reqs))): G, GL, U, XY = GA, GLA, UA, XYA
                    else:
                        ix = torch.as_tensor(rs, device=self.dev); G, GL, U, XY = GA[ix], GLA[ix], UA[ix], XYA[ix]
                    g, UL, KL = self.nets[k].forward(G, GL, U, XY)
                    UL = UL.float().cpu().numpy(); KL = KL.float().cpu().numpy(); gcache[k] = (g, rs)
                    for b, (r, j) in enumerate(lst): outs[(r, j)] = (UL[b], KL[b])
                tm['gpu'] += time.time() - t1; t1 = time.time()
                states = []
                for r, (i, seat, P, o, enc, msk) in enumerate(reqs):
                    n = enc[3]; um, mm, qmax = msk
                    ULs = [outs[(r, j)][0] for j in range(len(P.keys))]; KLs = [outs[(r, j)][1] for j in range(len(P.keys))]
                    states.append(decode_stage1(P, o, n, um, mm, ULs, KLs, shadow=self.shadow))
                tm['dec1'] += time.time() - t1; t1 = time.time()
                qouts = {}
                for k, (g, rs) in gcache.items():
                    K = torch.from_numpy(np.stack([np.asarray(states[r]['kinds'], np.int64) for r in rs])).to(self.dev)
                    QL = self.nets[k].qty(g, K).float().cpu().numpy()
                    for b, r in enumerate(rs): qouts[(r, k)] = QL[b]
                tm['gpu_q'] += time.time() - t1; t1 = time.time()
                for r, (i, seat, P, o, enc, msk) in enumerate(reqs):
                    QLs = [qouts[(r, k)] for k in P.keys]
                    acts[i][seat] = decode_stage2(P, o, states[r], msk[2], QLs)
                    if self.shadow and 'u0' in states[r] and seat == int(cells[i]['seat']):
                        st0 = dict(u=states[r]['u0'], kinds=states[r]['kinds0'], ph=None)
                        Pn = NetPolicy.__new__(NetPolicy); Pn.__dict__.update(P.__dict__); Pn.corr = None
                        a0 = decode_stage2(Pn, o, st0, msk[2], QLs) if np.array_equal(st0['kinds'], states[r]['kinds']) else None
                        if a0 is None or a0 != acts[i][seat]: diff[i] += 1
                tm['dec2'] += time.time() - t1
            t1 = time.time()
            for i in live:
                if trace: acrc[i].append((json.dumps(acts[i][0], sort_keys=True), json.dumps(acts[i][1], sort_keys=True)))
                games[i].step(acts[i][0], acts[i][1])
            tm['engine'] += time.time() - t1
        secs = time.time() - t0
        rows = []
        for i, c in enumerate(cells):
            s = int(c['seat']); rw = games[i].rewards; mine, theirs = float(rw[s]), float(rw[1 - s])
            Pa = seatpol[i][s]
            r = dict(c, mine=mine, theirs=theirs, win=1.0 if mine > theirs else (0.5 if mine == theirs else 0.0), margin=mine - theirs,
                     secs=round(secs / len(cells), 3), engine=self.engine)
            if isinstance(Pa, NetPolicy) and Pa.corr is not None: r['cstats'] = dict(Pa.corr.stats)
            if self.shadow: r['act_diff_steps'] = int(diff[i])
            if trace: r['trace'] = acrc[i]
            rows.append(r)
        return rows

def numpy_policy(key, base_spec, corr, r1_dir=None):
    from a5_select import _opp_policy
    if key == 'E3': mem, w, _ = load_ens_spec(base_spec, None); return EnsNpPolicy(mem, w)
    if key.startswith('cand:'): mem, w, _ = load_ens_spec(base_spec, None); return EnsNpPolicy(mem, w, corr)
    return _opp_policy(key, r1_dir)

def _numpy_game(args):
    c, spec, corr, r1, trace = args
    from lean_engine import LeanGame
    A = numpy_policy(c['a'], spec, corr, r1); B = numpy_policy(c['b'], spec, corr, r1)
    g = LeanGame(c['seed']); cfg = g.env.configuration; s = c['seat']; k = 0; first = None; nbad = 0
    while not g.done:
        oa, ob = g.obs(s), g.obs(1 - s)
        aa = A(oa, cfg); ab = B(ob, cfg); x = [None, None]; x[s] = aa; x[1 - s] = ab
        ja, jb = json.dumps(x[0], sort_keys=True), json.dumps(x[1], sort_keys=True)
        if k >= len(trace) or (ja, jb) != tuple(trace[k]):
            nbad += 1
            if first is None: first = dict(step=k, numpy=[ja[:300], jb[:300]], gpu=[t[:300] for t in trace[k]] if k < len(trace) else None)
        g.step(x[0], x[1]); k += 1
    return nbad, first, g.rewards

def validate(a):
    ev = Evaluator(a.spec, a.device, a.engine, a.r1, fastenc=bool(a.fastenc))
    corr = json.load(open(a.corr)) if a.corr else None
    if corr is not None: ev.add_cand('c', corr)
    me = 'cand:c' if corr is not None else 'E3'
    lo, hi = map(int, a.seeds.split(':'))
    cells = [dict(seed=s, seat=st, a=me, b=o) for s in range(lo, hi) for st in (0, 1) for o in a.opps.split(',')]
    t0 = time.time(); rows = ev.play(cells, trace=True); t_gpu = time.time() - t0
    from multiprocessing import get_context
    with get_context('spawn').Pool(min(len(cells), a.workers)) as pool:
        res = pool.map(_numpy_game, [(c, a.spec, corr, a.r1, r['trace']) for c, r in zip(cells, rows)], chunksize=1)
    tot_bad = 0; out = []
    for c, r, (nbad, first, rw) in zip(cells, rows, res):
        s = c['seat']; tot_bad += nbad
        out.append(dict(seed=c['seed'], seat=s, a=c['a'], b=c['b'], mismatching_steps=nbad, first=first, numpy_money=[rw[s], rw[1 - s]], gpu_money=[r['mine'], r['theirs']],
                        cstats=r.get('cstats')))
        print(f"seed {c['seed']} seat {s} {c['a']} vs {c['b']}: mismatching steps {nbad}; money numpy {rw[s]:.0f}/{rw[1 - s]:.0f} gpu {r['mine']:.0f}/{r['theirs']:.0f}"
              + (f" FIRST at step {first['step']}" if first else ''), flush=True)
    print(f'VALIDATE {"PASS" if tot_bad == 0 else "FAIL"}: {len(cells)} games, {tot_bad} mismatching steps; GPU path {t_gpu:.0f}s for all games', flush=True)
    if a.out: json.dump(dict(rows=out, PASS=tot_bad == 0, gpu_secs=t_gpu, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(a.out, 'w'), indent=1)

def bench(a):
    ev = Evaluator(a.spec, a.device, a.engine, a.r1, shadow=bool(a.shadow), fastenc=bool(a.fastenc))
    corr = json.load(open(a.corr)) if a.corr else None
    if corr is not None: ev.add_cand('c', corr)
    me = 'cand:c' if corr is not None else 'E3'
    lo = int(a.seeds.split(':')[0]); opps = a.opps.split(',')
    cells = [dict(seed=lo + i // (2 * len(opps)), seat=(i // len(opps)) % 2, a=me, b=opps[i % len(opps)]) for i in range(a.batch)]
    t0 = time.time(); rows = ev.play(cells); dt = time.time() - t0
    print(f'bench: {len(cells)} games in {dt:.1f}s = {len(cells) / dt:.2f} games/s (one process, engine {a.engine}); seconds by part: '
          + ', '.join(f'{k} {v:.1f}' for k, v in ev.timing.items()), flush=True)
    for r in rows[:6]: print({k: v for k, v in r.items() if k != 'trace'})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for nm in ('validate', 'bench'):
        p = sub.add_parser(nm); p.add_argument('--spec', required=True); p.add_argument('--corr', default=''); p.add_argument('--seeds', default='32309000:32309001')
        p.add_argument('--opps', default='E3'); p.add_argument('--device', default='cuda'); p.add_argument('--engine', default='lean'); p.add_argument('--r1', default='')
        p.add_argument('--out', default=''); p.add_argument('--batch', type=int, default=32); p.add_argument('--shadow', type=int, default=0); p.add_argument('--workers', type=int, default=16)
        p.add_argument('--fastenc', type=int, default=1)
    a = ap.parse_args()
    validate(a) if a.cmd == 'validate' else bench(a)
