import json, zlib
import numpy as np
import codec as C, encoder as E

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def conv3(x, w, b):
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))
    return np.einsum('cyxij,ocij->oyx', win, w, optimize=True) + b[:, None, None]

def gnorm(x, w, b, groups=8, eps=1e-5):
    c = x.shape[0]; g = x.reshape(groups, -1)
    g = (g - g.mean(1, keepdims=True)) / np.sqrt(g.var(1, keepdims=True) + eps)
    return g.reshape(x.shape) * w[:, None, None] + b[:, None, None]

_TILE_OPS = {'WATER', 'HARVEST', 'FERTILIZE', 'DIG', 'BUILD_COOP', 'BUILD_PASTURE', 'FEED', 'CARE', 'COLLECT_FERTILIZER', 'PLANT'}

def lin(x, w, b):
    return x @ w.T + b

def _unit_groups():
    named = {'PASS': 0, 'NORTH': 1, 'SOUTH': 1, 'EAST': 1, 'WEST': 1, 'PICKUP': 2, 'PLACE': 3, 'PLANT': 4}; names = ['PASS', 'MOVE', 'PICKUP', 'PLACE', 'PLANT']
    gid = []
    for a in C.UNIT_ACTIONS:
        if a[0] not in named: named[a[0]] = len(names); names.append(a[0])
        gid.append(named[a[0]])
    gid = np.array(gid, np.int64); onehot = np.zeros((len(gid), len(names))); onehot[np.arange(len(gid)), gid] = 1.0
    return gid, names, onehot

UNIT_GROUP, UNIT_GROUP_NAMES, _UNIT_GROUP_ONEHOT = _unit_groups()

def grouped_argmax(ul, tnat=1.0):
    z = np.asarray(ul, np.float64) * float(tnat); z = z - z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
    g = (p @ _UNIT_GROUP_ONEHOT).argmax(-1)
    return np.where(UNIT_GROUP[None, :] == g[:, None], ul, -np.inf).argmax(-1)

class UnitHist:
    def __init__(self):
        self.seats = {}

    def get(self, obs):
        me = int(obs['player']); step = int(obs.get('step', 0)); s = self.seats.get(me)
        if s is None or step == 0 or step < s['step']:
            s = self.seats[me] = {'step': -1, 'acts': {}}
        uh = np.full((C.MAX_UNITS, 2), C.N_UNIT, np.int64)
        for j in (0, 1):
            a = s['acts'].get(step - 1 - j)
            if a is not None: uh[:, j] = a
        return uh

    def push(self, obs, u):
        me = int(obs['player']); step = int(obs.get('step', 0)); s = self.seats.setdefault(me, {'step': -1, 'acts': {}})
        a = np.full(C.MAX_UNITS, C.N_UNIT, np.int64); a[:len(u)] = np.asarray(u, np.int64)[:C.MAX_UNITS]
        s['acts'] = {k: v for k, v in s['acts'].items() if step - 1 <= k < step}; s['acts'][step] = a; s['step'] = step

def obs_seed(obs):
    me = int(obs['player']); farm = obs['farms'][me]
    v = [int(obs.get('step', 0)), me] + [int(round(float(f['money']))) for f in obs['farms']]
    v += [int(obs['market']['inventory'][p]) for p in C.PRODUCTS]
    v += [int(obs['private']['shed'].get(it, 0)) for it in C.ITEMS]
    v += [int(farm['farmer'][0]), int(farm['farmer'][1])] + [int(c) for h in farm['hands'] for c in h[:2]]
    return zlib.crc32(','.join(map(str, v)).encode('ascii'))

class NpPolicy:
    def __init__(self, path, mode='argmax', temp=1.0, team=0, temp_market=None, salt=0, record=False, dedup=False):
        self.dedup = bool(dedup)
        self.salt, self.record, self.last = int(salt), bool(record), None
        self.mode, self.temp, self.team = mode, float(temp), team
        self.temp_market = float(temp) if temp_market is None else float(temp_market)
        z = np.load(path)
        _dq = lambda k: z[k].astype(np.float32) * z['_q8s__' + k].astype(np.float32).reshape((-1,) + (1,) * (z[k].ndim - 1)) if z[k].dtype == np.int8 else z[k].astype(np.float32)
        self.w = {k: _dq(k) for k in z.files if not k.startswith('_')}
        _wd = int(z['g0.0.weight'].shape[1]) - 64; self.glead = int(z['_glead']) if '_glead' in z.files else 0
        self.gextra, self.ghist = (int(z['_gextra']), int(z['_ghist'])) if '_gextra' in z.files else {0: (0, 0), 31: (31, 0), 27: (0, 27), 58: (31, 27)}[_wd]; assert self.gextra + self.glead + self.ghist == _wd, (self.gextra, self.glead, self.ghist, _wd)

        self.uhist = int(z['_uhist']) if '_uhist' in z.files else 0
        if ('uh1.weight' in self.w or 'uh2.weight' in self.w) != bool(self.uhist):
            raise ValueError(f'{path}: _uhist flag {self.uhist} disagrees with the uh1/uh2 weights present: re-export with a5_select.export')
        if any(k.startswith('ua_') for k in self.w):
            raise ValueError(f'{path}: unit attention (uattn) weights present; NpPolicy does not implement uattn')
        if self.uhist:
            _ud = self.w['uenc.0.weight'].shape[0]
            assert self.w['uh1.weight'].shape == (C.N_UNIT + 1, _ud) and self.w['uh2.weight'].shape == (C.N_UNIT + 1, _ud), (self.w['uh1.weight'].shape, C.N_UNIT, _ud)
        self.blocks = int(z['_blocks']); ys, xs = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), indexing='ij')
        self.coords = np.stack([ys, xs]).astype(np.float32)
        self.tvec = self.style_vector(self.team)

    def style_vector(self, spec):
        E_ = self.w['temb.weight']
        if isinstance(spec, (int, np.integer)) or str(spec).lstrip('-').isdigit():
            return E_[int(spec)].copy()
        if str(spec).endswith('.npy'):
            return np.load(spec).astype(np.float32)
        v = np.zeros_like(E_[0])
        for part in str(spec).split(','):
            k, wt = part.split(':'); v += float(wt) * E_[int(k)]
        return v

    def seq(self, name, x, idxs, act_last=False):
        for n, i in enumerate(idxs):
            x = lin(x, self.w[f'{name}.{i}.weight'], self.w[f'{name}.{i}.bias'])
            if n < len(idxs) - 1 or act_last:
                x = gelu(x)
        return x

    def _pick(self, logits, rng, temp=None):
        if rng is None:
            return logits.argmax(-1)
        z = logits.astype(np.float64) / max(self.temp if temp is None else temp, 1e-6); z = z - z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
        c = p.cumsum(-1); r = rng.random((p.shape[0], 1))
        return np.minimum((c < r).sum(-1), p.shape[1] - 1)

    def features(self, grid, glob, ghist=None):
        W = self.w; base = glob
        if self.gextra:
            hour = int(np.clip(np.round(glob[2] * 24), 0, 23)); day = np.round(glob[1] * 30); step = glob[0] * 720; tf = np.zeros(31, np.float32)
            tf[hour] = 1.0; tf[24 + hour % 4] = 1.0; tf[28] = float(day >= 28); tf[29] = float(day >= 29); tf[30] = float(np.clip(1 - (720 - step) / 48, 0, 1))
            glob = np.concatenate([glob, tf])
        if self.glead:
            own = np.expm1(np.float32(base[5]) * np.float32(12.0)); opp = np.expm1(np.float32(base[11]) * np.float32(12.0)); lead = own - opp
            glob = np.concatenate([glob, np.array([np.tanh(lead / 3000), np.tanh(lead / 12000), np.clip(lead / (own + opp + 1000), -1, 1), np.tanh((own + opp) / 200000)], np.float32)])
        if self.ghist:
            glob = np.concatenate([glob, ghist if ghist is not None else np.zeros(self.ghist, np.float32)])
        g = self.seq('g0', glob, (0, 2), act_last=True) + self.tvec
        x = conv3(np.concatenate([grid[0], self.coords]), W['stem.weight'], W['stem.bias'])
        for b in range(self.blocks):
            p = f'blocks.{b}.'
            h = conv3(gelu(gnorm(x, W[p + 'n1.weight'], W[p + 'n1.bias'])), W[p + 'c1.weight'], W[p + 'c1.bias'])
            ab = lin(g, W[p + 'film.weight'], W[p + 'film.bias']); ch = x.shape[0]
            h = gnorm(h, W[p + 'n2.weight'], W[p + 'n2.bias']) * (1 + ab[:ch, None, None]) + ab[ch:, None, None]
            x = x + conv3(gelu(h), W[p + 'c2.weight'], W[p + 'c2.bias'])
        o = gelu(conv3(grid[1], W['ostem.0.weight'], W['ostem.0.bias'])); o = gelu(conv3(o, W['ostem.2.weight'], W['ostem.2.bias']))
        pooled = np.concatenate([x.mean((1, 2)), x.max((1, 2)), o.mean((1, 2)), o.max((1, 2))])
        return x, self.seq('g1', np.concatenate([g, pooled]), (0, 2), act_last=True)

    def unit_logits(self, x, g, units, xy, n, uh=None):
        W = self.w
        feat = x[:, xy[:n, 1], xy[:n, 0]].T
        order = (np.arange(n, dtype=np.float32) / C.MAX_UNITS)[:, None]
        ue = gelu(lin(np.concatenate([units[:n], order], 1), W['uenc.0.weight'], W['uenc.0.bias']))
        if self.uhist:
            h = np.full((n, 2), C.N_UNIT, np.int64) if uh is None else np.asarray(uh, np.int64)[:n]
            ue = ue + W['uh1.weight'][h[:, 0]] + W['uh2.weight'][h[:, 1]]
        return self.seq('uhead', np.concatenate([feat, ue, np.repeat(g[None], n, 0)], 1), (0, 2, 4))

    def kind_logits(self, g):
        return self.seq('khead', np.concatenate([np.repeat(g[None], C.MAX_SLOTS, 0), self.w['slot.weight']], 1), (0, 2))

    def qty_logits(self, g, kinds):
        return self.seq('qhead', np.concatenate([np.repeat(g[None], C.MAX_SLOTS, 0), self.w['slot.weight'], self.w['kemb.weight'][kinds]], 1), (0, 2))

    def __call__(self, obs, cfg=None):
        return self.act(obs)

    def act(self, obs):
        rng = None
        if self.mode == 'sample':
            rng = np.random.default_rng(obs_seed(obs) + self.salt * 1_000_003)
        _dw = getattr(self, 'dev_window', None)
        if _dw is not None and _dw[0] <= int(obs.get('step', 0)) < _dw[1]: rng = np.random.default_rng(obs_seed(obs) + self.salt * 1_000_003)
        grid, units, xy, n, glob = E.encode(obs); um, mm, qmax = E.masks(obs)
        if not hasattr(self, '_hist'): self._hist = E.InvHist()
        x, g = self.features(grid, glob, self._hist.push(obs))
        day = min(int(obs.get('step', 0)) // 24, 30); W = self.w
        uh = None
        if self.uhist:
            if not hasattr(self, '_uh'): self._uh = UnitHist()
            uh = self._uh.get(obs)
        ulr = self.unit_logits(x, g, units, xy, n, uh)
        if 'sched_ub' in W: ulr = ulr + W['sched_ub'][day][None, :]
        ul = np.where(um[:n], ulr, -1e9)
        if rng is None and 'rule_unit_group' in W and float(W['rule_unit_group']) > 0:
            u = grouped_argmax(ul, float(W['rule_cons_tnat']) if 'rule_cons_tnat' in W else 1.0).tolist()
        else:
            u = self._pick(ul, rng).tolist()
        u_raw = list(u)
        if self.dedup:
            taken = set()
            for i in range(n):
                for _ in range(4):
                    op = C.UNIT_ACTIONS[u[i]][0]; key = (int(xy[i, 0]), int(xy[i, 1]), op)
                    if op in _TILE_OPS and key in taken:
                        ul[i, u[i]] = -1e9; u[i] = int(ul[i].argmax())
                    else:
                        if op in _TILE_OPS: taken.add(key)
                        break
        u = C.fix_plant_budget(u, obs['private'].get('seeds', {}))
        if self.uhist: self._uh.push(obs, u)
        klr = self.kind_logits(g)
        if 'sched_kb' in W: klr = klr + W['sched_kb'][day][None, :]
        kl = np.where(mm[None], klr, -1e9); kinds = self._pick(kl, rng, self.temp_market)
        if 'rule_consolidate' in W and float(W['rule_consolidate']) > 0:
            tau = float(W['rule_consolidate']); tn = float(W['rule_cons_tnat']) if 'rule_cons_tnat' in W else 1.0
            z = kl.astype(np.float64) * tn; z = z - z.max(-1, keepdims=True); p_ = np.exp(z); p_ = p_ / p_.sum(-1, keepdims=True)
            any_ = 1.0 - np.prod(1.0 - p_, axis=0); any_[C.MK_NONE] = 0.0; kinds = np.array(kinds).copy()
            for kind in np.argsort(-any_, kind='stable').tolist():
                if any_[kind] > tau and mm[kind] and kind not in kinds:
                    real = np.nonzero(kinds != C.MK_NONE)[0]; nxt = int(real.max()) + 1 if len(real) else 0
                    if nxt < len(kinds): kinds[nxt] = kind
        if 'rule_hazard' in W and float(W['rule_hazard']) > 0:
            th = float(W['rule_hazard']); tn = float(W['rule_cons_tnat']) if 'rule_cons_tnat' in W else 1.0
            if not hasattr(self, '_hz') or int(obs.get('step', 0)) < 1: self._hz = np.zeros(C.N_MK, np.float32)
            z = kl.astype(np.float64) * tn; z = z - z.max(-1, keepdims=True); p_ = np.exp(z); p_ = p_ / p_.sum(-1, keepdims=True)
            an_ = (np.minimum(p_.sum(0), 1.0) * mm).astype(np.float32); an_[C.MK_NONE] = 0.0
            H_ = self._hz + an_; kinds = np.array(kinds).copy(); H_[kinds] = 0.0
            fire_ = (H_ >= th) & mm.astype(bool)
            while fire_.any():
                kd = int(np.where(fire_, H_, -1.0).argmax())
                if H_[kd] <= 0: break
                fire_[kd] = False
                real = np.nonzero(kinds != C.MK_NONE)[0]; nxt = int(real.max()) + 1 if len(real) else 0
                if nxt < len(kinds): kinds[nxt] = kd; H_[kd] = 0.0
            self._hz = H_
        qm = np.maximum(qmax[kinds], 1); ql = np.where(np.arange(C.MAX_Q)[None] < qm[:, None], self.qty_logits(g, kinds), -1e9); qs = self._pick(ql, rng, self.temp_market) + 1
        if self.record:
            self.last = dict(grid=grid, units=units, xy=xy, n=n, glob=glob, umask=um, mmask=mm, qmax=qmax, a_unit=list(u), a_kind=kinds.copy(), a_q=qs - 1,
                             uh=None if uh is None else uh.copy(), a_unit_raw=u_raw, step=int(obs.get('step', 0)), ul=ul.copy(), kl=kl.copy())
        act_ = C.decode_action(u, list(zip(kinds.tolist(), qs.tolist())), n_hands=len(obs['farms'][int(obs['player'])]['hands']))
        if 'rule_overflow' in self.w and float(self.w['rule_overflow']) > 0: act_ = C.overflow_sale(obs, act_)
        return act_

def weights_path():
    from pathlib import Path as _P
    return str(_P(__file__).resolve().parent / 'weights.npz')

def export(ckpt_path, out_npz):
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arrs = {k: v.float().numpy() for k, v in ck['state'].items() if k != 'coords'}
    arrs['_blocks'] = np.array(ck.get('blocks', 6)); arrs['_ch'] = np.array(ck.get('ch', 96))
    if ck.get('glead'): arrs['_glead'] = np.array(int(ck['glead'])); arrs['_gextra'] = np.array(int(ck.get('gextra', 0))); arrs['_ghist'] = np.array(int(ck.get('ghist', 0)))
    assert not ck.get('uattn') and not ck.get('gplan'), 'NpPolicy implements uhist but not uattn / gplan'
    if ck.get('uhist'): assert 'uh1.weight' in arrs and 'uh2.weight' in arrs; arrs['_uhist'] = np.array(1)
    import os as _os, json as _json
    _rf = ckpt_path + '.rules.json'
    if _os.path.exists(_rf):
        for k_, v_ in _json.load(open(_rf)).items(): arrs['rule_' + k_] = np.array(float(v_))
        if 'rule_consolidate' in arrs: arrs['rule_cons_tnat'] = np.array(float((ck.get('baked') or {}).get('temp', 1.0)))
    if ck.get('sched'): arrs['sched_ub'] = np.asarray(ck['sched']['ub'], np.float32); arrs['sched_kb'] = np.asarray(ck['sched']['kb'], np.float32)
    np.savez_compressed(out_npz, **arrs)

if __name__ == '__main__':
    import sys
    export(sys.argv[1], sys.argv[2]); print('exported', sys.argv[2])

class EnsemblePolicy:
    def __init__(self, paths, mode='sample', temp=1.0, salt=0):
        self.members = [NpPolicy(p, mode=mode, temp=temp) for p in paths]; self.mode, self.salt = mode, int(salt); self.m0 = self.members[0]; self.m0.temp = float(temp)

    @staticmethod
    def _lsm(z):
        z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def __call__(self, obs, cfg=None):
        rng = np.random.default_rng(obs_seed(obs) + self.salt * 1_000_003) if self.mode == 'sample' else None
        grid, units, xy, n, glob = E.encode(obs); um, mm, qmax = E.masks(obs)
        fs = [m.features(grid, glob) for m in self.members]
        uh = None
        if any(m.uhist for m in self.members):
            if not hasattr(self, '_uh'): self._uh = UnitHist()
            uh = self._uh.get(obs)
        ul = np.mean([self._lsm(np.where(um[:n], m.unit_logits(x, g, units, xy, n, uh), -1e9)) for m, (x, g) in zip(self.members, fs)], 0)
        u = C.fix_plant_budget(self.m0._pick(ul, rng).tolist(), obs['private'].get('seeds', {}))
        if uh is not None: self._uh.push(obs, u)
        kl = np.mean([self._lsm(np.where(mm[None], m.kind_logits(g), -1e9)) for m, (x, g) in zip(self.members, fs)], 0); kinds = self.m0._pick(kl, rng)
        qm = np.maximum(qmax[kinds], 1); ok = np.arange(C.MAX_Q)[None] < qm[:, None]
        ql = np.mean([self._lsm(np.where(ok, m.qty_logits(g, kinds), -1e9)) for m, (x, g) in zip(self.members, fs)], 0); qs = self.m0._pick(ql, rng) + 1
        return C.decode_action(u, list(zip(kinds.tolist(), qs.tolist())), n_hands=len(obs['farms'][int(obs['player'])]['hands']))

class SplitPolicy:
    def __init__(self, unit_npz, market_npz, merge='replace'):
        self.A = NpPolicy(unit_npz, mode='argmax'); self.B = NpPolicy(market_npz, mode='argmax'); self.merge = merge

    def _market(self, P, grid, glob, mm, qmax):
        x, g = P.features(grid, glob); kinds = np.where(mm[None], P.kind_logits(g), -1e9).argmax(-1)
        qm = np.maximum(qmax[kinds], 1); qs = np.where(np.arange(C.MAX_Q)[None] < qm[:, None], P.qty_logits(g, kinds), -1e9).argmax(-1) + 1
        return x, g, list(zip(kinds.tolist(), qs.tolist()))

    def __call__(self, obs, cfg=None):
        grid, units, xy, n, glob = E.encode(obs); um, mm, qmax = E.masks(obs)
        xa, ga, oa = self._market(self.A, grid, glob, mm, qmax)
        uh = None
        if self.A.uhist:
            if not hasattr(self, '_uh'): self._uh = UnitHist()
            uh = self._uh.get(obs)
        u = np.where(um[:n], self.A.unit_logits(xa, ga, units, xy, n, uh), -1e9).argmax(-1).tolist(); u = C.fix_plant_budget(u, obs['private'].get('seeds', {}))
        if uh is not None: self._uh.push(obs, u)
        _, _, ob = self._market(self.B, grid, glob, mm, qmax)
        if self.merge == 'replace':
            slots = ob
        else:
            used = {k for k, _ in oa if k != C.MK_NONE}; keep = [o for o in oa if o[0] != C.MK_NONE]
            slots = (keep + [o for o in ob if o[0] != C.MK_NONE and o[0] not in used])[:C.MAX_SLOTS]
        return C.decode_action(u, slots, n_hands=len(obs['farms'][int(obs['player'])]['hands']))
