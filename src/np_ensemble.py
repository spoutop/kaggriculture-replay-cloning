import json
import math
import numpy as np
import codec as C
import encoder as E
from np_policy import NpPolicy, UnitHist, grouped_argmax

def _lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))

def kind_name(k):
    kk = C.MK_KINDS[int(k)]
    return 'NONE' if kk is None else '_'.join(kk)

KIND_NAMES = [kind_name(k) for k in range(C.N_MK)]
PLANT_IDX = tuple(range(C.A_PLANT0, C.A_PLANT0 + len(C.CROPS)))
SERVICE_IDX = (C.A_WATER, C.A_HARVEST, C.A_FERT, C.A_FEED, C.A_CARE, C.A_COLLECT)
QTY_FAMILIES = ('SELL', 'BUY_PRODUCT', 'BUY_SEED', 'BUY_ANIMAL')

SHOP_DEMAND = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
               "YARN_STORE": ["WOOL"], "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
               "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
ANIMAL_PRODUCT = {'GOOSE': 'EGG', 'COW': 'MILK', 'SHEEP': 'WOOL'}

def town_demand(obs):
    d = dict.fromkeys(C.PRODUCTS, 0.0)
    for sname in ((obs.get('town') or {}).get('unlocked_shops') or []):
        prods = SHOP_DEMAND.get(sname, [])
        m = 2.0 if len(prods) == 1 else 1.0
        for p in prods: d[p] += m
    return d
_NEG = -1e8

class Correction:
    KEYS = ('format', 'meta', 'phase_days', 'kind_bias', 'crop_bias', 'plant2', 'svc_bias', 'qty_scale', 'town_bias', 'k3')

    def __init__(self, spec):
        if isinstance(spec, str):
            with open(spec) as f: spec = json.load(f)
        spec = dict(spec or {})
        bad = [k for k in spec if k not in self.KEYS]
        if bad: raise ValueError(f'unknown correction keys {bad}')
        self.spec = spec
        pd = [int(x) for x in spec.get('phase_days', [0, 5, 15, 25])]
        if not pd or pd[0] != 0 or any(b <= a for a, b in zip(pd, pd[1:])): raise ValueError(f'phase_days must start at 0 and increase: {pd}')
        self.phase_days = np.array(pd, np.int64); P = len(pd); self.n_phase = P

        def table(d, names, what):
            t = np.zeros((P, len(names)), np.float64)
            for nm, v in (d or {}).items():
                if nm not in names: raise ValueError(f'unknown {what} name {nm!r}')
                v = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v] * P)]
                if len(v) != P: raise ValueError(f'{what} {nm}: {len(v)} values for {P} phases')
                t[:, names.index(nm)] = v
            return t
        self.kb = table(spec.get('kind_bias'), KIND_NAMES, 'kind_bias')
        self.cb = table(spec.get('crop_bias'), list(C.CROPS), 'crop_bias')
        self.plant2 = float(spec.get('plant2', 0.0))
        if self.plant2 < 0: raise ValueError('plant2 must be >= 0')
        sb = spec.get('svc_bias', 0.0); sb = [float(x) for x in (sb if isinstance(sb, (list, tuple)) else [sb] * P)]
        if len(sb) != P: raise ValueError(f'svc_bias: {len(sb)} values for {P} phases')
        self.sb = np.array(sb, np.float64)
        qf = table(spec.get('qty_scale'), list(QTY_FAMILIES), 'qty_scale')
        fam = np.full(C.N_MK, -1, np.int64)
        for k in range(C.N_MK):
            kk = C.MK_KINDS[k]
            if kk is not None and kk[0] in QTY_FAMILIES: fam[k] = QTY_FAMILIES.index(kk[0])
        self.qk = np.where(fam[None, :] >= 0, qf[:, np.maximum(fam, 0)], 0.0)
        tb = dict(spec.get('town_bias') or {})
        badtb = [k for k in tb if k not in ('seed', 'plant', 'animal', 'sell')]
        if badtb: raise ValueError(f'unknown town_bias groups {badtb}')
        self.tk = np.zeros(C.N_MK, np.float64); self.tk_prod = [None] * C.N_MK; self.tc = np.zeros(len(C.CROPS), np.float64)
        for grp, names, kindof, prodof in (('seed', list(C.CROPS), lambda x: ('BUY_SEED', x), lambda x: x),
                                           ('animal', list(C.ANIMALS), lambda x: ('BUY_ANIMAL', x), lambda x: ANIMAL_PRODUCT[x]),
                                           ('sell', [p for p in C.PRODUCTS if p != 'FERTILIZER'], lambda x: ('SELL', x), lambda x: x)):
            for nm, v in (tb.get(grp) or {}).items():
                if nm not in names: raise ValueError(f'unknown town_bias {grp} name {nm!r}')
                k = C.MK_INDEX[kindof(nm)]; self.tk[k] = float(v); self.tk_prod[k] = prodof(nm)
        for nm, v in (tb.get('plant') or {}).items():
            if nm not in C.CROPS: raise ValueError(f'unknown town_bias plant crop {nm!r}')
            self.tc[C.CROPS.index(nm)] = float(v)
        self.town_on = bool(np.any(self.tk != 0) or np.any(self.tc != 0))
        k3 = spec.get('k3', False)
        if isinstance(k3, dict):
            bad3 = [k for k in k3 if k not in ('sell_step', 'unit_from')]
            if bad3: raise ValueError(f'unknown k3 keys {bad3}')
            self.k3_on = True; self.k3_sell = int(k3.get('sell_step', 718)); self.k3_unit = int(k3.get('unit_from', 716))
        else:
            if k3 not in (True, False, 0, 1): raise ValueError(f'k3 must be a bool or a dict, got {k3!r}')
            self.k3_on = bool(k3); self.k3_sell, self.k3_unit = 718, 716
        self.kind_on = bool(np.any(self.kb != 0) or np.any(self.tk != 0)); self.crop_on = bool(np.any(self.cb != 0) or self.plant2 > 0 or np.any(self.tc != 0))
        self.svc_on = bool(np.any(self.sb != 0)); self.qty_on = bool(np.any(self.qk != 0))
        self.active = self.kind_on or self.crop_on or self.svc_on or self.qty_on or self.town_on or self.k3_on
        self.stats = dict(steps=0, kind_steps=0, crop_units=0, svc_units=0, qty_orders=0, k3_units=0, k3_sold=0)

    def phase(self, obs):
        day = min(int(obs.get('step', 0)) // 24, 29)
        return int(np.searchsorted(self.phase_days, day, side='right') - 1)

    def to_dict(self):
        P = self.n_phase; d = dict(format='bes_correction_v1', phase_days=[int(x) for x in self.phase_days])
        kb = {KIND_NAMES[k]: [float(x) for x in self.kb[:, k]] for k in range(C.N_MK) if np.any(self.kb[:, k] != 0)}
        cb = {c: [float(x) for x in self.cb[:, i]] for i, c in enumerate(C.CROPS) if np.any(self.cb[:, i] != 0)}
        if kb: d['kind_bias'] = kb
        if cb: d['crop_bias'] = cb
        if self.plant2 > 0: d['plant2'] = self.plant2
        if np.any(self.sb != 0): d['svc_bias'] = [float(x) for x in self.sb]
        qs = {}
        for f in QTY_FAMILIES:
            ks = [k for k in range(C.N_MK) if C.MK_KINDS[k] is not None and C.MK_KINDS[k][0] == f]
            if np.any(self.qk[:, ks[0]] != 0): qs[f] = [float(x) for x in self.qk[:, ks[0]]]
        if qs: d['qty_scale'] = qs
        if self.town_on:
            t = {}
            for k in range(C.N_MK):
                if self.tk[k] != 0:
                    grp = {'BUY_SEED': 'seed', 'BUY_ANIMAL': 'animal', 'SELL': 'sell'}[C.MK_KINDS[k][0]]
                    t.setdefault(grp, {})[C.MK_KINDS[k][1]] = float(self.tk[k])
            for i, c in enumerate(C.CROPS):
                if self.tc[i] != 0: t.setdefault('plant', {})[c] = float(self.tc[i])
            d['town_bias'] = t
        if self.k3_on: d['k3'] = {'sell_step': self.k3_sell, 'unit_from': self.k3_unit}
        if 'meta' in self.spec: d['meta'] = self.spec['meta']
        assert P == len(d['phase_days'])
        return d

    def units(self, ul, um, u, seeds, ph, obs=None):
        out = list(u)
        if self.crop_on:
            cb = self.cb[ph]; g = self.plant2; changed = False
            if self.town_on and np.any(self.tc != 0):
                dem = town_demand(obs); cb = cb + self.tc * np.array([dem[c] for c in C.CROPS], np.float64)
            for i, a in enumerate(out):
                if not (C.A_PLANT0 <= a < C.A_PLANT0 + len(C.CROPS)): continue
                legal = [x for x in PLANT_IDX if um[i, x] and ul[i, x] > _NEG]
                if not legal: continue
                sc = sorted(((-(float(ul[i, x]) + float(cb[x - C.A_PLANT0])), x) for x in legal))
                a1 = sc[0][1]
                if g > 0 and len(sc) > 1 and (-sc[0][0]) - (-sc[1][0]) < g: a1 = sc[1][1]
                if a1 != a: out[i] = int(a1); changed = True; self.stats['crop_units'] += 1
            if changed: out = C.fix_plant_budget(out, seeds)
        if self.svc_on and self.sb[ph] != 0:
            b = float(self.sb[ph])
            for i, a in enumerate(out):
                if a != C.A_PASS and a not in SERVICE_IDX: continue
                legal = [x for x in SERVICE_IDX if um[i, x] and ul[i, x] > _NEG]
                if not legal: continue
                best = max(legal, key=lambda x: (float(ul[i, x]), -x))
                new = best if float(ul[i, best]) + b > float(ul[i, C.A_PASS]) else C.A_PASS
                if new != a: out[i] = int(new); self.stats['svc_units'] += 1
        return out

    def kinds(self, kl, ph, obs=None):
        if not self.kind_on: return kl
        b = self.kb[ph]
        if self.town_on and np.any(self.tk != 0):
            dem = town_demand(obs); b = b + self.tk * np.array([dem[p] if p is not None else 0.0 for p in self.tk_prod], np.float64)
        if not np.any(b != 0): return kl
        self.stats['kind_steps'] += 1
        return kl + b[None, :]

    def k3_apply(self, obs, act):
        step = int(obs.get('step', 0))
        if not self.k3_on or step < self.k3_unit: return act
        me = int(obs['player']); farm = obs['farms'][me]; priv = obs['private']
        pos = [farm['farmer']] + list(farm['hands']); invs = priv.get('inventories', []) or []
        units = [list(act.get('farmer') or ['PASS'])] + [list(h) for h in (act.get('hands') or [])]
        dropped = {}
        for i in range(min(len(pos), len(units))):
            inv = invs[i] if i < len(invs) else {}
            if not any(int(v) > 0 for v in inv.values()): continue
            x, y = int(pos[i][0]), int(pos[i][1])
            if (x, y) in C.ACCESS:
                units[i] = ['DROP']
                for k, v in inv.items():
                    if int(v) > 0 and k in C.PRODUCTS: dropped[k] = dropped.get(k, 0) + int(v)
            else:
                tx, ty = min(C.ACCESS, key=lambda t: (abs(t[0] - x) + abs(t[1] - y), t))
                if tx != x: units[i] = ['EAST'] if tx > x else ['WEST']
                else: units[i] = ['SOUTH'] if ty > y else ['NORTH']
            self.stats['k3_units'] += 1
        out = dict(act); out['farmer'] = units[0]; out['hands'] = units[1:]
        if step >= self.k3_sell:
            shed = priv.get('shed', {}) or {}; prices = (obs.get('market') or {}).get('prices', {}) or {}
            market = [list(m) if m else [] for m in (act.get('market') or [])]
            selling = {}
            for m in market:
                if m and m[0] == 'SELL' and len(m) >= 3: selling[m[1]] = selling.get(m[1], 0) + int(m[2])
            need = []
            for p in C.PRODUCTS:
                have = int(shed.get(p, 0)) + dropped.get(p, 0) - selling.get(p, 0)
                if have > 0: need.append((-(float(prices.get(p, 0.0)) * have), p, have))
            for _, p, have in sorted(need):
                while have > 0 and len(market) < C.MAX_SLOTS:
                    q = min(have, C.MAX_Q); market.append(['SELL', p, q]); have -= q; self.stats['k3_sold'] += q
            out['market'] = market
        return out

    def qty(self, kinds, qs, qmax, ph):
        if not self.qty_on or not np.any(self.qk[ph] != 0): return qs
        qs = np.array(qs).copy(); s = self.qk[ph]
        for i, k in enumerate(np.asarray(kinds).tolist()):
            if k != C.MK_NONE and s[k] != 0:
                q = int(min(max(1, int(round(float(qs[i]) * math.exp(float(s[k]))))), max(1, int(qmax[k]))))
                if q != int(qs[i]): qs[i] = q; self.stats['qty_orders'] += 1
        return qs

def load_correction(spec):
    if spec is None: return None
    c = spec if isinstance(spec, Correction) else Correction(spec)
    return c if c.active else None

def spec_correction(spec, spec_path):
    c = spec.get('correction')
    if c is None or isinstance(c, dict): return c
    import os
    return c if os.path.isabs(c) else os.path.join(os.path.dirname(spec_path), c)

class EnsNpPolicy:
    def __init__(self, paths, weights=None, correction=None):
        self.members = [NpPolicy(p, mode='argmax', temp=1.0, team=0, salt=0) for p in paths]
        w = np.ones(len(paths)) if weights is None else np.asarray(weights, np.float64)
        self.w = w / w.sum()
        W0 = self.members[0].w
        self.tau = float(W0['rule_consolidate']) if 'rule_consolidate' in W0 else 0.0
        self.tnat = float(W0['rule_cons_tnat']) if 'rule_cons_tnat' in W0 else 1.0
        self.overflow = 'rule_overflow' in W0 and float(W0['rule_overflow']) > 0
        for m in self.members[1:]:
            for k in ('rule_consolidate', 'rule_cons_tnat', 'rule_overflow', 'rule_unit_group'):
                assert (k in m.w) == (k in W0) and (k not in W0 or float(m.w[k]) == float(W0[k])), f'member decode rule {k} differs'
            assert 'sched_ub' not in m.w and 'rule_hazard' not in m.w, 'unsupported member rule'
        assert 'sched_ub' not in W0 and 'rule_hazard' not in W0, 'unsupported member rule'
        self.uhist = any(m.uhist for m in self.members); self._uh = UnitHist()
        self.ugroup = 'rule_unit_group' in W0 and float(W0['rule_unit_group']) > 0
        self.corr = load_correction(correction)
        self.last_u = self.last_ul = self.last_kl = None

    def __call__(self, obs, cfg=None):
        return self.act(obs)

    def act(self, obs):
        grid, units, xy, n, glob = E.encode(obs); um, mm, qmax = E.masks(obs)
        feats = []
        for m in self.members:
            if not hasattr(m, '_hist'): m._hist = E.InvHist()
            feats.append(m.features(grid, glob, m._hist.push(obs)))
        tn = self.tnat
        uh = self._uh.get(obs) if self.uhist else None
        ul = sum(w * _lsm(np.where(um[:n], m.unit_logits(x, g, units, xy, n, uh) * tn, -1e9)) for w, m, (x, g) in zip(self.w, self.members, feats))
        u = C.fix_plant_budget((grouped_argmax(ul, 1.0) if self.ugroup else ul.argmax(-1)).tolist(), obs['private'].get('seeds', {}))
        cr = self.corr
        if cr is not None:
            ph = cr.phase(obs); cr.stats['steps'] += 1
            u = cr.units(ul, um[:n], u, obs['private'].get('seeds', {}), ph, obs)
        if uh is not None: self._uh.push(obs, u)
        kl = sum(w * _lsm(np.where(mm[None], m.kind_logits(g) * tn, -1e9)) for w, m, (x, g) in zip(self.w, self.members, feats))
        self.last_u, self.last_ul, self.last_kl = list(u), ul, kl
        if cr is not None: kl = cr.kinds(kl, ph, obs)
        kinds = kl.argmax(-1)
        if self.tau > 0:
            z = kl.astype(np.float64); z = z - z.max(-1, keepdims=True); p_ = np.exp(z); p_ = p_ / p_.sum(-1, keepdims=True)
            any_ = 1.0 - np.prod(1.0 - p_, axis=0); any_[C.MK_NONE] = 0.0; kinds = np.array(kinds).copy()
            for kind in np.argsort(-any_, kind='stable').tolist():
                if any_[kind] > self.tau and mm[kind] and kind not in kinds:
                    real = np.nonzero(kinds != C.MK_NONE)[0]; nxt = int(real.max()) + 1 if len(real) else 0
                    if nxt < len(kinds): kinds[nxt] = kind
        qm = np.maximum(qmax[kinds], 1); ok = np.arange(C.MAX_Q)[None] < qm[:, None]
        ql = sum(w * _lsm(np.where(ok, m.qty_logits(g, kinds) * tn, -1e9)) for w, m, (x, g) in zip(self.w, self.members, feats))
        qs = ql.argmax(-1) + 1
        if cr is not None: qs = cr.qty(kinds, qs, qmax, ph)
        act_ = C.decode_action(u, list(zip(kinds.tolist(), qs.tolist())), n_hands=len(obs['farms'][int(obs['player'])]['hands']))
        if self.overflow: act_ = C.overflow_sale(obs, act_)
        if cr is not None and cr.k3_on: act_ = cr.k3_apply(obs, act_)
        return act_

DET_STEPS = (1, 2, 4, 8, 12, 16, 20, 24)
DET_PRODUCTS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
DET_TILES = ('PLANT:WHEAT', 'PLANT:CARROT', 'PLANT:TOMATO', 'PLANT:STRAWBERRY', 'PLANT:MELON', 'COOP', 'PASTURE:COW', 'PASTURE:SHEEP', 'WEED')

def det_feature_names(K):
    n = []
    for s in DET_STEPS:
        if s <= K: n += [f'opp_money_k@{s}', f'opp_hands@{s}']
    return n + [f'opp_quadrants@{K}'] + [f'opp_tile:{t}@{K}' for t in DET_TILES] + [f'mkt_taken:{p}@{K}' for p in DET_PRODUCTS] + [f'price_h:{p}@{K}' for p in DET_PRODUCTS]

def _det_get(o, k):
    return o[k] if isinstance(o, dict) else getattr(o, k)

class DetRecorder:
    def __init__(self, K):
        self.K = int(K); self.snap = {}

    def record(self, obs):
        t = int(_det_get(obs, 'step')); me = int(_det_get(obs, 'player')); f = _det_get(obs, 'farms')[1 - me]
        if t in DET_STEPS and t <= self.K: self.snap[('m', t)] = (float(f['money']) / 1000.0, float(len(f['hands'] or [])))
        if t == self.K:
            tiles = {}
            for row in f['tiles']:
                for x in row:
                    if isinstance(x, dict):
                        k = x.get('kind'); key = f"{k}:{x.get('crop') or x.get('animal')}" if k in ('PLANT', 'PASTURE') else k; tiles[key] = tiles.get(key, 0) + 1
            mk = _det_get(obs, 'market'); inv = mk['inventory']; pr = mk['prices']
            self.snap['K'] = ([float(len(f['unlocked_quadrants'] or []))] + [float(tiles.get(t, 0)) for t in DET_TILES]
                              + [(10000.0 - float(inv.get(p, 10000))) / 10.0 for p in DET_PRODUCTS] + [float(pr.get(p, 0)) / 100.0 for p in DET_PRODUCTS])

    def vector(self):
        v = []
        for s in DET_STEPS:
            if s <= self.K: v += list(self.snap.get(('m', s), (0.0, 0.0)))
        return v + list(self.snap['K'])

def load_detector(path):
    import json as _j
    d = _j.load(open(path)); d.setdefault('threshold', 0.5); d.setdefault('pre', 'a')
    if 'town_rule' in d:
        assert set(d['town_rule']) == {'shop', 'k'}; return d
    if 'money_rule' in d:
        assert set(d['money_rule']) == {'m1_min', 'm2_lo', 'm2_hi'} and int(d['K']) == 2; return d
    assert list(d['names']) == det_feature_names(d['K']), 'detector feature names do not match det_feature_names(K)'
    assert len(d['w']) == len(d['names']); return d

class DetectorSwitch:
    def __init__(self, a, b, det):
        self.a, self.b, self.det = a, b, det; self.reset()

    def reset(self):
        self.rec = DetRecorder(self.det['K']); self.choice = None; self.p = None; self.cur = self.a if self.det['pre'] == 'a' else self.b

    def pick(self, obs):
        t = int(_det_get(obs, 'step'))
        if t < 1 and self.choice is not None: self.reset()
        self.rec.record(obs)
        if t >= self.det['K'] and self.choice is None and 'town_rule' in self.det:
            tr = self.det['town_rule']; tw = _det_get(obs, 'town'); shops = (tw.get('unlocked_shops') if isinstance(tw, dict) else getattr(tw, 'unlocked_shops', None)) or []
            self.p = float(sum(1 for x in shops if x == tr['shop'])); self.choice = 'b' if self.p >= tr['k'] else 'a'
        if t >= self.det['K'] and self.choice is None and 'money_rule' in self.det:
            mr = self.det['money_rule']; m1 = round(1000.0 * self.rec.snap.get(('m', 1), (0.0, 0.0))[0]); m2 = round(1000.0 * self.rec.snap.get(('m', 2), (0.0, 0.0))[0])
            self.p = 1.0 if (m1 >= mr['m1_min'] and mr['m2_lo'] <= m2 <= mr['m2_hi']) else 0.0; self.choice = 'b' if self.p >= 0.5 else 'a'
        if t >= self.det['K'] and self.choice is None:
            import math
            z = sum(w * x for w, x in zip(self.det['w'], self.rec.vector())) + float(self.det['b'])
            self.p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z)))); self.choice = 'b' if self.p >= self.det['threshold'] else 'a'
        self.cur = (self.b if self.choice == 'b' else self.a) if self.choice else (self.a if self.det['pre'] == 'a' else self.b)
        return self.cur

    def __call__(self, obs, cfg=None):
        return self.pick(obs)(obs, cfg)

    last_u = property(lambda self: getattr(self.cur, 'last_u', None))
    last_ul = property(lambda self: getattr(self.cur, 'last_ul', None))
    last_kl = property(lambda self: getattr(self.cur, 'last_kl', None))
