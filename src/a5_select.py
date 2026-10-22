import argparse, collections, glob, json, math, os, statistics as st, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
R1_DEFAULT = os.environ.get('KGR_R1_BUNDLE', 'r1')
KGX = os.environ.get('KGX_ROOT', 'kgx')

def export_ckpt(ck, dst, team=5, temp=0.5, overflow=1.0, consolidate=0.35, unit_group=0):
    import numpy as np, torch
    import codec as C
    c = torch.load(ck, map_location='cpu', weights_only=False)
    assert c.get('codec') == C.CODEC_TAG, (ck, c.get('codec'), C.CODEC_TAG)
    assert not c.get('uattn') and not c.get('gplan'), ('NpPolicy implements uhist but not uattn / gplan', ck)
    s = {k: v.clone().float() for k, v in c['state'].items() if k != 'coords'}
    s['temb.weight'][0] = s['temb.weight'][team].clone()
    for head, last in (('uhead', 4), ('khead', 2), ('qhead', 2)):
        for p in ('weight', 'bias'): s[f'{head}.{last}.{p}'] = s[f'{head}.{last}.{p}'] / temp
    arrs = {k: v.numpy() for k, v in s.items()}
    uh = int(c.get('uhist') or 0); assert (('uh1.weight' in arrs) and ('uh2.weight' in arrs)) == bool(uh), (ck, uh)
    arrs['_blocks'] = np.array(c.get('blocks', 8)); arrs['_ch'] = np.array(c.get('ch', 128)); arrs['_uhist'] = np.array(uh)
    arrs['rule_overflow'] = np.array(float(overflow)); arrs['rule_consolidate'] = np.array(float(consolidate)); arrs['rule_cons_tnat'] = np.array(float(temp))
    if unit_group: arrs['rule_unit_group'] = np.array(1.0)
    arrs['_codec'] = np.array(C.CODEC_TAG); arrs['_src'] = np.array(os.path.abspath(ck)); arrs['_team'] = np.array(team)
    assert arrs['uhead.4.weight'].shape[0] == C.N_UNIT and arrs['qhead.2.weight'].shape[0] == C.MAX_Q
    np.savez_compressed(dst, **arrs); print('exported', dst, f'uhist {uh}', flush=True)
    return dst

def export(runs, team=5, temp=0.5, overflow=1.0, consolidate=0.35, ep_range=None, eps=None, unit_group=0):
    out = []
    for run in runs:
        os.makedirs(os.path.join(run, 'np'), exist_ok=True)
        for ck in sorted(glob.glob(os.path.join(run, 'ck_epoch*.pt'))):
            ep = int(os.path.basename(ck)[8:10])
            if ep_range and not (ep_range[0] <= ep < ep_range[1]): continue
            if eps is not None and ep not in eps: continue
            dst = os.path.join(run, 'np', os.path.basename(ck)[:-3] + f'_tok{team}_t{temp}_k2c{int(round(consolidate * 100))}' + ('_ug' if unit_group else '') + '.npz')
            if os.path.exists(dst): out.append(dst); continue
            out.append(export_ckpt(ck, dst, team, temp, overflow, consolidate, unit_group))
    return out

_R1 = {}

def _r1_policy(r1_dir):
    if r1_dir not in _R1:
        if r1_dir not in sys.path: sys.path.append(r1_dir)
        mods = [f[:-3] for f in os.listdir(r1_dir) if f.endswith('_np_policy.py')]
        assert len(mods) == 1, mods
        import importlib
        _R1[r1_dir] = importlib.import_module(mods[0])
    m = _R1[r1_dir]
    return m.NpPolicy(m.weights_path(), mode='argmax', temp=1.0, team=0, salt=0)

ROSTER_DEFAULT = os.environ.get('A5_ROSTER', 'roster/agents')

def _opp_policy(name, r1_dir):
    if name == 'R1': return _r1_policy(r1_dir)
    if name.startswith('npz:'):
        import np_policy as NP
        return NP.NpPolicy(name[4:], mode='argmax', temp=1.0, team=0, salt=0)
    if name.startswith('ens:'):
        from np_ensemble import EnsNpPolicy, spec_correction
        spec = json.load(open(name[4:])); d = os.path.dirname(name[4:])
        return EnsNpPolicy([m if os.path.isabs(m) else os.path.join(d, m) for m in spec['members']], spec.get('weights'), spec_correction(spec, name[4:]))
    for p in (KGX, KGX + '/env', KGX + '/arena', KGX + '/league'):
        if p not in sys.path: sys.path.append(p)
    import roster
    from pathlib import Path
    extra = Path(os.environ.get('A5_ROSTER', ROSTER_DEFAULT))
    if extra.exists() and extra not in roster.ROSTER_DIRS:
        roster.ROSTER_DIRS.append(extra); roster._discovery_cache = None
    return roster.load_agent(name, fresh=True)

def _play(task):
    pol_path, seed, seat, opp, r1_dir = task
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, '1')
    import numpy as np
    from lean_engine import LeanGame
    import np_policy as NP, codec as C
    if pol_path.endswith('.ens.json'):
        from np_ensemble import EnsNpPolicy, spec_correction
        spec = json.load(open(pol_path)); mem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(pol_path), m) for m in spec['members']]
        for m in mem: assert str(np.load(m)['_codec']) == C.CODEC_TAG, m
        A = EnsNpPolicy(mem, spec.get('weights'), spec_correction(spec, pol_path))
    else:
        z = np.load(pol_path); assert str(z['_codec']) == C.CODEC_TAG
        _m = os.environ.get('A5_MODE', 'argmax'); _t = float(os.environ.get('A5_TEMP', '1.0'))
        A = NP.NpPolicy(pol_path, mode=_m, temp=_t, team=0, salt=seed * 2 + seat if _m == 'sample' else 0)
    B = _opp_policy(opp, r1_dir)
    g = LeanGame(seed); cfg = g.env.configuration; t0 = time.time(); worst = 0.0; errors = 0
    while not g.done:
        oa = g.obs(seat); ob = g.obs(1 - seat)
        t1 = time.time()
        try:
            aa = A(oa, cfg)
        except Exception as e:
            errors += 1; aa = {'farmer': ['PASS'], 'hands': [], 'market': []}
            if errors == 1: print('POLICY ERROR', pol_path, seed, seat, repr(e)[:300], flush=True)
        worst = max(worst, time.time() - t1)
        ab = B(ob, cfg)
        acts = [None, None]; acts[seat] = aa; acts[1 - seat] = ab; g.step(acts[0], acts[1])
    mine, theirs = g.rewards[seat], g.rewards[1 - seat]
    return dict(pol=pol_path, opp=opp, seed=seed, seat=seat, mine=mine, theirs=theirs, win=1.0 if mine > theirs else (0.5 if mine == theirs else 0.0),
                secs=round(time.time() - t0, 1), worst_turn_ms=round(worst * 1000, 1), errors=errors)

def play(pols, seeds, opps, r1_dir, workers, out):
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(p, s, seat, o, r1_dir) for s in seeds for seat in (0, 1) for o in opps for p in pols]
    print(f'{len(tasks)} games: {len(pols)} policies x {len(seeds)} seeds x 2 seats x {len(opps)} opponents; workers {workers}', flush=True)
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_play, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 50 == 0: print(f'{i + 1}/{len(tasks)} {time.time() - t0:.0f}s', flush=True)
    json.dump(dict(rows=rows, seeds=[seeds[0], seeds[-1] + 1], opps=opps, r1=r1_dir, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(out, 'w'))
    return rows

def report(rows, top=0):
    by = collections.defaultdict(list)
    for r in rows: by[(r['pol'], r['opp'])].append(r)
    res = []
    for (p, o), v in by.items():
        seeds = collections.defaultdict(list)
        for r in v: seeds[r['seed']].append(r['win'])
        means = [sum(x) / len(x) for x in seeds.values()]
        m = sum(r['win'] for r in v) / len(v); se = (st.stdev(means) / math.sqrt(len(means))) if len(means) > 1 else float('nan')
        res.append(dict(pol=p, opp=o, n=len(v), seeds=len(means), win=m, lo=m - 1.96 * se, hi=m + 1.96 * se, margin=st.mean(r['mine'] - r['theirs'] for r in v),
                        cash=st.mean(r['mine'] for r in v), opp_cash=st.mean(r['theirs'] for r in v), errors=sum(r['errors'] for r in v),
                        worst_turn_ms=max(r['worst_turn_ms'] for r in v)))
    res.sort(key=lambda d: (d['opp'], -d['win'], -d['margin']))
    for d in (res[:top] if top else res):
        print(f"{os.path.basename(os.path.dirname(os.path.dirname(d['pol'])))}/{os.path.basename(d['pol'])[:24]:<24s} vs {d['opp']:<10s} n={d['n']:4d} seeds={d['seeds']:3d} "
              f"win {d['win']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] margin {d['margin']:+9,.0f} cash {d['cash']:9,.0f} opp {d['opp_cash']:9,.0f} err {d['errors']} worst {d['worst_turn_ms']:.0f}ms")
    return res

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export'); e.add_argument('--runs', nargs='+', required=True); e.add_argument('--team', type=int, default=5); e.add_argument('--temp', type=float, default=0.5)
    e.add_argument('--consolidate', type=float, default=0.35); e.add_argument('--overflow', type=float, default=1.0); e.add_argument('--epochs', default='')
    e.add_argument('--eps', default='', help='comma list of epochs to export (e.g. 19,12): only these')
    e.add_argument('--unit-group', type=int, default=0, help='1 = write rule_unit_group (grouped unit decoding; file suffix _ug); default off')
    p = sub.add_parser('play'); p.add_argument('--pols', nargs='+', required=True); p.add_argument('--r1', default=R1_DEFAULT); p.add_argument('--seeds', required=True)
    p.add_argument('--opps', default='R1', help="comma list: R1 and/or public agent names (A5_ROSTER dir or the kgx roster); FRONTIER17 = the 17 agents of the 09-23 tournament")
    p.add_argument('--workers', type=int, default=16); p.add_argument('--out', required=True)
    r = sub.add_parser('report'); r.add_argument('file'); r.add_argument('--top', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'export':
        er = tuple(int(x) for x in a.epochs.split(':')) if a.epochs else None
        export(a.runs, a.team, a.temp, a.overflow, a.consolidate, er, {int(x) for x in a.eps.split(',')} if a.eps else None, a.unit_group)
    elif a.cmd == 'play':
        pols = sorted({f for g_ in a.pols for f in glob.glob(g_)}); lo, hi = map(int, a.seeds.split(':'))
        opps = []
        for o in a.opps.split(','):
            if o == 'FRONTIER17': opps += sorted(os.listdir(os.environ.get('A5_ROSTER', ROSTER_DEFAULT)))
            else: opps.append(o)
        rows = play(pols, list(range(lo, hi)), opps, a.r1, a.workers, a.out); report(rows)
    else:
        report(json.load(open(a.file))['rows'], a.top)

if __name__ == '__main__':
    main()
