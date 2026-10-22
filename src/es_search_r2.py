import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import es_search as S

SEL_LO, SEL_HI = 32_392_000, 32_398_000
FV2 = ['npz:runs/field_v2/mm.npz', 'npz:runs/field_v2/tv.npz', 'npz:runs/field_v2/tm.npz', 'npz:runs/field_v2/td.npz',
       'npz:runs/field_v2/umg.npz', 'npz:runs/field_v2/ref.npz']
PHASE_DAYS_R2 = [0, 5, 15, 18, 24, 28]
EARLY, LATE, END, END4, MEL = (0, 1), (2, 3, 4, 5), (4, 5), (4,), (1, 2)
TOWN_SCALE = 0.25
SPACE_R2 = ([(c, n, EARLY if g == (0, 1) else LATE) if c in ('kind',) else (c, n, (0, 1, 2, 3, 4, 5)) for c, n, g in S.SPACE_V1]
            + [('svc', 'ALL', END), ('kind', 'BUY_SEED_CARROT', END4), ('crop', 'CARROT', END4), ('kind', 'BUY_SEED_MELON', MEL), ('crop', 'MELON', MEL)]
            + [('town', 'seed', c) for c in ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY')] + [('town', 'plant', c) for c in ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY')]
            + [('town', 'animal', a) for a in ('GOOSE', 'COW', 'SHEEP')] + [('town', 'sell', p) for p in ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'EGG', 'MILK', 'WOOL')])

def theta_to_corr_r2(theta, space=SPACE_R2, k3=False, extra=None):
    P = len(PHASE_DAYS_R2); kb, cb, tb = {}, {}, {}; sb = [0.0] * P
    for v, (comp, nm, phs) in zip(theta, space):
        v = float(v)
        if comp == 'kind':
            row = kb.setdefault(nm, [0.0] * P)
            for p in phs: row[p] += v
        elif comp == 'crop':
            row = cb.setdefault(nm, [0.0] * P)
            for p in phs: row[p] += v
        elif comp == 'svc':
            for p in phs: sb[p] += v
        elif comp == 'town':
            if v != 0: tb.setdefault(nm, {})[phs] = round(v * TOWN_SCALE, 6)
    d = dict(format='bes_correction_v1', phase_days=PHASE_DAYS_R2)
    kb = {k: [round(x, 6) for x in r] for k, r in kb.items() if any(x != 0 for x in r)}
    cb = {k: [round(x, 6) for x in r] for k, r in cb.items() if any(x != 0 for x in r)}
    if kb: d['kind_bias'] = kb
    if cb: d['crop_bias'] = cb
    if any(x != 0 for x in sb): d['svc_bias'] = [round(x, 6) for x in sb]
    if tb: d['town_bias'] = tb
    if k3: d['k3'] = True
    if extra: d.update(extra)
    return d

def template_r2(n_seeds, seed0):
    cells = []
    for i in range(n_seeds):
        opp = FV2[(seed0 + i) % len(FV2)]
        for seat in (0, 1):
            cells.append((i, seat, 'E3')); cells.append((i, seat, opp))
    return cells

def make_cells_r2(seed0, n_seeds, who, lo=SEL_LO, hi=SEL_HI):
    out = []
    for off, seat, opp in template_r2(n_seeds, seed0):
        s = seed0 + off
        assert lo <= s < hi, f'seed {s} outside the round-2 block'
        if who == 'E3' and opp == 'E3': continue
        out.append(dict(seed=s, seat=seat, a=who, b=opp))
    return out

def paired_r2(rows, name, base):
    r = S.paired(rows, name, base)
    return r

def run_screen(a, pool, log, out):
    cands_all = json.load(open(a.cands)); alive = list(cands_all); rnd = 0; seed0 = a.screen_seed0; allrows = []; seeds_r = a.seeds_per_round
    while True:
        assert seed0 + seeds_r <= 32_393_000, 'round-2 screen seeds must stay in 32,392,000-32,392,999'
        cells = make_cells_r2(seed0, seeds_r, 'E3')
        for nm in alive: cells += make_cells_r2(seed0, seeds_r, 'cand:' + nm)
        rows = pool.run(cells, {nm: cands_all[nm] for nm in alive}, log); allrows += rows
        base = S.base_of(allrows)
        res = sorted((paired_r2(allrows, nm, base) for nm in alive), key=lambda r: (-r['dwr'], -r['margin']))
        with open(os.path.join(out, 'screen_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(round=rnd, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seeds=[seed0, seed0 + seeds_r], res=res,
                                    stats={nm: S._mean_stats(allrows, 'cand:' + nm) for nm in alive})) + '\n')
        json.dump(dict(rows=allrows), open(os.path.join(out, 'screen_rows.json'), 'w'))
        log(f'round {rnd}: seeds {seed0}-{seed0 + seeds_r - 1}, {len(alive)} alive')
        for r in res: log(f"   {r['name']:<28s} dWR {r['dwr']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] h2h {r['h2h']:+.4f} field {r['panel']:+.4f} n {r['n']} margin {r['margin']:+.0f}")
        seed0 += seeds_r; rnd += 1
        if len(alive) <= a.keep_final or rnd >= a.max_rounds: break
        alive = [r['name'] for r in res[:max(a.keep_final, len(alive) // 2)]]; seeds_r *= 2

def run_es(a, pool, log, out):
    space = SPACE_R2; D = len(space); n = a.n
    st_path = os.path.join(out, 'es_state.json')
    if os.path.exists(st_path):
        st = json.load(open(st_path)); theta = np.array(st['theta']); mom = np.array(st['mom']); k0 = st['gen'] + 1
        assert [tuple(x) if not isinstance(x[2], list) else (x[0], x[1], tuple(x[2])) for x in st['space']] == [tuple(x) for x in space], 'resume: space mismatch'
        log(f'resume at generation {k0}')
    else:
        theta = np.zeros(D); mom = np.zeros(D); k0 = 0
    json.dump(dict(space=space, args=vars(a)), open(os.path.join(out, 'space.json'), 'w'), indent=1)
    mon_base = None; mon_path = os.path.join(out, 'monitor_base.json')
    tc = lambda th, extra=None: theta_to_corr_r2(th, space, bool(a.k3), extra)
    for k in range(k0, a.gens):
        t0 = time.time()
        rng = np.random.default_rng([20260925, k]); eps = rng.standard_normal((n, D))
        cands = {'c': tc(theta)}
        for i in range(n):
            cands[f'p{i}'] = tc(theta + a.sigma * eps[i]); cands[f'm{i}'] = tc(theta - a.sigma * eps[i])
        seed0 = a.gen_seed0 + k * a.seeds_per_gen
        assert a.gen_seed0 >= 32_393_000 and seed0 + a.seeds_per_gen <= 32_396_000, 'round-2 ES seeds must stay in 32,393,000-32,395,999'
        cells = make_cells_r2(seed0, a.seeds_per_gen, 'E3')
        for nm in cands: cells += make_cells_r2(seed0, a.seeds_per_gen, 'cand:' + nm)
        rows = pool.run(cells, cands, log)
        base = S.base_of(rows)
        res = {nm: paired_r2(rows, nm, base) for nm in cands}
        names = [f'p{i}' for i in range(n)] + [f'm{i}' for i in range(n)]
        order = sorted(names, key=lambda x: (res[x]['dwr'], res[x]['margin']))
        rank = {nm: i / (len(names) - 1) - 0.5 for i, nm in enumerate(order)}
        dr = np.array([rank[f'p{i}'] - rank[f'm{i}'] for i in range(n)])
        g = (dr[:, None] * eps).sum(0) / (2 * n * a.sigma)
        mom = a.momentum * mom + g; theta = theta + a.lr * mom
        if a.clip > 0: theta = np.clip(theta, -a.clip, a.clip)
        cres = res['c']
        rec = dict(gen=k, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), secs=round(time.time() - t0, 1), games=len(rows), seeds=[seed0, seed0 + a.seeds_per_gen],
                   centre=cres, pert_mean_dwr=float(np.mean([res[x]['dwr'] for x in names])), pert_best=max(res[x]['dwr'] for x in names),
                   grad_norm=float(np.linalg.norm(g)), theta_norm=float(np.linalg.norm(theta)),
                   top=[(f'{space[i][0]}:{space[i][1]}@{space[i][2]}', round(float(theta[i]), 3)) for i in np.argsort(-np.abs(theta))[:8]],
                   cstats_c=S._mean_stats(rows, 'cand:c'))
        with open(os.path.join(out, 'es_log.jsonl'), 'a') as f_: f_.write(json.dumps(rec) + '\n')
        json.dump(dict(gen=k, theta=theta.tolist(), mom=mom.tolist(), space=[list(x) for x in space]), open(st_path, 'w'))
        json.dump(tc(theta, dict(meta=dict(gen=k, source='es_search_r2.py'))), open(os.path.join(out, f'theta_{k:03d}.json'), 'w'), indent=1)
        log(f"gen {k}: centre dWR {cres['dwr']:+.3f} (h2h {cres['h2h']:+.3f}, field {cres['panel']:+.3f}, n {cres['n']}) | pert mean {rec['pert_mean_dwr']:+.3f} | "
            f"|g| {rec['grad_norm']:.3f} |theta| {rec['theta_norm']:.3f} | top {rec['top'][:4]} | {rec['secs']:.0f}s")
        if a.monitor_every and (k + 1) % a.monitor_every == 0:
            if mon_base is None:
                if os.path.exists(mon_path): mon_base = json.load(open(mon_path))['rows']
                else:
                    mon_base = pool.run(make_cells_r2(a.monitor_seed0, a.monitor_seeds, 'E3'), {}, log); json.dump(dict(rows=mon_base), open(mon_path, 'w'))
            mrows = pool.run(make_cells_r2(a.monitor_seed0, a.monitor_seeds, 'cand:c'), {'c': tc(theta)}, log)
            mres = paired_r2(mrows, 'c', S.base_of(mon_base))
            with open(os.path.join(out, 'monitor.jsonl'), 'a') as f_: f_.write(json.dumps(dict(gen=k, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), res=mres)) + '\n')
            log(f"  MONITOR gen {k}: dWR {mres['dwr']:+.4f} [{mres['lo']:+.4f}, {mres['hi']:+.4f}] h2h {mres['h2h']:+.4f} field {mres['panel']:+.4f} n {mres['n']}")

def run_eval(a, pool, log, out):
    cands = json.load(open(a.cands))
    lo, hi = map(int, a.seeds.split(':')); assert SEL_LO <= lo < hi <= SEL_HI
    cells = make_cells_r2(lo, hi - lo, 'E3')
    for nm in cands: cells += make_cells_r2(lo, hi - lo, 'cand:' + nm)
    rows = pool.run(cells, cands, log); base = S.base_of(rows)
    json.dump(dict(rows=rows, seeds=[lo, hi], cands=cands, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(os.path.join(out, 'eval_rows.json'), 'w'))
    for nm in cands:
        r = paired_r2(rows, nm, base)
        log(f"{nm:<28s} dWR {r['dwr']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] h2h {r['h2h']:+.4f} field {r['panel']:+.4f} n {r['n']} margin {r['margin']:+.0f} stats {S._mean_stats(rows, 'cand:' + nm)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['es', 'screen', 'eval']); ap.add_argument('--out', required=True)
    ap.add_argument('--spec', default='runs/a8/np/v3b_ens2_A_B.ens.json'); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--engine', default='cpp'); ap.add_argument('--device', default='cuda'); ap.add_argument('--shadow', type=int, default=1)
    ap.add_argument('--gens', type=int, default=60); ap.add_argument('--n', type=int, default=16); ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.2); ap.add_argument('--momentum', type=float, default=0.0); ap.add_argument('--clip', type=float, default=3.0)
    ap.add_argument('--k3', type=int, default=0); ap.add_argument('--seeds-per-gen', type=int, default=24); ap.add_argument('--gen-seed0', type=int, default=32_393_000)
    ap.add_argument('--monitor-every', type=int, default=5); ap.add_argument('--monitor-seed0', type=int, default=32_396_000); ap.add_argument('--monitor-seeds', type=int, default=200)
    ap.add_argument('--cands', default=''); ap.add_argument('--screen-seed0', type=int, default=32_392_000); ap.add_argument('--seeds-per-round', type=int, default=24)
    ap.add_argument('--keep-final', type=int, default=3); ap.add_argument('--max-rounds', type=int, default=5); ap.add_argument('--seeds', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f'{a.mode}.log'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%SZ ', time.gmtime()) + msg; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'start round-2 {a.mode} {vars(a)}')
    pool = S.Pool(a.spec, a.procs, a.batch, a.device, a.engine, shadow=bool(a.shadow))
    try:
        {'es': run_es, 'screen': run_screen, 'eval': run_eval}[a.mode](a, pool, log, a.out)
    finally:
        pool.close()
    log('done')

if __name__ == '__main__':
    main()
