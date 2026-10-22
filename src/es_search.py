import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

SEL_LO, SEL_HI = 32_380_000, 32_390_000
PUBLIC4 = ['hybu_submit', 'the_2965_master_hybrid_engine', 'master_engine_v3', 'frontier_the_soil_remembers_rain']
OPP_REFV3 = 'ens:incumbent/refv3.ens.json'
OPP_REFV2 = 'npz:incumbent/refv2.npz'
KIND_SPACE = ['SELL_WHEAT', 'SELL_CARROT', 'SELL_TOMATO', 'SELL_STRAWBERRY', 'SELL_MELON', 'SELL_EGG', 'SELL_MILK', 'SELL_WOOL',
              'BUY_PRODUCT_WHEAT', 'BUY_PRODUCT_FERTILIZER', 'BUY_SEED_WHEAT', 'BUY_SEED_CARROT', 'BUY_SEED_TOMATO', 'BUY_SEED_STRAWBERRY',
              'BUY_SEED_MELON', 'BUY_ANIMAL_GOOSE', 'BUY_ANIMAL_COW', 'BUY_ANIMAL_SHEEP', 'HIRE', 'BUY_LAND']
CROPS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']
PHASE_DAYS = [0, 5, 15, 25]

_EV = None

def _winit(spec, device, engine, r1, shadow):
    global _EV
    import es_gpu
    _EV = es_gpu.Evaluator(spec, device, engine, r1, shadow=shadow)

def _wtask(args):
    cells, cands = args
    _EV.set_cands(cands)
    rows = _EV.play(cells)
    out = []
    for r in rows:
        out.append({k: r[k] for k in ('seed', 'seat', 'a', 'b', 'mine', 'theirs', 'win', 'margin') if k in r} | {k: r[k] for k in ('cstats', 'act_diff_steps', 'secs') if k in r})
    return out

class Pool:
    def __init__(self, spec, procs, batch, device='cuda', engine='cpp', r1=os.environ.get('KGR_R1_BUNDLE', 'r1_bundle'), shadow=False):
        from multiprocessing import get_context
        self.batch = batch; self.procs = procs
        self.pool = get_context('spawn').Pool(procs, initializer=_winit, initargs=(spec, device, engine, r1, shadow))

    def run(self, cells, cands, log=None):
        nb = max(self.procs, math.ceil(len(cells) / self.batch))
        nb = min(nb, len(cells)); blocks = [cells[i::nb] for i in range(nb)]
        t0 = time.time(); rows = []
        for i, r in enumerate(self.pool.imap_unordered(_wtask, [(b, cands) for b in blocks], chunksize=1)):
            rows += r
        if log: log(f'  {len(cells)} games in {time.time() - t0:.0f}s ({len(cells) / max(1e-9, time.time() - t0):.2f} games/s)')
        return rows

    def close(self):
        self.pool.terminate(); self.pool.join()

def template(n_seeds, h2h_frac=0.5, publics=PUBLIC4):
    cells = []
    for i in range(n_seeds):
        j = i % 8
        opp = (OPP_REFV3, OPP_REFV2)[j % 2] if j < 6 else publics[(2 * (i // 8) + (j - 6)) % len(publics)]
        for seat in (0, 1):
            cells.append((i, seat, 'E3'))
            cells.append((i, seat, opp))
    return cells

def make_cells(seed0, n_seeds, who, publics=PUBLIC4):
    out = []
    for off, seat, opp in template(n_seeds, publics=publics):
        s = seed0 + off
        assert SEL_LO <= s < SEL_HI, f'seed {s} outside the selection block'
        if who == 'E3' and opp == 'E3': continue
        out.append(dict(seed=s, seat=seat, a=who, b=opp))
    return out

def teacher_win(rows_by_key, seed, seat, opp):
    if opp == 'E3': return 0.5
    return rows_by_key[('E3', seed, seat, opp)]['win']

def paired(rows, name, base):
    import collections
    d_by_seed = collections.defaultdict(list); grp = collections.defaultdict(list); mg = []
    for r in rows:
        if r['a'] != 'cand:' + name: continue
        k = (r['seed'], r['seat'], r['b'])
        t = 0.5 if r['b'] == 'E3' else base.get(k)
        if t is None: continue
        d = r['win'] - t; d_by_seed[r['seed']].append(d); g = 'h2h' if r['b'] == 'E3' else 'panel'; grp[g].append(d); mg.append(r['margin'])
    m = np.array([np.mean(v) for v in d_by_seed.values()]); n = sum(len(v) for v in d_by_seed.values())
    est = float(np.mean([x for v in d_by_seed.values() for x in v])) if n else float('nan')
    se = float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else float('nan')
    return dict(name=name, n=n, dwr=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se, h2h=float(np.mean(grp['h2h'])) if grp['h2h'] else float('nan'),
                panel=float(np.mean(grp['panel'])) if grp['panel'] else float('nan'), n_h2h=len(grp['h2h']), n_panel=len(grp['panel']), margin=float(np.mean(mg)) if mg else float('nan'))

def base_of(rows):
    return {(r['seed'], r['seat'], r['b']): r['win'] for r in rows if r['a'] == 'E3'}

EARLY, LATE, ALL = (0, 1), (2, 3), (0, 1, 2, 3)

SPACE_V1 = [('kind', 'HIRE', EARLY), ('kind', 'HIRE', LATE), ('kind', 'BUY_SEED_WHEAT', EARLY), ('kind', 'BUY_SEED_WHEAT', LATE),
            ('kind', 'BUY_SEED_STRAWBERRY', EARLY), ('kind', 'BUY_SEED_TOMATO', EARLY), ('kind', 'BUY_SEED_TOMATO', LATE),
            ('kind', 'BUY_SEED_CARROT', EARLY), ('kind', 'BUY_SEED_CARROT', LATE), ('kind', 'BUY_SEED_MELON', EARLY), ('kind', 'BUY_PRODUCT_WHEAT', EARLY),
            ('kind', 'BUY_ANIMAL_COW', EARLY), ('kind', 'BUY_ANIMAL_GOOSE', EARLY), ('kind', 'BUY_ANIMAL_SHEEP', EARLY), ('kind', 'BUY_LAND', EARLY),
            ('kind', 'SELL_WHEAT', EARLY), ('kind', 'SELL_WHEAT', LATE), ('kind', 'SELL_FERTILIZER', EARLY), ('kind', 'SELL_FERTILIZER', LATE),
            ('kind', 'SELL_STRAWBERRY', EARLY), ('kind', 'SELL_STRAWBERRY', LATE), ('kind', 'SELL_MILK', EARLY), ('kind', 'SELL_MILK', LATE),
            ('kind', 'SELL_WOOL', EARLY), ('kind', 'SELL_WOOL', LATE), ('kind', 'SELL_EGG', EARLY), ('kind', 'SELL_EGG', LATE),
            ('kind', 'SELL_CARROT', LATE), ('kind', 'SELL_TOMATO', LATE), ('kind', 'SELL_MELON', EARLY)] + [('crop', c, ALL) for c in CROPS]

def space_def(name='v1'):
    assert name == 'v1', name
    return [tuple(x) for x in SPACE_V1]

def theta_to_corr(theta, space, extra=None):
    kb, cb = {}, {}
    for v, (comp, nm, phs) in zip(theta, space):
        t = kb if comp == 'kind' else cb
        row = t.setdefault(nm, [0.0] * len(PHASE_DAYS))
        for p in phs: row[p] += float(v)
    d = dict(format='bes_correction_v1', phase_days=PHASE_DAYS)
    if kb: d['kind_bias'] = {k: [round(x, 6) for x in v] for k, v in kb.items()}
    if cb: d['crop_bias'] = {k: [round(x, 6) for x in v] for k, v in cb.items()}
    if extra: d.update(extra)
    return d

def run_es(a, pool, log, out):
    space = space_def()
    D = len(space); n = a.n
    st_path = os.path.join(out, 'es_state.json')
    if os.path.exists(st_path):
        st = json.load(open(st_path)); theta = np.array(st['theta']); mom = np.array(st['mom']); k0 = st['gen'] + 1
        log(f'resume at generation {k0}')
    else:
        theta = np.zeros(D); mom = np.zeros(D); k0 = 0
    json.dump(dict(space=space, args=vars(a)), open(os.path.join(out, 'space.json'), 'w'), indent=1)
    mon_base = None
    mon_path = os.path.join(out, 'monitor_teacher.json')
    for k in range(k0, a.gens):
        t0 = time.time()
        rng = np.random.default_rng([20260924, k])
        eps = rng.standard_normal((n, D))
        cands = {'c': theta_to_corr(theta, space)}
        for i in range(n):
            cands[f'p{i}'] = theta_to_corr(theta + a.sigma * eps[i], space); cands[f'm{i}'] = theta_to_corr(theta - a.sigma * eps[i], space)
        seed0 = a.gen_seed0 + k * a.seeds_per_gen
        assert a.gen_seed0 >= 32_381_000 and seed0 + a.seeds_per_gen <= 32_388_000, 'ES generation seeds must stay in 32,381,000-32,387,999'
        cells = make_cells(seed0, a.seeds_per_gen, 'E3')
        for nm in cands: cells += make_cells(seed0, a.seeds_per_gen, 'cand:' + nm)
        rows = pool.run(cells, cands, log)
        base = base_of(rows)
        res = {nm: paired(rows, nm, base) for nm in cands}

        names = [f'p{i}' for i in range(n)] + [f'm{i}' for i in range(n)]
        order = sorted(names, key=lambda x: (res[x]['dwr'], res[x]['margin']))
        rank = {nm: i / (len(names) - 1) - 0.5 for i, nm in enumerate(order)}
        dr = np.array([rank[f'p{i}'] - rank[f'm{i}'] for i in range(n)])
        g = (dr[:, None] * eps).sum(0) / (2 * n * a.sigma)
        mom = a.momentum * mom + g
        theta = (theta + a.lr * mom) * (1.0 - a.decay)
        if a.clip > 0: theta = np.clip(theta, -a.clip, a.clip)
        cres = res['c']
        rec = dict(gen=k, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), secs=round(time.time() - t0, 1), games=len(rows), seeds=[seed0, seed0 + a.seeds_per_gen],
                   centre=cres, pert_mean_dwr=float(np.mean([res[x]['dwr'] for x in names])), pert_best=max(res[x]['dwr'] for x in names),
                   grad_norm=float(np.linalg.norm(g)), theta_norm=float(np.linalg.norm(theta)), theta_max=float(np.abs(theta).max()),
                   top=[(f'{space[i][0]}:{space[i][1]}@{"".join(map(str, space[i][2]))}', round(float(theta[i]), 3)) for i in np.argsort(-np.abs(theta))[:8]],
                   cstats_c=_mean_stats(rows, 'cand:c'))
        with open(os.path.join(out, 'es_log.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
        json.dump(dict(gen=k, theta=theta.tolist(), mom=mom.tolist(), space=space), open(st_path, 'w'))
        json.dump(theta_to_corr(theta, space, dict(meta=dict(gen=k, source='es_search.py'))), open(os.path.join(out, f'theta_{k:03d}.json'), 'w'), indent=1)
        log(f"gen {k}: centre dWR {cres['dwr']:+.3f} (h2h {cres['h2h']:+.3f}, panel {cres['panel']:+.3f}, n {cres['n']}) | pert mean {rec['pert_mean_dwr']:+.3f} | "
            f"|g| {rec['grad_norm']:.3f} |theta| {rec['theta_norm']:.3f} max {rec['theta_max']:.3f} | top {rec['top'][:4]} | {rec['secs']:.0f}s")
        if a.monitor_every and (k + 1) % a.monitor_every == 0:
            if mon_base is None:
                if os.path.exists(mon_path): mon_base = json.load(open(mon_path))['rows']
                else:
                    mon_base = pool.run(make_cells(a.monitor_seed0, a.monitor_seeds, 'E3'), {}, log)
                    json.dump(dict(rows=mon_base), open(mon_path, 'w'))
            mc = {'c': theta_to_corr(theta, space)}
            mrows = pool.run(make_cells(a.monitor_seed0, a.monitor_seeds, 'cand:c'), mc, log)
            mres = paired(mrows, 'c', base_of(mon_base))
            with open(os.path.join(out, 'monitor.jsonl'), 'a') as f: f.write(json.dumps(dict(gen=k, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), res=mres, cstats=_mean_stats(mrows, 'cand:c'))) + '\n')
            log(f"  MONITOR gen {k}: dWR {mres['dwr']:+.4f} [{mres['lo']:+.4f}, {mres['hi']:+.4f}] h2h {mres['h2h']:+.4f} panel {mres['panel']:+.4f} n {mres['n']}")

def _mean_stats(rows, who):
    v = [dict(r.get('cstats') or {}, **({'act_diff_steps': r['act_diff_steps']} if 'act_diff_steps' in r else {})) for r in rows if r['a'] == who]
    v = [x for x in v if x]
    if not v: return None
    return {k: round(float(np.mean([x.get(k, 0) for x in v])), 2) for k in v[0]}

def run_screen(a, pool, log, out):
    cands_all = json.load(open(a.cands))
    alive = list(cands_all); rnd = 0; seed0 = a.screen_seed0; allrows = []
    seeds_r = a.seeds_per_round
    while True:
        assert seed0 + seeds_r <= 32_381_000, 'screen seeds must stay in 32,380,000-32,380,999'
        cells = make_cells(seed0, seeds_r, 'E3')
        for nm in alive: cells += make_cells(seed0, seeds_r, 'cand:' + nm)
        rows = pool.run(cells, {nm: cands_all[nm] for nm in alive}, log)
        allrows += rows
        base = base_of(allrows)
        res = sorted((paired(allrows, nm, base) for nm in alive), key=lambda r: (-r['dwr'], -r['margin']))
        with open(os.path.join(out, 'screen_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(round=rnd, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seeds=[seed0, seed0 + seeds_r], res=res,
                                    stats={nm: _mean_stats(allrows, 'cand:' + nm) for nm in alive})) + '\n')
        json.dump(dict(rows=allrows), open(os.path.join(out, 'screen_rows.json'), 'w'))
        log(f'round {rnd}: seeds {seed0}-{seed0 + seeds_r - 1}, {len(alive)} alive')
        for r in res: log(f"   {r['name']:<28s} dWR {r['dwr']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] h2h {r['h2h']:+.4f} panel {r['panel']:+.4f} n {r['n']} margin {r['margin']:+.0f}")
        seed0 += seeds_r; rnd += 1
        if len(alive) <= a.keep_final or rnd >= a.max_rounds: break
        alive = [r['name'] for r in res[:max(a.keep_final, len(alive) // 2)]]; seeds_r *= 2

def run_eval(a, pool, log, out):
    cands = json.load(open(a.cands))
    lo, hi = map(int, a.seeds.split(':')); assert SEL_LO <= lo < hi <= SEL_HI
    cells = make_cells(lo, hi - lo, 'E3')
    for nm in cands: cells += make_cells(lo, hi - lo, 'cand:' + nm)
    rows = pool.run(cells, cands, log)
    base = base_of(rows)
    json.dump(dict(rows=rows, seeds=[lo, hi], cands=cands, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(os.path.join(out, 'eval_rows.json'), 'w'))
    for nm in cands:
        r = paired(rows, nm, base)
        log(f"{nm:<28s} dWR {r['dwr']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] h2h {r['h2h']:+.4f} panel {r['panel']:+.4f} n {r['n']} margin {r['margin']:+.0f} stats {_mean_stats(rows, 'cand:' + nm)}")

def run_pbench(a, pool, log, out):
    global SEL_LO, SEL_HI
    SEL_LO, SEL_HI = 32_309_200, 32_309_300
    cands = {'c': dict(format='bes_correction_v1', plant2=1.0)}
    cells = make_cells(32_309_200, a.seeds_per_gen, 'E3') + make_cells(32_309_200, a.seeds_per_gen, 'cand:c')
    cells = cells * max(1, a.n)
    t0 = time.time(); rows = pool.run(cells, cands, log)
    log(f'PBENCH procs {a.procs} batch {a.batch}: {len(rows)} games in {time.time() - t0:.0f}s = {len(rows) / (time.time() - t0):.2f} games/s')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['es', 'screen', 'eval', 'pbench']); ap.add_argument('--out', required=True)
    ap.add_argument('--spec', default='runs/a8/np/ens3_V1e39_E1.ens.json'); ap.add_argument('--procs', type=int, default=16); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--engine', default='cpp'); ap.add_argument('--device', default='cuda'); ap.add_argument('--shadow', type=int, default=0)

    ap.add_argument('--gens', type=int, default=40); ap.add_argument('--n', type=int, default=16); ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.2); ap.add_argument('--momentum', type=float, default=0.0); ap.add_argument('--decay', type=float, default=0.0)
    ap.add_argument('--clip', type=float, default=3.0); ap.add_argument('--seeds-per-gen', type=int, default=24); ap.add_argument('--gen-seed0', type=int, default=32_381_000)
    ap.add_argument('--monitor-every', type=int, default=5); ap.add_argument('--monitor-seed0', type=int, default=32_388_000); ap.add_argument('--monitor-seeds', type=int, default=200)

    ap.add_argument('--cands', default=''); ap.add_argument('--screen-seed0', type=int, default=32_380_000); ap.add_argument('--seeds-per-round', type=int, default=24)
    ap.add_argument('--keep-final', type=int, default=3); ap.add_argument('--max-rounds', type=int, default=5); ap.add_argument('--seeds', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f'{a.mode}.log'), 'a')

    def log(msg):
        line = time.strftime('%H:%M:%SZ ', time.gmtime()) + msg; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'start {a.mode} {vars(a)}')
    pool = Pool(a.spec, a.procs, a.batch, a.device, a.engine, shadow=bool(a.shadow))
    try:
        {'es': run_es, 'screen': run_screen, 'eval': run_eval, 'pbench': run_pbench}[a.mode](a, pool, log, a.out)
    finally:
        pool.close()
    log('done')

if __name__ == '__main__':
    main()
