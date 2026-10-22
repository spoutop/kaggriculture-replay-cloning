import argparse, collections, hashlib, json, math, os, subprocess, sys, time

CORR = '../../../btrain_es/corr/CHOSEN_R2.json'
OPPS = {'v7': 'ens:runs/a8/np/v3c_C_F6.ens.json', 'cx': 'ens:runs/a8/np/v3e_x_ens2.ens.json', 'abo': 'abo_v45'}
WT = {'v7': 0.45, 'cx': 0.45, 'abo': 0.10}
GP = os.environ.get('KGR_GPU_PLAY', 'gpu_play_main.py')

def spec_path(outdir, ms):
    key = '_'.join(f'{m}x{c}' for m, c in sorted(collections.Counter(ms).items()))
    p = os.path.join('runs/a8/np', f'enssel_{hashlib.md5(key.encode()).hexdigest()[:12]}.ens.json')
    return p, key

def write_spec(pool, ms, p):
    cnt = collections.Counter(ms); mem = sorted(cnt)
    json.dump(dict(members=[os.path.relpath(pool[m], 'runs/a8/np') for m in mem], weights=[cnt[m] for m in mem], correction=CORR, enssel=dict(cnt)), open(p, 'w'))

def evaluate(pool, cands, seeds, procs, out):
    specs = {}
    for name, ms in cands.items():
        p, key = spec_path(None, ms); write_spec(pool, ms, p); specs[name] = p
    if not os.path.exists(out):
        cmd = [sys.executable, GP, '--base-spec', 'runs/a8/np/v3e_x_ens2.ens.json', '--pols', ','.join('ens:' + specs[n] for n in cands),
               '--opps', ','.join(OPPS.values()), '--seeds', seeds, '--procs', str(procs), '--out', out]
        subprocess.run(cmd, check=True)
    rows = json.load(open(out))['rows']; inv = {'ens:' + v: k for k, v in specs.items()}; opk = {v: k for k, v in OPPS.items()}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    res = {}
    for n in cands:
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if inv.get(r['pol']) != n: continue
            by[opk[r['opp']]][r['seed']].append(r['win'])
        w = {o: sum(sum(v) for v in by[o].values()) / max(1, sum(len(v) for v in by[o].values())) for o in OPPS}
        obj = sum(WT[o] * w[o] for o in OPPS)
        for s in set().union(*[set(by[o]) for o in OPPS]):
            per[n][s] = sum(WT[o] * (sum(by[o][s]) / len(by[o][s])) for o in OPPS if by[o].get(s))
        res[n] = dict(obj=obj, **{f'win_{o}': w[o] for o in OPPS}, n=sum(len(v) for o in OPPS for v in by[o].values()))
    return res, per

def paired_se(per, a, b):
    ss = sorted(set(per[a]) & set(per[b])); d = [per[a][s] - per[b][s] for s in ss]
    if len(d) < 2: return 0.0, float('inf')
    m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)); return m, sd / math.sqrt(len(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pool'); ap.add_argument('outdir'); ap.add_argument('--seeds', required=True); ap.add_argument('--procs', type=int, default=20)
    ap.add_argument('--max-members', type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); pool = json.load(open(a.pool)); log = open(os.path.join(a.outdir, 'enssel.log'), 'a')
    P = lambda *x: (print(*x, flush=True), print(*x, file=log, flush=True))
    P(f'START {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} pool {len(pool)} seeds {a.seeds}')
    res, per = evaluate(pool, {m: [m] for m in pool}, a.seeds, a.procs, os.path.join(a.outdir, 'step00.json'))
    cur = max(res, key=lambda n: (res[n]['obj'], n)); ms = [cur]; best = res[cur]; curper = per[cur]
    P('step 0', cur, json.dumps(best))
    hist = [dict(step=0, members=list(ms), **best)]
    for step in range(1, a.max_members):
        cands = {m: ms + [m] for m in pool}
        res, per = evaluate(pool, cands, a.seeds, a.procs, os.path.join(a.outdir, f'step{step:02d}.json'))
        per['__cur__'] = curper
        nb = max(res, key=lambda n: (res[n]['obj'], n)); gain, se = paired_se(per, nb, '__cur__')
        P(f'step {step} best add {nb} obj {res[nb]["obj"]:.4f} (cur {best["obj"]:.4f}) paired gain {gain:+.4f} se {se:.4f}', json.dumps(res[nb]))
        if gain <= se:
            P(f'STOP at step {step}: gain {gain:+.4f} <= se {se:.4f}'); break
        ms = ms + [nb]; best = res[nb]; curper = per[nb]; hist.append(dict(step=step, members=list(ms), **best))
    final = collections.Counter(ms); p, key = spec_path(None, ms); write_spec(pool, ms, p)
    json.dump(dict(members=dict(final), spec=p, history=hist), open(os.path.join(a.outdir, 'FINAL.json'), 'w'), indent=1)
    P('FINAL', dict(final), p)

if __name__ == '__main__':
    main()
