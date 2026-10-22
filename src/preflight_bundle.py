import argparse, hashlib, json, os, sys, tarfile, time
ap = argparse.ArgumentParser(); ap.add_argument('tar'); ap.add_argument('--r1', required=True); ap.add_argument('--out', default='')
a = ap.parse_args(); T = os.path.abspath(a.tar); B = json.load(open(T[:-len('.tar.gz')] + '/BUNDLE.json'))
res = dict(tar=T, sha256=hashlib.sha256(open(T, 'rb').read()).hexdigest(), size=os.path.getsize(T), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
mod = [k for k in B['sha256'] if k.endswith('_np_policy.py')][0][:-len('_np_policy.py')]
exp = set(B['sha256'])
work = os.path.join(os.path.dirname(T), 'preflight_' + os.path.basename(T)[:-7]); os.makedirs(work, exist_ok=True)
with tarfile.open(T) as tf:
    names = [m.name for m in tf.getmembers()]; tf.extractall(work)
res['members_ok'] = set(names) == exp and len(names) == len(exp); res['no_appledouble'] = not any(os.path.basename(n).startswith('._') for n in names)
res['member_sha_ok'] = all(hashlib.sha256(open(os.path.join(work, n), 'rb').read()).hexdigest() == B['sha256'][n] for n in names if n in B['sha256'])
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable, read_file
main_py = os.path.join(work, 'main.py')
def load(path):
    return get_last_callable(read_file(path, path), path=path)
times = {}
def timed(fn, key):
    times.setdefault(key, [])
    def w(obs, cfg):
        t = time.perf_counter(); r = fn(obs, cfg); times[key].append(time.perf_counter() - t); return r
    return w
def episode(f0, f1, k0, k1):
    env = make('kaggriculture', debug=True); t0 = time.time(); steps = env.run([timed(f0, k0), timed(f1, k1)]); last = steps[-1]
    return dict(statuses=[s['status'] for s in last], rewards=[s['reward'] for s in last], wall_s=round(time.time() - t0, 1))
c0, c1 = load(main_py), load(main_py)
res['selfplay'] = episode(c0, c1, 'cand_a', 'cand_b')
r1 = load(os.path.join(a.r1, 'main.py')); c2 = load(main_py)
res['vs_r1_cand_seat0'] = episode(c2, r1, 'cand_vs', 'r1')
res['agent_errors'] = [f.__globals__.get('_ERRORS') for f in (c0, c1, c2)]
res['torch_imported'] = any(m == 'torch' or m.startswith('torch.') for m in sys.modules)
res['timing_ms'] = {k: dict(turns=len(v), worst=round(1000 * max(v), 1), p99=round(1000 * sorted(v)[int(0.99 * len(v))], 1), mean=round(1000 * sum(v) / len(v), 2)) for k, v in times.items()}
ok_ep = lambda e: e['statuses'] == ['DONE', 'DONE'] and all(r is not None and r > 3000 for r in e['rewards'])
res['PASS'] = bool(res['members_ok'] and res['no_appledouble'] and res['member_sha_ok'] and ok_ep(res['selfplay']) and ok_ep(res['vs_r1_cand_seat0'])
                   and res['agent_errors'] == [0, 0, 0] and not res['torch_imported'] and max(v['worst'] for k, v in res['timing_ms'].items() if k != 'r1') < 200)
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
