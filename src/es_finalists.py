import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--last', type=int, default=10)
a = ap.parse_args()
st = json.load(open(os.path.join(a.run, 'es_state.json'))); space = [tuple(x) for x in st['space']]; last_gen = st['gen']
mon = [json.loads(x) for x in open(os.path.join(a.run, 'monitor.jsonl'))]
best = max(mon, key=lambda m: m['res']['dwr'])
print('monitor reads:', [(m['gen'], round(m['res']['dwr'], 4)) for m in mon], '-> best gen', best['gen'])
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_search import theta_to_corr

def theta_of_file(k):
    c = json.load(open(os.path.join(a.run, f'theta_{k:03d}.json')))
    v = []
    for comp, nm, phs in space:
        tab = c.get('kind_bias' if comp == 'kind' else 'crop_bias', {})
        v.append(float(tab.get(nm, [0.0] * 4)[phs[0]]))
    return np.array(v)

cent = [theta_of_file(g - 1) for g in range(last_gen - a.last + 1, last_gen + 1)]
poly = np.mean(cent, 0)
fin = theta_of_file(last_gen)
bestv = theta_of_file(best['gen'])
meta = dict(run=a.run, last_gen=last_gen)
out = {'ES_best': theta_to_corr(bestv, space, dict(meta=dict(meta, kind='best monitor centre', monitor_gen=best['gen'], monitor=best['res']))),
       'ES_polyak': theta_to_corr(poly, space, dict(meta=dict(meta, kind=f'mean of the centres of generations {last_gen - a.last + 1}-{last_gen}'))),
       'ES_final': theta_to_corr(fin, space, dict(meta=dict(meta, kind='final centre', file=f'theta_{last_gen:03d}.json')))}
for k, v in out.items():
    json.dump(v, open(os.path.join(a.out_dir, k + '.json'), 'w'), indent=1)
    print(k, 'written; |theta|', round(float(np.linalg.norm({'ES_best': bestv, 'ES_polyak': poly, 'ES_final': fin}[k])), 3))

chk = theta_to_corr(theta_of_file(best['gen']), space)
orig = json.load(open(os.path.join(a.run, f"theta_{best['gen']:03d}.json")))
assert all(np.allclose(chk['kind_bias'][k], orig['kind_bias'][k], atol=1e-6) for k in orig['kind_bias']), 'theta round trip failed'
print('round trip OK')
