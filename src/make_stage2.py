import argparse, copy, itertools, json

ap = argparse.ArgumentParser(); ap.add_argument('--screen', required=True); ap.add_argument('--names', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--mags', default='0.5,1.5,2')
a = ap.parse_args()
S = json.load(open(a.screen)); names = a.names.split(','); mags = [float(x) for x in a.mags.split(',')]

def scale(c, m):
    c = copy.deepcopy(c)
    for key in ('kind_bias', 'crop_bias', 'qty_scale'):
        if key in c: c[key] = {k: [x * m for x in v] for k, v in c[key].items()}
    if 'svc_bias' in c: c['svc_bias'] = [x * m for x in c['svc_bias']]
    if 'plant2' in c: c['plant2'] = c['plant2'] * m
    return c

def add(cs):
    out = dict(format='bes_correction_v1', phase_days=[0, 5, 15, 25])
    for c in cs:
        assert c.get('phase_days', [0, 5, 15, 25]) == [0, 5, 15, 25]
        for key in ('kind_bias', 'crop_bias', 'qty_scale'):
            for k, v in c.get(key, {}).items():
                t = out.setdefault(key, {}).setdefault(k, [0.0] * 4); out[key][k] = [x + y for x, y in zip(t, v)]
        if 'svc_bias' in c: out['svc_bias'] = [x + y for x, y in zip(out.get('svc_bias', [0.0] * 4), c['svc_bias'])]
        if 'plant2' in c: out['plant2'] = max(out.get('plant2', 0.0), c['plant2'])
    return out

C = {}
for n in names:
    C[n] = S[n]
    for m in mags: C[f'{n}_x{m:g}'] = scale(S[n], m)
for k in range(2, len(names) + 1):
    for combo in itertools.combinations(names, k):
        C['+'.join(combo)] = add([S[n] for n in combo])
json.dump(C, open(a.out, 'w'), indent=1)
print(len(C), list(C))
