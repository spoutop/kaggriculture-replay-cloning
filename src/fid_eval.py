import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import codec as C
from model import Net
import bc_train2 as T

SUMS = ['n_kind', 'nll_kind', 'n_knn', 'c_knn', 'n_q', 'c_q', 'e_q', 'nll_q', 'n_plant', 'c_plant', 'nll_plant', 'n_exec', 'c_exec', 'nll_exec', 'n_exnp', 'c_exnp']

def load_net(ck, dev):
    c = torch.load(ck, map_location='cpu', weights_only=False)
    assert c.get('codec') == C.CODEC_TAG, (ck, c.get('codec'))
    net = Net(ch=c.get('ch', 128), blocks=c.get('blocks', 8), gextra=c.get('gextra', 0), ghist=c.get('ghist', 0), uhist=c.get('uhist', 0), uattn=c.get('uattn', 0), gplan=c.get('gplan', 0))
    net.load_state_dict(c['state']); return net.to(dev).eval()

@torch.no_grad()
def shard_sums(net, D, dev):
    s = dict.fromkeys(SUMS, 0.0)
    for i in range(0, len(D['glob']), 1024):
        b = T.batch(D, np.arange(i, min(i + 1024, len(D['glob']))), dev)
        ul, kl, ql, _ = net(b['grid'], b['units'], b['xy'], b['glob'], b['umask'], b['mmask'], b['qmax'], b['a_kind'], team=b['team'])
        ul, kl, ql = ul.float(), kl.float(), ql.float()
        au, ak, aq = b['a_unit'], b['a_kind'], b['a_q']

        lk = F.cross_entropy(kl.flatten(0, 1), ak.flatten(), reduction='none'); s['n_kind'] += lk.numel(); s['nll_kind'] += lk.sum().item()
        nn_ = ak != C.MK_NONE; s['n_knn'] += nn_.sum().item(); s['c_knn'] += (kl.argmax(-1)[nn_] == ak[nn_]).sum().item()
        hasq = nn_ & (ak != C.MK_HIRE) & (ak != C.MK_LAND)
        if hasq.any():
            qq = ql[hasq]; tq = aq[hasq]; pq = qq.argmax(-1)
            s['n_q'] += tq.numel(); s['c_q'] += (pq == tq).sum().item(); s['e_q'] += (pq - tq).abs().float().sum().item()
            s['nll_q'] += F.cross_entropy(qq, tq, reduction='sum').item()

        present = au >= 0; plant = present & (au >= C.A_PLANT0) & (au < C.A_PLANT0 + len(C.CROPS)); ex = present & ~plant; exnp = ex & (au != C.A_PASS)
        for m, n, c, l in ((plant, 'n_plant', 'c_plant', 'nll_plant'), (ex, 'n_exec', 'c_exec', 'nll_exec')):
            if m.any():
                lo = ul[m]; t = au[m]; s[n] += t.numel(); s[c] += (lo.argmax(-1) == t).sum().item(); s[l] += F.cross_entropy(lo, t, reduction='sum').item()
        if exnp.any(): s['n_exnp'] += exnp.sum().item(); s['c_exnp'] += (ul[exnp].argmax(-1) == au[exnp]).sum().item()
    return s

METRICS = {
    'kind_acc': ('c_knn', 'n_knn', True), 'kind_nll': ('nll_kind', 'n_kind', False),
    'qty_acc': ('c_q', 'n_q', True), 'qty_abs_err': ('e_q', 'n_q', False), 'qty_nll': ('nll_q', 'n_q', False),
    'plant_acc': ('c_plant', 'n_plant', True), 'plant_nll': ('nll_plant', 'n_plant', False),
    'exec_acc': ('c_exec', 'n_exec', True), 'exec_np_acc': ('c_exnp', 'n_exnp', True), 'exec_nll': ('nll_exec', 'n_exec', False)}

def summarise(per_game, n_boot=2000, seed=0):
    A = {k: np.array([g[k] for g in per_game], np.float64) for k in SUMS}; rng = np.random.default_rng(seed); G = len(per_game)
    idx = rng.integers(0, G, size=(n_boot, G)); out = {}
    for name, (num, den, _) in METRICS.items():
        tot = A[den].sum(); v = A[num].sum() / tot if tot else float('nan')
        bn = A[num][idx].sum(1); bd = A[den][idx].sum(1); bs = np.where(bd > 0, bn / np.maximum(bd, 1e-12), np.nan)
        out[name] = dict(v=v, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)), n=int(tot))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ck', nargs='+', required=True); ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--list', default='')
    ap.add_argument('--device', default='mps'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); dev = torch.device(a.device)
    files = sorted(f for d in a.data for f in glob.glob(f'{d}/*.npz'))
    if a.list: keep = set(json.load(open(a.list))); files = [f for f in files if Path(f).stem in keep]; assert len(files) == len(keep), (len(files), len(keep))
    print(f'{len(files)} shards', flush=True)
    data = [(Path(f).stem, T.load([f])) for f in files]
    res = {}
    for ck in a.ck:
        t0 = time.time(); net = load_net(ck, dev); pg = []
        for stem, D in data: g = shard_sums(net, D, dev); g['shard'] = stem; pg.append(g)
        res[ck] = dict(summary=summarise(pg), per_game=pg, n_games=len(pg))
        sm = res[ck]['summary']
        print(f"{ck}: " + ' '.join(f"{k} {v['v']:.4f}" for k, v in sm.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(files=[s for s, _ in data], data=a.data, list=a.list, codec=C.CODEC_TAG, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), results=res), open(a.out, 'w'), indent=1)
    print('wrote', a.out)

if __name__ == '__main__':
    main()
