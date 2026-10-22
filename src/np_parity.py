import argparse, glob, random, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import codec as C
from model import Net
import np_policy as NP

ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('npz'); ap.add_argument('shards'); ap.add_argument('--n', type=int, default=400)
ap.add_argument('--team', type=int, default=5); ap.add_argument('--temp', type=float, default=0.5)
a = ap.parse_args(); random.seed(0)
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); assert ck.get('codec') == C.CODEC_TAG
net = Net(ch=ck['ch'], blocks=ck['blocks']); net.load_state_dict(ck['state']); net.eval()
pol = NP.NpPolicy(a.npz, mode='argmax')
files = sorted(glob.glob(a.shards)); random.shuffle(files)
du = dk = dq = 0.0; agree_u = agree_k = agree_q = tot_u = tot_k = tot_q = 0; n = 0
for f in files:
    d = np.load(f); T = len(d['glob'])
    for t in random.sample(range(T), min(40, T)):
        grid = d['grid'][t].astype(np.float32); units = d['units'][t].astype(np.float32); xy = d['xy'][t].astype(np.int64); nu = int(d['n_units'][t])
        glob_ = d['glob'][t].astype(np.float32); um = d['umask'][t]; mm = d['mmask'][t]; qm = d['qmax'][t].astype(np.int64)

        with torch.no_grad():
            tb = lambda x, dt: torch.as_tensor(np.asarray(x)[None]).to(dt)
            team = torch.tensor([a.team])
            fmap, g = net.trunk(tb(grid, torch.float32), tb(glob_, torch.float32), team)
            ul = net.unit_logits(fmap, g, tb(units, torch.float32), tb(xy, torch.int64), tb(um, torch.bool))[0, :nu].numpy() / a.temp
            kl = net.kind_logits(g, tb(mm, torch.bool))[0].numpy() / a.temp
            kinds = kl.argmax(-1)
            ql = net.qty_logits(g, torch.as_tensor(kinds)[None], tb(qm, torch.int64))[0].numpy() / a.temp
        x, gn = pol.features(grid, glob_)
        uln = np.where(um[:nu], pol.unit_logits(x, gn, units, xy, nu), -1e9); kln = np.where(mm[None], pol.kind_logits(gn), -1e9)
        qmk = np.maximum(qm[kinds], 1); qln = np.where(np.arange(C.MAX_Q)[None] < qmk[:, None], pol.qty_logits(gn, kinds), -1e9)
        valid_u = um[:nu]; du = max(du, float(np.abs(np.where(valid_u, ul - uln, 0)).max()))
        dk = max(dk, float(np.abs(np.where(mm[None], kl - kln, 0)).max()))
        vq = np.arange(C.MAX_Q)[None] < qmk[:, None]; dq = max(dq, float(np.abs(np.where(vq, ql - qln, 0)).max()))
        agree_u += int((ul.argmax(-1) == uln.argmax(-1)).sum()); tot_u += nu
        agree_k += int((kinds == kln.argmax(-1)).sum()); tot_k += len(kinds)
        agree_q += int((ql.argmax(-1) == qln.argmax(-1)).sum()); tot_q += len(kinds); n += 1
    if n >= a.n: break
print(f'{n} states: max|dlogit| unit {du:.2e} kind {dk:.2e} qty {dq:.2e} | argmax agreement unit {agree_u}/{tot_u} kind {agree_k}/{tot_k} qty {agree_q}/{tot_q}')
ok = agree_u == tot_u and agree_k == tot_k and agree_q == tot_q and max(du, dk, dq) < 1e-2
print('PARITY', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
