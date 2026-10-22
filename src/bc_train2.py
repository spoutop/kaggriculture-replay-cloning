import argparse, glob, json, random, time
from pathlib import Path
import numpy as np, torch
import codec as C
import encoder as E_
from model import Net
from bc_train import losses
import vocab_adapt
CANON_MARKET = False
GHIST_DIR = ''
UHIST = 0; PLAN_DIR = ""
KEYS = ['grid', 'units', 'xy', 'n_units', 'glob', 'umask', 'mmask', 'qmax', 'a_unit', 'a_kind', 'a_q']

def load(files, stride=1):
    parts = {k: [] for k in KEYS}; team = []; hist = []; uh = []; pl = []
    for f in files:
        d = np.load(f); meta = d['meta']; tm = int(meta[4]) if len(meta) > 4 else 0
        if 'codec' not in d.files or str(d['codec']) != C.CODEC_TAG:
            raise SystemExit(f'shard {f} was not built with codec {C.CODEC_TAG} (found {str(d["codec"]) if "codec" in d.files else "none"})')
        n = len(d['glob']); sel = slice(random.randrange(stride) if stride > 1 else 0, n, stride)
        for k in KEYS: parts[k].append(d[k][sel])
        team.append(np.full(len(parts['glob'][-1]), tm, np.int64))
        if UHIST:
            au_ = d['a_unit'].astype(np.int64); none_ = np.full((1, au_.shape[1]), C.N_UNIT, np.int64); p1 = np.concatenate([none_, au_[:-1]]); p2 = np.concatenate([none_, none_, au_[:-2]])
            p1 = np.where(p1 < 0, C.N_UNIT, p1); p2 = np.where(p2 < 0, C.N_UNIT, p2); uh.append(np.stack([p1, p2], -1)[sel])
        if PLAN_DIR: pl.append(np.load(Path(PLAN_DIR) / (Path(f).stem + '.npy'))[sel].astype(np.float32))
        if GHIST_DIR: hist.append(np.load(Path(GHIST_DIR) / (Path(f).stem + '.npy'))[sel].astype(np.float32))
    D = {k: np.concatenate(v) for k, v in parts.items()}; D['team'] = np.concatenate(team)
    if GHIST_DIR: D['ghist'] = np.concatenate(hist); assert len(D['ghist']) == len(D['glob'])
    if UHIST: D['uh'] = np.concatenate(uh); assert len(D['uh']) == len(D['glob'])
    if PLAN_DIR: D['gplan'] = np.concatenate(pl); assert len(D['gplan']) == len(D['glob'])
    if CANON_MARKET:

        k = D['a_kind'].astype(np.int64); q = D['a_q'].astype(np.int64)
        if CANON_MARKET == 2:
            n_sell = len(C.PRODUCTS); is_sell = (k >= 1) & (k <= n_sell); pos = np.arange(k.shape[1])[None, :]
            key = np.where(k == 0, 10**7, np.where(is_sell, k * 100 + (99 - q), 10**6 + pos))
        else:
            key = np.where(k == 0, 10**6, k * 100 + (99 - q))
        order = np.argsort(key, axis=1, kind='stable')
        D['a_kind'] = np.take_along_axis(D['a_kind'], order, 1); D['a_q'] = np.take_along_axis(D['a_q'], order, 1)
    return D

def _load_one(f):
    return load([f])

def _concat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}

class Prefetch:
    def __init__(self, n):
        import concurrent.futures as cf, multiprocessing as mp
        self.ex = cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context('spawn'), initializer=_worker_init, initargs=(CANON_MARKET, GHIST_DIR, UHIST, PLAN_DIR))
        self.fut = None
    def submit(self, files):
        self.fut = [self.ex.submit(_load_one, f) for f in files]
    def get(self):
        ds = [x.result() for x in self.fut]; self.fut = None; return _concat(ds)

def _worker_init(cm, gd, uh, pd):
    global CANON_MARKET, GHIST_DIR, UHIST, PLAN_DIR
    CANON_MARKET, GHIST_DIR, UHIST, PLAN_DIR = cm, gd, uh, pd
    import sys as _s
    if _s.platform.startswith('linux'):
        import ctypes, signal
        ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)

def batch(D, idx, dev):
    g = lambda k, dt: torch.as_tensor(D[k][idx]).to(dev, dtype=dt, non_blocking=True)
    b = dict(grid=g('grid', torch.float32), units=g('units', torch.float32), xy=g('xy', torch.int64), glob=g('glob', torch.float32), umask=g('umask', torch.bool),
             mmask=g('mmask', torch.bool), qmax=g('qmax', torch.int64), a_unit=g('a_unit', torch.int64), a_kind=g('a_kind', torch.int64), a_q=g('a_q', torch.int64), team=g('team', torch.int64))
    qm = b['qmax'].gather(1, b['a_kind']).clamp(min=1); b['a_q'] = torch.minimum(b['a_q'], qm - 1)
    if 'ghist' in D: b['ghist'] = g('ghist', torch.float32)
    if 'uh' in D: b['uh'] = g('uh', torch.int64)
    if 'gplan' in D: b['gplan'] = g('gplan', torch.float32)
    return b

class TeamNet(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net; self.team = None; self.ghist = None; self.uh = None; self.gplan = None
    def forward(self, *args): return self.net(*args, team=self.team, ghist=self.ghist, uh=self.uh, gplan=self.gplan)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--out', required=True); ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=768); ap.add_argument('--lr', type=float, default=6e-4); ap.add_argument('--ch', type=int, default=128); ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=160); ap.add_argument('--tb-repeat', type=int, default=1); ap.add_argument('--init', default=None); ap.add_argument('--allow', default='', help='json list of shard basenames (eid_seat): train only on these (ONE agent version per team: teams upload new agents, mixing versions is label noise)'); ap.add_argument('--teams', default='', help='comma list of team indices: train (and validate) only on these teams\' seats (specialisation fine-tune)'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--gextra', type=int, default=0, help='model.N_TIME (31) = explicit clock features; 0 = off'); ap.add_argument('--val-last', type=int, default=0, help='1 = validation = the most RECENT tenth of the shards (time split) instead of every tenth')
    ap.add_argument('--ghist-dir', default='', help='sidecar directory of market-inventory history blocks (encoder.InvHist); enables the ghist input')
    ap.add_argument('--uhist', type=int, default=0, help='1 = per-unit last-two-action history input (V2)'); ap.add_argument('--uattn', type=int, default=0, help='1 = attention over the units (V2)')
    ap.add_argument('--plan-dir', default='', help='sidecars of plan_sidecar.py: enables the plan + progress input (V2)'); ap.add_argument('--uh-drop', type=float, default=0.15, help='share of training samples whose unit history is blanked (keeps the policy functional without it; guards against the copycat shortcut)')
    ap.add_argument('--seed', type=int, default=-1, help='>= 0: seed python/numpy/torch (multi-seed recipe comparisons: single runs cannot separate 0.016)')
    ap.add_argument('--early-w', type=float, default=1.0); ap.add_argument('--early-until', type=int, default=384); ap.add_argument('--wk', type=float, default=1.0, help='weight of the market-kind loss (all measured clone infidelity is in the market head)')
    ap.add_argument('--canon-market', type=int, default=0, help='1 = canonical order of the market-order targets (see load)')
    ap.add_argument('--val-list', default='', help='JSON list of shard stems = the validation (select) split; excluded from training')
    ap.add_argument('--weights', default='', help='JSON {shard stem: weight}: per-epoch weighted shard sampling with replacement (missing stems weight 0)')
    ap.add_argument('--resume', type=int, default=0, help='1 = continue from OUT/resume.pt (same command line)')
    ap.add_argument('--max-val', type=int, default=200)
    ap.add_argument('--epoch-draws', type=int, default=0, help='with --weights: shards drawn per epoch (0 = number of weighted train shards)')
    ap.add_argument('--load-workers', type=int, default=0, help='>0: parallel shard decompression + next-chunk prefetch with this many processes')
    a = ap.parse_args(); dev = torch.device(a.device); amp = a.device == 'cuda'
    if a.seed >= 0: random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    global UHIST, PLAN_DIR; UHIST = int(a.uhist); PLAN_DIR = a.plan_dir
    global CANON_MARKET, GHIST_DIR; CANON_MARKET = int(a.canon_market); GHIST_DIR = a.ghist_dir
    files = sorted(f for d in a.data for f in glob.glob(f'{d}/*.npz'))
    if a.ghist_dir: files = [f for f in files if (Path(a.ghist_dir) / (Path(f).stem + '.npy')).exists()]; print(f'history sidecars found for {len(files)} shards', flush=True)
    if a.allow:
        allow = set(json.load(open(a.allow))); files = [f for f in files if Path(f).stem in allow]; print(f'allow-list {a.allow}: {len(files)} shards', flush=True)
    if a.teams:
        import zipfile, io
        keep_t = {int(x) for x in a.teams.split(',')}
        def team_of(f):
            with zipfile.ZipFile(f) as z: m = np.load(io.BytesIO(z.read('meta.npy')))
            return int(m[4]) if len(m) > 4 else 0
        files = [f for f in files if team_of(f) in keep_t]; print(f'team filter {sorted(keep_t)}: {len(files)} shards', flush=True)
    by = {}
    for f in files:
        by.setdefault(Path(f).parent.name + '/' + ('x'), []).append(f)
    if a.val_list:
        vs = set(json.load(open(a.val_list))); val = [f for f in files if Path(f).stem in vs]; tr = [f for f in files if Path(f).stem not in vs]
        print(f'val-list {a.val_list}: {len(val)} validation shards (excluded from training)', flush=True)
    else:
        val = (sorted(files, key=lambda f: Path(f).stem)[-max(1, len(files) // 10):] if a.val_last else files[::10]); tr = [f for f in files if f not in set(val)]
    W_ = None
    if a.weights:
        wm = json.load(open(a.weights)); W_ = np.array([float(wm.get(Path(f).stem, 0.0)) for f in tr]); keep_ = W_ > 0
        tr = [f for f, k in zip(tr, keep_) if k]; W_ = W_[keep_]; W_ = W_ / W_.sum()
        print(f'weights {a.weights}: {len(tr)} weighted train shards; effective n (1/sum p^2) {1.0 / float((W_ ** 2).sum()):.1f}', flush=True)
    tb = [f for f in tr if 'bc_tb' in f]; tr = tr + tb * (a.tb_repeat - 1)
    print(f'train shards {len(tr)} (tb x{a.tb_repeat}: {len(tb)}) | val shards {len(val)}', flush=True)
    GPLAN = 24 if a.plan_dir else 0
    net = Net(ch=a.ch, blocks=a.blocks, gextra=a.gextra, ghist=(E_.F_HIST if a.ghist_dir else 0), uhist=a.uhist, uattn=a.uattn, gplan=GPLAN).to(dev)
    if a.init:
        st = torch.load(a.init, map_location=dev, weights_only=False)['state']; st = vocab_adapt.adapt_state(st); w = st['g0.0.weight']
        if w.shape[1] < net.g0[0].in_features:
            st['g0.0.weight'] = torch.cat([w, torch.zeros(w.shape[0], net.g0[0].in_features - w.shape[1], device=w.device, dtype=w.dtype)], dim=1)
        init_blocks = 1 + max(int(k.split('.')[1]) for k in st if k.startswith('blocks.'))
        new_blk = lambda k: k.startswith('blocks.') and int(k.split('.')[1]) >= init_blocks
        ms_ = net.load_state_dict(st, strict=False); assert not ms_.unexpected_keys and all(k.split('.')[0] in ('uh1', 'uh2', 'ua_q', 'ua_k', 'ua_v', 'ua_o', 'temb') or new_blk(k) for k in ms_.missing_keys), ms_
        if len(net.blocks) > init_blocks:
            with torch.no_grad():
                for b_ in range(init_blocks, len(net.blocks)): net.blocks[b_].c2.weight.zero_(); net.blocks[b_].c2.bias.zero_()
            print(f'deepened {init_blocks} -> {len(net.blocks)} blocks (new blocks = identity at init)', flush=True)
    tn = TeamNet(net); print(f'params {sum(p.numel() for p in net.parameters())/1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    n_ep = (a.epoch_draws or len(tr)) if W_ is not None else len(tr)
    est_steps = a.epochs * n_ep * 719 // a.bs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=est_steps + 100, pct_start=0.03)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); step = 0; best = 1e9; ep0 = 0
    if a.resume and (out / 'resume.pt').exists():
        R_ = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(R_['state']); opt.load_state_dict(R_['opt']); sched.load_state_dict(R_['sched']); step = R_['step']; best = R_['best']; ep0 = R_['epoch'] + 1
        random.setstate(R_['py_rng']); np.random.set_state(R_['np_rng']); torch.set_rng_state(R_['torch_rng'].cpu())
        print(f'RESUMED from {out / "resume.pt"}: next epoch {ep0}, step {step}', flush=True)
    Dv = load(val[:a.max_val]); vb = [f for f in val if 'bc_tb' in f]; Dvb = load(vb) if vb else None

    def evaluate(D):
        net.eval(); agg = {}; nb = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            for i in range(0, len(D['glob']), 2048):
                b = batch(D, np.arange(i, min(i + 2048, len(D['glob']))), dev); tn.team = b['team']; tn.ghist = b.get('ghist'); tn.uh = b.get('uh'); tn.gplan = b.get('gplan'); lu, lk, lq, m = losses(tn, b); nb += 1
                for k, v in dict(m, lu=lu.item(), lk=lk.item(), lq=lq.item()).items(): agg[k] = agg.get(k, 0) + v
        net.train(); return {k: v / nb for k, v in agg.items()}

    if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    PF = Prefetch(a.load_workers) if a.load_workers > 0 else None
    for ep in range(ep0, a.epochs):
        t0 = time.time()
        if W_ is not None: tr_ep = [tr[i] for i in np.random.choice(len(tr), n_ep, replace=True, p=W_)]
        else: tr_ep = tr[:]; random.shuffle(tr_ep)
        chunks = [tr_ep[c0:c0 + a.chunk] for c0 in range(0, len(tr_ep), a.chunk)]
        if PF is not None: PF.submit(chunks[0])
        for ci, chunk_files in enumerate(chunks):
            c0 = ci * a.chunk
            if PF is not None:
                D = PF.get()
                if ci + 1 < len(chunks): PF.submit(chunks[ci + 1])
            else:
                D = load(chunk_files)
            n = len(D['glob']); perm = np.random.permutation(n)
            if a.early_w != 1.0:
                w_ = np.where(np.asarray(D['glob'][:, 0], np.float32) * 720 < a.early_until, a.early_w, 1.0); perm = np.random.choice(n, n, p=w_ / w_.sum())
            for i in range(n // a.bs):
                b = batch(D, np.sort(perm[i * a.bs:(i + 1) * a.bs]), dev); tn.team = b['team']; tn.ghist = b.get('ghist'); tn.uh = b.get('uh'); tn.gplan = b.get('gplan')
                if tn.uh is not None and a.uh_drop > 0: dm_ = torch.rand(len(tn.uh), device=dev) < a.uh_drop; tn.uh = torch.where(dm_[:, None, None], torch.full_like(tn.uh, C.N_UNIT), tn.uh)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    lu, lk, lq, m = losses(tn, b)
                loss = lu + a.wk * lk + 0.5 * lq
                if not torch.isfinite(loss):
                    open(out / 'NONFINITE', 'w').write(f'epoch {ep} step {step} lu {lu.item()} lk {lk.item()} lq {lq.item()}\n'); raise SystemExit(f'non-finite loss at epoch {ep} step {step}')
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                if step < est_steps + 99: sched.step()
                step += 1
            print(f'  ep {ep} chunk {c0//a.chunk} step {step} lu {lu.item():.3f} lk {lk.item():.3f} lq {lq.item():.3f} {time.time()-t0:.0f}s', flush=True)
        va = evaluate(Dv); vbo = evaluate(Dvb) if Dvb else {}
        row = dict(epoch=ep, step=step, val_all=va, val_tb=vbo, secs=round(time.time() - t0, 1), train_shards=len(tr_ep), val_shards=min(len(val), a.max_val),
                   peak_gpu_mb=round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == 'cuda' else None, codec=C.CODEC_TAG, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        open(out / 'log.jsonl', 'a').write(json.dumps(row) + '\n')
        print(f"EPOCH {ep} val-all lu {va['lu']:.3f} acc {va['acc_unit']:.3f} kind {va['acc_kind']:.3f} | val-tb lu {vbo.get('lu',0):.3f} lk {vbo.get('lk',0):.3f} acc unit {vbo.get('acc_unit',0):.3f} kind {vbo.get('acc_kind',0):.3f}", flush=True)
        ck = {'state': net.state_dict(), 'ch': a.ch, 'blocks': a.blocks, 'gextra': a.gextra, 'ghist': (E_.F_HIST if a.ghist_dir else 0), 'canon_market': a.canon_market, 'uhist': a.uhist, 'uattn': a.uattn, 'gplan': GPLAN, 'epoch': ep, 'val_tb': vbo, 'val_all': va,
              'codec': C.CODEC_TAG, 'n_unit': C.N_UNIT, 'max_q': C.MAX_Q, 'args': vars(a)}
        torch.save(ck, out / 'last.pt'); torch.save(ck, out / f'ck_epoch{ep:02d}.pt')
        score = vbo.get('lu', va['lu']) + vbo.get('lk', va['lk'])
        if score < best: best = score; torch.save(ck, out / 'best.pt')
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=best, epoch=ep,
                        py_rng=random.getstate(), np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), out / 'resume.tmp'); (out / 'resume.tmp').replace(out / 'resume.pt')
        print(f'EPOCH_DONE {ep} {time.time() - t0:.0f}s peak_gpu_mb {row["peak_gpu_mb"]}', flush=True)
    open(out / 'DONE', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')

if __name__ == '__main__':
    main()
