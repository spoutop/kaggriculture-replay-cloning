import argparse, collections, glob, gzip, json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'fid'))
ROOT = os.environ.get('KGR_ROOT', '.')
PANEL = os.environ.get('GT_PANEL') or os.path.join(HERE, 'GT9_PANEL.json')
LOSS_SHARE = {'TG': 10.3, 'TF': 8.4, 'TV': 8.4, '吃白饭的大肥鱼': 7.5, 'TK 🏆': 7.5,
              'tetsuya & yuanzhe & guoqin': 6.5, 'TQ': 6.5, 'TD': 5.6, 'TP': 4.7}
ALIAS = {'tetsuya & yuanzhe & guoqi': 'tetsuya & yuanzhe & guoqin'}
MIN_REF_SUB = 56409623

def build():
    subs = {}
    for f in glob.glob(f'{ROOT}/replays/**/EPISODES_*.json', recursive=True) + glob.glob(f'{ROOT}/ws/maclearn/rank1_20260923/ladder/ep/EP_*.json'):
        try: d = json.load(open(f))
        except Exception: continue
        for e in (d.get('episodes', []) if isinstance(d, dict) else []):
            for a in e.get('agents', []): subs[(int(e['id']), a.get('index', 0) or 0)] = (a.get('submissionId'), a.get('initialScore'))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, 'ws/maclearn/rank1_20260923/clonegap/abo/opening_rows.jsonl'))]
    by = collections.defaultdict(dict)
    for r in rows: by[r['eid']][r['seat']] = r
    games = []
    for eid, s in by.items():
        if 0 not in s or 1 not in s: continue
        for me in (0, 1):
            opp = ALIAS.get(s[1 - me]['team'], s[1 - me]['team'])
            if s[me]['team'] != 'REF' or opp not in LOSS_SHARE: continue
            ds, dscore = subs.get((eid, me), (None, None)); os_, oscore = subs.get((eid, 1 - me), (None, None))
            if ds is None or ds < MIN_REF_SUB: continue
            games.append(dict(eid=eid, file=s[me]['file'], ref_seat=me, opp_team=opp, ref_sub=ds, opp_sub=os_, ref_rating=dscore, opp_rating=oscore,
                              ref_real=[s[me]['me'], s[me]['them']], ref_real_win=1.0 if s[me]['me'] > s[me]['them'] else (0.5 if s[me]['me'] == s[me]['them'] else 0.0)))
    n = collections.Counter(g['opp_team'] for g in games); tot = sum(LOSS_SHARE[t] for t in n)
    for g in games: g['weight'] = LOSS_SHARE[g['opp_team']] / tot / n[g['opp_team']]
    games.sort(key=lambda g: (g['opp_team'], g['eid']))
    meta = dict(built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), min_ref_sub=MIN_REF_SUB, loss_share=LOSS_SHARE, n_by_team=dict(n),
                team_weight={t: LOSS_SHARE[t] / tot for t in n}, missing_teams=[t for t in LOSS_SHARE if t not in n],
                ref_real_wr={t: float(np.mean([g['ref_real_win'] for g in games if g['opp_team'] == t])) for t in n})
    meta['ref_real_weighted_wr'] = float(sum(g['weight'] * g['ref_real_win'] for g in games))
    json.dump(dict(meta=meta, games=games), open(PANEL, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1, ensure_ascii=False)); print(len(games), 'games ->', PANEL)

def _mk(spec, seed, seat):
    import numpy as np_
    if spec.startswith('endgame:'):
        rules, inner = spec[8:].split('|', 1)
        sys.path.insert(0, os.environ.get('GT9_EGDIR') or os.path.join(ROOT, 'ws/maclearn/rank1_20260923/clonegap/matchup3')); import endgame as EG_
        return EG_.EndgamePolicy(_mk(inner, seed, seat), rules)
    if spec.startswith('book'):
        head, rest = spec.split(':', 1); bp, inner = rest.split('|', 1); o = {}
        if '[' in head: o = dict(kv.split('=') for kv in head[head.index('[') + 1:head.index(']')].split(','))
        sys.path.insert(0, os.environ.get('GT9_BOOKDIR') or os.path.join(ROOT, 'ws/maclearn/rank1_20260923/clonegap/book')); import ref_book as DB_
        return DB_.BookPolicy(_mk(inner, seed, seat), bp, keys=('full',), cutoff=int(o.get('cut', 10 ** 9)), min_count=int(o.get('k', 1)), sticky=o.get('sticky') == '1', opp_detect=o.get('opp') == '1' or o.get('hyb') == '1', soft=float(o.get('soft', 0)), hybrid=o.get('hyb') == '1')
    if spec.startswith('bundle:'):
        import importlib.util
        d_ = os.path.abspath(spec[7:]); sys.path.insert(0, d_)
        sp_ = importlib.util.spec_from_file_location('bundle_main_' + str(abs(hash(d_))), os.path.join(d_, 'main.py')); m_ = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(m_)
        return m_.agent
    import np_policy as NP, codec as C
    if spec.endswith('.ens.json'):
        from np_ensemble import EnsNpPolicy, spec_correction
        sp = json.load(open(spec)); mem = [m if os.path.isabs(m) else os.path.join(os.path.dirname(spec), m) for m in sp['members']]
        for m in mem: assert str(np_.load(m)['_codec']) == C.CODEC_TAG, m
        return EnsNpPolicy(mem, sp.get('weights'), spec_correction(sp, spec))
    z = np_.load(spec); assert str(z['_codec']) == C.CODEC_TAG
    return NP.NpPolicy(spec, mode='argmax', temp=1.0, team=0, salt=0)

class TapeSteps:
    def __init__(self, tp):
        self.tp = tp; self.g = tp['ghost_seat']; self.c = tp['cand_seat']
    def __len__(self): return len(self.tp['actions']) + 1
    def __getitem__(self, t):
        tp = self.tp; out = [{}, {}]
        if t >= 1:
            out[self.g]['action'] = tp['actions'][t - 1]; out[self.c]['action'] = tp['ref_actions'][t - 1]
            f = tp['forced'][t - 1]; farms = [None, None]; farms[self.g] = f['farm']
            out[0]['observation'] = dict(farms=farms, town=f['town']); out[self.g].setdefault('observation', {})
            if self.g == 0: out[0]['observation']['private'] = f['private']
            else: out[1]['observation'] = dict(private=f['private'])
        return out

def _mset(a, obs):
    import codec as C
    me = int(obs['player']); mm, _ = C.market_mask(obs['farms'][me], obs['private'], obs['market']['prices']); out = []
    for o in ((a or {}).get('market') or [])[:10] if isinstance(a, dict) else []:
        if isinstance(o, list) and o:
            (k, q), _ok = C.encode_order(o)
            out.append((k, bool(mm[k]) if k != C.MK_NONE else False))
    return out

def play_one(task):
    name, spec, g, mode = task
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
    from lean_engine import LeanGame
    if g.get('tape'):
        tp = json.load(gzip.open(g['tape'], 'rt')); assert mode.startswith('forced'), 'tape files support forced modes only'
        d = dict(info=dict(seed=tp['seed']), rewards=tp['recorded_final']); steps = TapeSteps(tp); me = tp['cand_seat']; op = 1 - me
    else:
        d = json.load(gzip.open(g['file'], 'rt')) if g['file'].endswith('.gz') else json.load(open(g['file']))
        me = g.get('cand_seat', g['ref_seat']); op = 1 - me; steps = d['steps']
    K = 0; pol = None
    TF = False
    if spec.startswith('tf:'): TF = True; K = 10 ** 9; pol = _mk(spec[3:], 0, me)
    elif spec == 'reftape': K = 10 ** 9
    elif spec.startswith('tape') and ':' in spec: K = int(spec[4:spec.index(':')]); pol = _mk(spec[spec.index(':') + 1:], 0, me)
    else: pol = _mk(spec, 0, me)
    gm = LeanGame(d['info']['seed']); cfg = gm.env.configuration; t = 0
    first_opp_state = first_opp_money = None; rej_orders = legal_rec_orders = 0; rej_late = 0; worst = 0.0; errors = 0
    import copy
    offset = 0.0; town_div = 0; first_town_div = None
    STATS = os.environ.get('GT9_STATS') == '1'
    import codec as C_
    day_stats = []; sells_day = {}; first_mkt = first_unit = None; tf_m = [0] * 30; tf_u = [0] * 30; tf_n = [0] * 30; tf_act = [0] * 30
    def _ms(a_, ob_):
        me_ = int(ob_['player']); mm_, _ = C_.market_mask(ob_['farms'][me_], ob_['private'], ob_['market']['prices']); out_ = []
        for o in ((a_ or {}).get('market') or [])[:10] if isinstance(a_, dict) else []:
            if isinstance(o, list) and o:
                (k_, q_), _ok = C_.encode_order(o)
                if k_ != C_.MK_NONE and mm_[k_]: out_.append(k_)
        return sorted(out_)
    def _us(a_, ob_):
        n_ = C_.normalize_units(a_ or {}, ob_); return json.dumps([n_.get('farmer')] + list(n_.get('hands') or []))
    while not gm.done:
        t += 1
        obs_me = gm.obs(me); obs_op = gm.obs(op)
        if STATS and (t - 1) % 24 == 0:
            fm = obs_me['farms'][me]; tl = [x for row in fm['tiles'] for x in row if isinstance(x, dict)]
            day_stats.append(dict(money=float(fm['money']), opp_money=float(obs_me['farms'][op]['money']), quads=len(fm.get('unlocked_quadrants', [])), hands=len(fm['hands']),
                                  animals=collections.Counter(x['animal'] for x in tl if 'animal' in x), plants=collections.Counter(x['crop'] for x in tl if x.get('kind') == 'PLANT'),
                                  coops=sum(1 for x in tl if x.get('kind') == 'COOP'), pastures=sum(1 for x in tl if x.get('kind') == 'PASTURE'), shed=dict(obs_me['private']['shed'])))
        rec_op = steps[t][op].get('action') if t < len(steps) else None
        ro = (steps[t - 1][op].get('observation') or {}) if not g.get('tape') else {}
        if ro.get('farms'):
            lf, rf = obs_op['farms'][op], ro['farms'][op]
            if first_opp_money is None and float(lf['money']) != float(rf['money']): first_opp_money = t - 1
            if first_opp_state is None and (lf['tiles'] != rf['tiles'] or lf['money'] != rf['money'] or ro.get('town') != obs_op.get('town')): first_opp_state = t - 1

        if isinstance(rec_op, dict) and rec_op.get('market') and ro.get('farms') and 'private' in ro:
            live = _mset(rec_op, obs_op); rec = _mset(rec_op, ro)
            for (k1, ok_live), (k2, ok_rec) in zip(live, rec):
                if ok_rec: legal_rec_orders += 1; rej_orders += (not ok_live); rej_late += (not ok_live) and t - 1 >= 480
        if STATS or TF:
            rec_me = steps[t][me].get('action') if t < len(steps) else None
            if TF or t - 1 >= K:
                mine_ = pol(obs_me, cfg) if TF else None
        if t - 1 < K: a = steps[t][me].get('action')
        else:
            t0 = time.time()
            try: a = pol(obs_me, cfg)
            except Exception as e:
                errors += 1; a = {'farmer': ['PASS'], 'hands': [], 'market': []}
            worst = max(worst, time.time() - t0)
        if STATS:
            for o in ((a or {}).get('market') or [])[:10] if isinstance(a, dict) else []:
                if isinstance(o, list) and len(o) > 2 and o[0] == 'SELL' and isinstance(o[2], (int, float)):
                    sells_day.setdefault((t - 1) // 24, collections.Counter())[o[1]] += min(int(o[2]), 1000)
        if STATS and not TF and t < len(steps) and (first_mkt is None or first_unit is None):
            if first_mkt is None and _ms(a, obs_me) != _ms(rec_me, obs_me): first_mkt = t - 1
            if first_unit is None and _us(a, obs_me) != _us(rec_me, obs_me): first_unit = t - 1
        if TF and t < len(steps):
            dd_ = (t - 1) // 24; x1, x2 = _ms(mine_, obs_me), _ms(rec_me, obs_me); tf_n[dd_] += 1
            if x1 or x2: tf_act[dd_] += 1; tf_m[dd_] += (sorted(set(x1)) != sorted(set(x2)))
            tf_u[dd_] += (_us(mine_, obs_me) != _us(rec_me, obs_me))
            if first_mkt is None and x1 != x2: first_mkt = t - 1
            if first_unit is None and _us(mine_, obs_me) != _us(rec_me, obs_me): first_unit = t - 1
        acts = [None, None]; acts[me] = a; acts[op] = rec_op
        gm.step(acts[0], acts[1])
        FN = int(mode[6:]) if mode.startswith('forced') and mode[6:].isdigit() else 1
        if mode.startswith('forced') and t < len(steps) and (t % FN == 0 or t == len(steps) - 1):

            r0 = steps[t][0]['observation']; rop = steps[t][op]['observation']
            offset += float(gm.state[0].observation.farms[op]['money']) - float(r0['farms'][op]['money'])
            if gm.state[0].observation.town != r0['town']:
                town_div += 1
                if first_town_div is None: first_town_div = t
            gm.state[0].observation.farms[op] = copy.deepcopy(r0['farms'][op])
            gm.state[op].observation.private = copy.deepcopy(rop['private'])
            tw = copy.deepcopy(r0['town'])
            for s_ in gm.state: s_.observation.town = tw
    FM = mode.startswith('forced')
    mine = float(gm.state[0].observation.farms[me]['money']) if FM else gm.rewards[me]
    theirs_rec = float(gm.state[0].observation.farms[op]['money']) if FM else gm.rewards[op]
    theirs = theirs_rec + offset if FM else gm.rewards[op]
    if hasattr(pol, 'stats') and isinstance(getattr(pol, 'stats'), dict) and 'feed_blocked' in pol.stats: extra_eg = dict(endgame_stats=dict(pol.stats))
    else: extra_eg = {}
    if hasattr(pol, 'hits_by_day'): extra_book = dict(book_hits=dict(pol.stats), book_hits_by_day={int(k): v for k, v in pol.hits_by_day.items()})
    else: extra_book = {}
    extra = dict(day_stats=[dict(d_, animals=dict(d_['animals']), plants=dict(d_['plants'])) for d_ in day_stats], first_market_diff=first_mkt, first_unit_diff=first_unit, sells_by_day={int(k): dict(v) for k, v in sells_day.items()}) if (STATS or TF) else {}
    extra.update(extra_book); extra.update(extra_eg)
    if TF: extra.update(tf_kindset_mis=tf_m, tf_active=tf_act, tf_unit_mis=tf_u, tf_steps=tf_n)
    return dict(extra, agent=name, spec=spec, mode=mode, cand_seat=me, ghost_team=g.get('ghost_team', g['opp_team']), ghost_recorded_money=theirs_rec, offset=offset, win_vs_recorded=1.0 if mine > theirs_rec else (0.5 if mine == theirs_rec else 0.0),
                town_diverged_steps=town_div, first_town_diverge=first_town_div, eid=g['eid'], opp_team=g['opp_team'], weight=g['weight'], ref_seat=me, seed=d['info']['seed'], mine=mine, theirs=theirs,
                win=1.0 if mine > theirs else (0.5 if mine == theirs else 0.0), margin=mine - theirs, opp_recorded=float(d['rewards'][op]), ref_recorded=float(d['rewards'][me]),
                opp_live_minus_recorded=theirs - float(d['rewards'][op]), first_opp_money_diverge=first_opp_money, first_opp_state_diverge=first_opp_state,
                opp_orders_rejected=rej_orders, opp_orders_legal_in_recording=legal_rec_orders, opp_orders_rejected_late=rej_late, errors=errors, worst_turn_ms=round(worst * 1000, 1))

def play(a):
    P = json.load(open(PANEL)); games = P['games']
    if a.tapes:
        M = json.load(open(os.path.join(a.tapes, 'MANIFEST.json')))['tapes']; byk = {(m['tape_id'], 1 - m['ghost_seat']): m for m in M}
        for g in games: g['tape'] = os.path.join(a.tapes, byk[(g['eid'], g.get('cand_seat', g['ref_seat']))]['path'])
    if a.split: games = [g for g in games if g.get('split') == a.split]
    if a.subset: keep = set(tuple(x) for x in json.load(open(a.subset))); games = [g for g in games if (g['eid'], g.get('cand_seat', g['ref_seat'])) in keep]
    if a.teams: games = [g for g in games if g['opp_team'] in a.teams.split(',')]
    games = games[:a.limit] if a.limit else games
    agents = [x.split('=', 1) for x in a.agent]
    import random
    order = list(range(len(games))); random.Random(20260925).shuffle(order)
    tasks = [(n, s, games[i], a.mode) for n, s in agents for i in order] if a.policy_major else [(n, s, games[i], a.mode) for i in order for n, s in agents]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): r = json.loads(l); done.add((r['agent'], r['eid'], r['ref_seat']))
    tasks = [t for t in tasks if (t[0], t[2]['eid'], t[2].get('cand_seat', t[2]['ref_seat'])) not in done]
    print('mode', a.mode, flush=True)
    print(len(tasks), 'games to play', flush=True); t0 = time.time()
    with Pool(a.workers) as p, open(a.out, 'a') as fo:
        for i, r in enumerate(p.imap_unordered(play_one, tasks, chunksize=1)):
            fo.write(json.dumps(r, ensure_ascii=False) + '\n'); fo.flush()
            if i % 20 == 0: print(i, r['agent'], r['opp_team'], r['win'], int(r['margin']), f'{time.time() - t0:.0f}s', flush=True)
    print('DONE', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)

def score(files, B=4000, split=''):
    rows = list({(r['agent'], r['eid'], r['ref_seat']): r for r in (json.loads(l) for f in files for l in open(f))}.values())
    P = json.load(open(PANEL)); tw = P['meta']['team_weight']
    rng = np.random.default_rng(0); out = {}
    for ag in sorted(set(r['agent'] for r in rows)):
        R = [r for r in rows if r['agent'] == ag]; teams = sorted(set(r['opp_team'] for r in R))
        per = {t: np.array([r['win'] for r in R if r['opp_team'] == t]) for t in teams}
        if split:
            sp = {g['eid']: g.get('split') for g in P['games']}; R = [r for r in R if sp.get(r['eid']) == split]; teams = sorted(set(r['opp_team'] for r in R))
            per = {t: np.array([r['win'] for r in R if r['opp_team'] == t]) for t in teams}
        wsum = sum(tw[t] for t in teams); K_ = 20.0
        pool = np.mean(np.concatenate([per[t] for t in teams]))
        v_raw = sum(tw[t] * per[t].mean() for t in teams) / wsum
        v = sum(tw[t] * (len(per[t]) * per[t].mean() + K_ * pool) / (len(per[t]) + K_) for t in teams) / wsum
        bm = {t: per[t][rng.integers(0, len(per[t]), (B, len(per[t])))].mean(1) for t in teams}
        bpool = sum(bm[t] * len(per[t]) for t in teams) / sum(len(per[t]) for t in teams)
        bs = sum(tw[t] * (len(per[t]) * bm[t] + K_ * bpool) / (len(per[t]) + K_) for t in teams) / wsum
        mg = sum(tw[t] * np.mean([r['margin'] for r in R if r['opp_team'] == t]) for t in teams) / wsum
        div = dict(first_opp_state_diverge_median=float(np.median([r['first_opp_state_diverge'] if r['first_opp_state_diverge'] is not None else 720 for r in R])),
                   first_opp_money_diverge_median=float(np.median([r['first_opp_money_diverge'] if r['first_opp_money_diverge'] is not None else 720 for r in R])),
                   opp_orders_rejected_share=float(sum(r['opp_orders_rejected'] for r in R) / max(1, sum(r['opp_orders_legal_in_recording'] for r in R))),
                   opp_money_live_minus_recorded_mean=float(np.mean([r['opp_live_minus_recorded'] for r in R])))
        out[ag] = dict(n=len(R), weighted_wr=round(float(v), 4), weighted_wr_unshrunk=round(float(v_raw), 4), wr_vs_recorded=round(float(np.mean([r['win_vs_recorded'] for r in R])), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                       weighted_margin=int(mg), unweighted_wr=round(float(np.mean([r['win'] for r in R])), 4),
                       per_team={t: [round(float(per[t].mean()), 3), len(per[t])] for t in teams}, divergence=div, errors=int(sum(r['errors'] for r in R)))
        print(ag, json.dumps(out[ag], ensure_ascii=False))

    ags = sorted(out); pair = {}
    key = lambda r: (r['eid'], r['ref_seat'])
    W = {ag: {key(r): r for r in rows if r['agent'] == ag} for ag in ags}
    for i, a1 in enumerate(ags):
        for a2 in ags[i + 1:]:
            common = sorted(set(W[a1]) & set(W[a2]))
            if split:
                sp = {g['eid']: g.get('split') for g in P['games']}; common = [k for k in common if sp.get(k[0]) == split]
            if len(common) < 20: continue
            teams = sorted(set(W[a1][k]['opp_team'] for k in common))
            d = {t: np.array([W[a2][k]['win'] - W[a1][k]['win'] for k in common if W[a1][k]['opp_team'] == t]) for t in teams}
            wsum = sum(tw[t] for t in teams); v = sum(tw[t] * d[t].mean() for t in teams) / wsum
            bs = sum(tw[t] * d[t][rng.integers(0, len(d[t]), (B, len(d[t])))].mean(1) for t in teams) / wsum
            pair[f'{a2}-{a1}'] = dict(n=len(common), d_weighted_wr=round(float(v), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
            print('paired', f'{a2} - {a1}', pair[f'{a2}-{a1}'])
    out['_paired'] = pair
    for ag in ags:
        R = [r for r in rows if r['agent'] == ag]
        out[ag]['divergence']['town_diverged_steps_mean'] = float(np.mean([r.get('town_diverged_steps', 0) for r in R]))
        out[ag]['divergence']['first_town_diverge_median'] = float(np.median([r['first_town_diverge'] if r.get('first_town_diverge') is not None else 720 for r in R]))
        out[ag]['divergence']['offset_mean'] = float(np.mean([r.get('offset', 0) for r in R]))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd')
    sp.add_parser('build')
    p = sp.add_parser('play'); p.add_argument('--agent', nargs='+', required=True); p.add_argument('--workers', type=int, default=6); p.add_argument('--out', required=True); p.add_argument('--limit', type=int, default=0); p.add_argument('--mode', default='forced', choices=['forced', 'tape', 'forced24', 'forced72', 'forced6']); p.add_argument('--teams', default='', help='comma list of team names to restrict the panel'); p.add_argument('--policy-major', type=int, default=1); p.add_argument('--tapes', default=''); p.add_argument('--split', default=''); p.add_argument('--subset', default='', help='JSON list of [eid, cand_seat]')
    s = sp.add_parser('score'); s.add_argument('files', nargs='+'); s.add_argument('--json', default=''); s.add_argument('--split', default='')
    a = ap.parse_args()
    if a.cmd == 'build': build()
    elif a.cmd == 'play': play(a)
    elif a.cmd == 'score':
        o = score(a.files, split=a.split)
        if a.json: json.dump(o, open(a.json, 'w'), indent=1, ensure_ascii=False)
