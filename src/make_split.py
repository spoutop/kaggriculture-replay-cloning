import argparse, glob, io, json, os, re, zipfile
from datetime import datetime, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get('KGR_ROOT', '.') + '/replays'
META_DIRS = [f'{R}/frontier_20260923/meta', f'{R}/top_20260918/meta']

def ts(s):
    s = s.rstrip('Z')
    if '.' in s:
        a, b = s.split('.'); s = a + '.' + b[:6]; return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)

def listing(sub):
    for d in META_DIRS:
        p = os.path.join(d, f'EPISODES_{sub}.json')
        if os.path.exists(p): return json.load(open(p)), p
    return None, None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--team-token', type=int, default=5); ap.add_argument('--team-id', type=int, default=16732748)
    ap.add_argument('--shards', default='data/ref'); ap.add_argument('--current', required=True); ap.add_argument('--recent', default='')
    ap.add_argument('--lam', type=float, default=2.0, help='recency weight w = exp(-lam * days between the version upload and the newest upload)')
    ap.add_argument('--select-frac', type=float, default=0.2)
    a = ap.parse_args()
    cur = [int(x) for x in a.current.split(',')]; rec = [int(x) for x in a.recent.split(',') if x]
    shards = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(HERE, a.shards, '*.npz'))}
    by_eid = {}
    for stem in shards:
        eid, seat = map(int, stem.split('_')); by_eid.setdefault(eid, []).append(stem)
    info = {}; sub_upload = {}; per_sub = {}
    for s in cur + rec:
        L, p = listing(s)
        if L is None: print('NO LISTING', s); continue
        subrow = [x for x in L.get('submissions', []) if x.get('id') == s]
        if subrow: sub_upload[s] = subrow[0]['dateSubmitted']
        per_sub[s] = []
        for e in L['episodes']:
            if e.get('state') != 'COMPLETED' or e.get('type') != 'EPISODE_TYPE_PUBLIC': continue
            me = [ag for ag in e['agents'] if ag.get('submissionId') == s]
            if not me: continue
            seat = int(me[0].get('index', 0) or 0); stem = f"{int(e['id'])}_{seat}"
            if stem not in shards: continue
            info[stem] = dict(sub=s, end=e['endTime'], seat=seat); per_sub[s].append(stem)
        print(f'sub {s}: uploaded {sub_upload.get(s)}; {len(per_sub[s])} shards matched (listing {p}, fetched {L.get("fetched_utc")})')
    os.makedirs(os.path.join(HERE, 'allow'), exist_ok=True); os.makedirs(os.path.join(HERE, 'weights'), exist_ok=True)
    for s, v in per_sub.items():
        json.dump(sorted(v), open(os.path.join(HERE, 'allow', f'ref_sub{s}.json'), 'w'))
    curstems = sorted((st for s in cur for st in per_sub.get(s, [])), key=lambda st: ts(info[st]['end']))
    n_sel = int(round(a.select_frac * len(curstems))); train = curstems[:len(curstems) - n_sel]; select = curstems[len(curstems) - n_sel:]
    cutoff = info[select[0]]['end']; last_train = info[train[-1]]['end']; data_end = info[curstems[-1]]['end']
    json.dump(train, open(os.path.join(HERE, 'allow', 'ref_cur_train.json'), 'w')); json.dump(select, open(os.path.join(HERE, 'allow', 'ref_cur_select.json'), 'w'))
    json.dump(sorted(train + select), open(os.path.join(HERE, 'allow', 'ref_cur_all.json'), 'w'))
    newest = max(ts(sub_upload[s]) for s in cur)
    w = {}
    for s in cur + rec:
        if s not in per_sub: continue
        dt = max(0.0, (newest - ts(sub_upload[s])).total_seconds() / 86400.0) if s not in cur else 0.0
        for st in per_sub[s]:
            if st in select: continue
            w[st] = float(np.exp(-a.lam * dt))
    json.dump(w, open(os.path.join(HERE, 'weights', 'refB_weights.json'), 'w'))
    tot = sum(w.values()); cur_w = sum(v for k, v in w.items() if info[k]['sub'] in cur)
    per_ver = {str(s): dict(uploaded=sub_upload.get(s), shards=len([x for x in per_sub.get(s, []) if x not in select]),
                             weight=float(np.exp(-a.lam * max(0.0, (newest - ts(sub_upload[s])).total_seconds() / 86400.0))) if s not in cur else 1.0) for s in cur + rec if s in per_sub}
    split = dict(written_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), team='REF', team_token=a.team_token, current_version=cur,
                 version_decision='56463942 + 56468867 pooled as ONE version family (see TRACKA_STATUS.md A3: openings identical to step 20, per-day action-class TVD not distinguishable from the permutation null, p=0.154; NOT byte-identical: an hour-22 day-0 decision differs vs the same opponent submission)',
                 train_select=dict(n_current=len(curstems), n_train=len(train), n_select=len(select), select_cutoff_endTime=cutoff, last_train_endTime=last_train,
                                   rule='chronological by episode endTime (ListEpisodes): train = oldest 80%, select = newest 20% (select_cutoff_endTime inclusive)'),
                 holdout=dict(rule='UNTOUCHED fidelity holdout = public games of the REF version family with endTime > holdout_after_endTime, fetched only AFTER A5 checkpoint selection; never used for training or checkpoint choice',
                              holdout_after_endTime=data_end),
                 recency=dict(lam_per_day=a.lam, rule='w = exp(-lam * days(newest current upload - version upload)); current pair w = 1; select-split shards excluded', per_version=per_ver,
                              total_weight=tot, current_weight_share=cur_w / tot if tot else None))
    json.dump(split, open(os.path.join(HERE, 'SPLIT.json'), 'w'), indent=1)
    print(json.dumps(split, indent=1))

if __name__ == '__main__':
    main()
