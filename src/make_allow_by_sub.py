import glob, json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get('KGR_ROOT', '.') + '/replays'
LISTINGS = glob.glob(f'{R}/frontier_20260923/meta/EPISODES_*.json') + glob.glob(f'{R}/top_20260918/meta/EPISODES_*.json') + \
    glob.glob(os.path.join(R, 'ladder/ep/EP_*.json'))
seat_sub = {}; subinfo = {}; teams = {}
for f in LISTINGS:
    try: d = json.load(open(f))
    except Exception: continue
    for t in d.get('teams', []) or []: teams[t['id']] = t.get('teamName')
    for s in d.get('submissions', []) or []:
        if isinstance(s, dict): subinfo.setdefault(s['id'], {}).update(uploaded=s.get('dateSubmitted'), teamId=s.get('teamId'))
    for e in d.get('episodes', []) or []:
        for i, a in enumerate(e.get('agents', []) or []):
            seat = int(a.get('index', i) or 0)
            seat_sub[(int(e['id']), seat)] = (a.get('submissionId'), a.get('teamId'))
by = collections.defaultdict(lambda: collections.defaultdict(list)); unassigned = collections.Counter()
for d in ('ref', 'multi', 'frontier'):
    for f in glob.glob(os.path.join(HERE, 'data', d, '*.npz')):
        stem = os.path.basename(f)[:-4]; eid, seat = map(int, stem.split('_'))
        r = seat_sub.get((eid, seat))
        if r is None or r[0] is None: unassigned[d] += 1; continue
        by[r[0]][d].append(stem)
os.makedirs(os.path.join(HERE, 'allow', 'by_sub'), exist_ok=True); index = {}
for sub, dd in by.items():
    allst = sorted({s for v in dd.values() for s in v})
    json.dump(allst, open(os.path.join(HERE, 'allow', 'by_sub', f'sub{sub}.json'), 'w'))
    tid = subinfo.get(sub, {}).get('teamId'); index[str(sub)] = dict(team=teams.get(tid, str(tid)), teamId=tid, uploaded=subinfo.get(sub, {}).get('uploaded'),
                                                                     shards={k: len(v) for k, v in dd.items()}, total=len(allst))
json.dump(index, open(os.path.join(HERE, 'allow', 'by_sub', 'INDEX.json'), 'w'), indent=1)
print(f'{len(LISTINGS)} listings, {len(seat_sub):,} (episode, seat) -> submission; {len(index)} submissions with shards; unassigned shards {dict(unassigned)}')
top = sorted(index.items(), key=lambda kv: -kv[1]['total'])[:25]
for sub, v in top: print(f"  {v['team'][:22]:<22s} {sub} up {str(v['uploaded'])[:16]} shards {v['total']:4d} {v['shards']}")
