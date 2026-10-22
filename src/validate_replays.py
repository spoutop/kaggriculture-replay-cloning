import argparse, collections, glob, gzip, json, os, re
from multiprocessing import get_context

def one(f):
    try:
        d = json.load(gzip.open(f, 'rt')) if f.endswith('.gz') else json.load(open(f))
        info = d.get('info') or {}; tn = info.get('TeamNames') or []; rw = d.get('rewards')
        fid = int(re.search(r'(\d{8,})', os.path.basename(f)).group(1))
        checks = dict(steps720=len(d.get('steps', [])) == 720, rewards=isinstance(rw, list) and len(rw) == 2 and all(r is not None for r in rw),
                      teams=len(tn) == 2 and all(tn), seed=info.get('seed') is not None, id_match=int(info.get('EpisodeId') or -1) == fid)
        return dict(f=os.path.basename(f), ok=all(checks.values()), checks=checks, teams=tn)
    except Exception as e:
        return dict(f=os.path.basename(f), ok=False, error=str(e)[:200], teams=[])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('globs', nargs='+'); ap.add_argument('--workers', type=int, default=12); ap.add_argument('--out', default='')
    a = ap.parse_args(); files = sorted({f for g in a.globs for f in glob.glob(g)})
    with get_context('spawn').Pool(a.workers) as p: rows = p.map(one, files, chunksize=4)
    bad = [r for r in rows if not r['ok']]; seats = collections.Counter(t for r in rows if r['ok'] for t in r['teams'])
    print(f'{len(rows)} files, valid {len(rows) - len(bad)}, invalid {len(bad)}')
    for r in bad[:10]: print('  INVALID', r)
    print('seats of the top teams:', {t: seats[t] for t in ['REF', 'TM', 'TG', 'TV', 'TR', 'TD', 'TP', 'TF']})
    if a.out: json.dump(dict(n=len(rows), invalid=bad, seats=dict(seats)), open(a.out, 'w'), indent=0)
