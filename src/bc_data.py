import argparse, glob, gzip, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec as C, encoder as E
from lean_engine import LeanGame

from seals import sealed_ids

TB_LAST_RANK1_EPISODE = 110326264
OUR_TEAM_NAME = 'Smackaveli'
TEAMS = ['TB', 'TP', 'TG', 'TM', 'TS', 'REF', 'TA \U0001F345', 'ymg_aq', 'Otter Vibe',
         'feel the agi', 'TO', 'Sida Zuo',
         'TV', 'TR', 'TD', 'TF']
assert len(TEAMS) <= 16 and TEAMS[5] == 'REF' and TEAMS[10] == 'TO'

def one(args):
    f, team, out = args
    if team == '*':
        res = []
        try:
            d0 = json.load(gzip.open(f, 'rt')) if f.endswith('.gz') else json.load(open(f))
        except Exception:
            return None
        for tm in TEAMS:
            r = one_team(d0, tm, out)
            if r: res.append(r)
        return res or None
    d0 = json.load(gzip.open(f, 'rt')) if f.endswith('.gz') else json.load(open(f))
    r = one_team(d0, team, out)
    return [r] if r else None

def one_team(d, team, out):
    if not isinstance(d, dict) or 'info' not in d or 'steps' not in d or 'TeamNames' not in d['info']:
        return None
    names = d['info']['TeamNames']; eid = int(d['info'].get('EpisodeId') or 0)
    if OUR_TEAM_NAME in names:
        return None
    if team == 'TB' and eid > TB_LAST_RANK1_EPISODE:
        return None
    if team not in names or d['info'].get('seed') is None or names.count(team) != 1 or len(d['steps']) != 720:
        return None
    if any((Path(d) / f'{eid}_{names.index(team)}.npz').exists() for d in [out] + [x for x in os.environ.get('BC_SKIP_DIRS', '').split(':') if x]):
        return None
    s = names.index(team); steps = d['steps']; T = len(steps) - 1
    U, A, K_, S = C.MAX_UNITS, C.N_UNIT, C.N_MK, C.MAX_SLOTS
    grid = np.zeros((T, 2, E.C_TILE, 10, 10), np.float16); units = np.zeros((T, U, E.F_UNIT), np.float16); xy = np.zeros((T, U, 2), np.int8)
    nun = np.zeros(T, np.int8); glob_ = np.zeros((T, E.F_GLOB), np.float16); umask = np.zeros((T, U, A), bool); mmask = np.zeros((T, K_), bool)
    qmax = np.zeros((T, K_), np.int8); a_unit = np.full((T, U), -1, np.int16); a_kind = np.zeros((T, S), np.int8); a_q = np.zeros((T, S), np.int8)
    relabel_u = relabel_k = n_u = n_k = 0
    g = LeanGame(d['info']['seed'])
    for t in range(1, T + 1):
        obs = g.obs(s); k = t - 1
        gr, un, xy_, n, gl = E.encode(obs); um, mm, qm = E.masks(obs)
        grid[k] = gr; units[k] = un; xy[k] = xy_; nun[k] = n; glob_[k] = gl; umask[k] = um; mmask[k] = mm; qmax[k] = np.minimum(qm, 127)
        rec = steps[t][s]['action'] or {}
        rec = C.normalize_units(rec, obs)
        u, slots, _ = C.encode_action(rec)
        for i in range(min(n, U)):
            ai = u[i] if i < len(u) else C.A_PASS
            if ai != C.A_PASS: n_u += 1
            if not um[i, ai]:
                ai = C.A_PASS; relabel_u += 1
            a_unit[k, i] = ai
        for j, (kind, q) in enumerate(slots[:S]):
            if kind != C.MK_NONE: n_k += 1
            if not mm[kind]:
                kind, q = C.MK_NONE, 1; relabel_k += 1
            a_kind[k, j] = kind; a_q[k, j] = max(0, min(C.MAX_Q, q) - 1)
        g.step(steps[t][0]['action'], steps[t][1]['action'])
    ok = g.rewards == [float(x) for x in d['rewards']]
    if not ok:
        return dict(eid=eid, error='replay mismatch')
    me, op = g.rewards[s], g.rewards[1 - s]
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out) / f'{eid}_{s}.npz', grid=grid, units=units, xy=xy, n_units=nun, glob=glob_, umask=umask, mmask=mmask, qmax=qmax,
                        a_unit=a_unit, a_kind=a_kind, a_q=a_q, meta=np.array([eid, s, me, op, TEAMS.index(team) if team in TEAMS else -1], dtype=np.float64),
                        codec=np.array(C.CODEC_TAG))
    return dict(eid=eid, seat=s, team=team, money=me, opp=op, relabel_u=relabel_u, n_u=n_u, relabel_k=relabel_k, n_k=n_k, opp_team=names[1 - s])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('globs', nargs='*'); ap.add_argument('--team', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--files-from', default='', help='JSON list of replay paths (in addition to the globs)')
    ap.add_argument('--exclude-eids', default='', help='JSON list of episode ids never to shard (e.g. the REF select split: its WHOLE episodes stay out of every training set)')
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    sealed = sealed_ids()
    files = sorted(set(f for g in a.globs for f in glob.glob(g)) | (set(json.load(open(a.files_from))) if a.files_from else set()))

    files = [f for f in files if 'tm_20260916_fresh56156662' not in f]
    import re
    keep = []; seen = set(); n_excl = 0
    excl = set(int(x) for x in json.load(open(a.exclude_eids))) if a.exclude_eids else set()
    for f in files:
        m0 = re.search(r'(\d{6,})', Path(f).name)
        if m0 and m0.group(1) in seen:
            continue
        if m0: seen.add(m0.group(1))
        m = re.search(r'(\d{6,})', Path(f).name)
        if m and int(m.group(1)) in sealed:
            print('SEALED, refused:', f); continue
        if m and int(m.group(1)) in excl:
            n_excl += 1; continue
        keep.append(f)
    if a.limit: keep = keep[:a.limit]
    print(f'{len(files)} files, {len(files)-len(keep)-n_excl} sealed refused (sealed list size {len(sealed)}), {n_excl} excluded by --exclude-eids', flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, [(f, a.team, a.out) for f in keep], chunksize=4)):
            if r: rows.extend(r)
            if k % 500 == 0: print(k, len(rows), flush=True)
    bad = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(a.out) / f'MANIFEST_{int(__import__("time").time())}.json', 'w'), indent=0)
    nu = sum(r['n_u'] for r in rows); nk = sum(r['n_k'] for r in rows)
    print(f"{len(rows)} seats written, {len(bad)} replay mismatches; unit labels {nu:,} (relabelled to PASS {sum(r['relabel_u'] for r in rows):,} = {sum(r['relabel_u'] for r in rows)/max(1,nu):.4f}); "
          f"orders {nk:,} (relabelled NONE {sum(r['relabel_k'] for r in rows):,} = {sum(r['relabel_k'] for r in rows)/max(1,nk):.4f}); mean money {np.mean([r['money'] for r in rows]) if rows else 0:,.0f}")
    import collections as _c
    print(dict(_c.Counter(r['team'] for r in rows)))
