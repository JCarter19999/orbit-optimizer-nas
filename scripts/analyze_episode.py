from pathlib import Path
import json
import sys

def read_jsonl(p: Path):
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_first_key(d, candidates):
    keys = set(d.keys())
    for k in candidates:
        if k in keys:
            return k
    return None


def main():
    if len(sys.argv) < 2:
        print('usage: python scripts/analyze_episode.py path/to/episode.jsonl')
        return
    path = Path(sys.argv[1])
    rows = read_jsonl(path)
    if not rows:
        print('no rows')
        return

    agent_key = find_first_key(rows[0], ['agents','interceptors','vehicles'])
    target_key = find_first_key(rows[0], ['targets_truth','targets','truth_targets','gt_targets','objects'])
    matches_key = find_first_key(rows[0], ['matches','assignments','engagement_matches'])

    n_agents = len(rows[0].get(agent_key, [])) if agent_key else 0
    n_targets = len(rows[0].get(target_key, [])) if target_key else 0

    print(f'agents key: {agent_key}  targets key: {target_key}  matches key: {matches_key}')
    print(f'n_agents={n_agents}  n_targets={n_targets}')

    assigned = set()
    assigned_times = {}
    last_matches = []
    for i, r in enumerate(rows):
        ms = r.get(matches_key, []) if matches_key else []
        if ms:
            last_matches = ms
        for a,t in ms:
            assigned.add(t)
            assigned_times.setdefault(t, []).append(i)

    final_row = rows[-1]
    final_targets = final_row.get(target_key, []) if target_key else []
    final_active = {i: bool(g.get('active', True)) for i,g in enumerate(final_targets)}

    print('\ntargets ever assigned (indices):', sorted(list(assigned)))
    never_assigned = [i for i in range(n_targets) if i not in assigned]
    print('targets never assigned (indices):', never_assigned)
    print('\nfinal active flags:')
    for i in range(n_targets):
        print(f'  target {i}: active={final_active.get(i,False)}')

    print('\nlast non-empty matches (if any):', last_matches)

    # show a short table of assigned counts
    print('\nassignment counts:')
    for t in range(n_targets):
        print(f'  target {t}: assigned {len(assigned_times.get(t, []))} times')

if __name__ == "__main__":
    main()
