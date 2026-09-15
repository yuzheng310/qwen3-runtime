"""Recompute published medians from per-run analyzed metrics (no GPU required)."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(path, counts):
    data = json.loads(path.read_text())
    assert data['complete'] is True
    assert len(data['rows']) == sum(counts.values())
    medians = {}
    for arm, count in counts.items():
        rows = [r for r in data['rows'] if r['arm'] == arm]
        assert len(rows) == count, arm
        assert len({r['repeat'] for r in rows}) == count, arm
        assert all(math.isfinite(r['elapsed_s']) and r['elapsed_s'] > 0 for r in rows)
        metrics = {}
        for key in ('elapsed_s', 'd2h_bytes', 'h2d_bytes', 'prefill_tokens'):
            metrics[key] = statistics.median(r[key] for r in rows)
            assert math.isclose(metrics[key], data['arms'][arm]['median'][key], rel_tol=1e-12)
        medians[arm] = metrics
    return medians


def reduction(treatment, control, key):
    return 100 * (1 - treatment[key] / control[key])


def main():
    base = ROOT / 'bench/results/kv-tiered'
    same = summarize(base / 'comparison.json', {'S4': 3, 'T4': 3, 'V0': 3, 'V4': 3})
    async_rows = summarize(base / 'async-comparison.json', {'S': 3, 'PS': 3, 'AS': 3, 'T': 1})
    print(json.dumps({
        'same_budget_medians': same,
        'T4_vs_V4_reduction_pct': {k: reduction(same['T4'], same['V4'], k)
                                 for k in ('elapsed_s', 'd2h_bytes', 'prefill_tokens')},
        'T4_vs_S4_reduction_pct': {k: reduction(same['T4'], same['S4'], k)
                                 for k in ('elapsed_s', 'd2h_bytes')},
        'async_medians': async_rows,
        'AS_vs_S_time_reduction_pct': reduction(async_rows['AS'], async_rows['S'], 'elapsed_s'),
        'AS_vs_PS_time_reduction_pct': reduction(async_rows['AS'], async_rows['PS'], 'elapsed_s'),
        'scope': 'Descriptive sample medians; a negative reduction means an increase.'
    }, indent=2))


if __name__ == '__main__':
    main()
