#!/usr/bin/env python3
"""Rebuild README figures from the published benchmark JSON (matplotlib required)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE = '#2563EB'
ORANGE = '#D97706'
INK = '#172B4D'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / 'docs/assets/performance'
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    values = {}

    def read(relative):
        p = root / relative
        d = json.loads(p.read_text())
        assert d['forced_length_ok'] is True, relative
        assert d['environment']['dirty'] is False, relative
        assert 'hostname' not in d['environment'], 'Use sanitized public inputs'
        sources[relative] = {
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'source_revision': d['environment']['git_commit'],
        }
        return d

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK,
        'ytick.color': INK, 'axes.edgecolor': '#CBD5E1',
        'axes.titleweight': 'bold', 'axes.titlesize': 12,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.hashsalt': 'qwen3-public-performance-v1',
    })

    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=180, facecolor='white')
        fig.savefig(out / f'{name}.svg', metadata={'Date': None}, facecolor='white')
        plt.close(fig)

    ns = [1, 2, 4, 6, 8, 12]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    axes = axes.flatten()
    fig.subplots_adjust(left=.08, right=.97, top=.77, bottom=.16, wspace=.26, hspace=.48)
    fig.suptitle('Session concurrency: throughput and tail latency', x=.08, y=.965,
                 ha='left', fontsize=21, fontweight='bold')
    fig.text(.08, .915, 'Qwen3-4B BF16  |  Same GPU and runtime revision  |  Two actual KV allocations', fontsize=11)
    for label, color, marker in [('kv-14gib', BLUE, 'o'), ('kv-28gib', ORANGE, 's')]:
        rows = []
        configs = []
        for n in ns:
            relative = ('bench/results/session-capacity/qwen3-runtime_n6.json'
                        if label == 'kv-14gib' and n == 6
                        else f'bench/results/session-scaling/{label}_n{n}.json')
            d = read(relative)
            assert d['environment']['git_commit'].startswith('070cdc5')
            assert d['workload']['n_sessions'] == n
            for k, v in {'turns': 5, 'prompt_tokens': 9981, 'suffix_tokens': 1633,
                         'think_s': 1.0, 'warmup': 1, 'max_num_seqs': 8}.items():
                assert d['workload'][k] == v, (relative, k)
            assert d['engine_config']['max_num_batched_tokens'] == 2048
            assert d['accounting']['completed'] == 5*n
            assert np.isclose(d['accounting']['goodput_per_s'],
                              d['accounting']['slo_met']/d['metrics']['wall_s'])
            configs.append(d['engine_config']['kv_budget_bytes'])
            rows.append({'n': n, 'output_tok_s': d['metrics']['output_tok_s'], 'goodput_req_s': d['accounting']['goodput_per_s'],
                         'ttft_p99_s': d['ttft_s_p99'], 'tpot_p99_ms': d['tpot_s_p99']*1000,
                         'completed': d['accounting']['completed'], 'slo_met': d['accounting']['slo_met']})
        assert len(set(configs)) == 1
        legend = f'{configs[0]/2**30:.2f} GiB KV'
        for ax, key in zip(axes, ['output_tok_s', 'goodput_req_s', 'ttft_p99_s', 'tpot_p99_ms']):
            ax.plot(ns, [r[key] for r in rows], marker=marker, color=color,
                    label=legend, linewidth=2.3, markersize=6)
        values[label] = rows
    for ax, title, ylabel in zip(axes,
            ['Output throughput ↑', 'SLO-compliant goodput ↑', 'Time to first token ↓', 'Time per output token ↓'],
            ['Output tokens / s', 'Completed turns meeting SLO / s', 'Recorded p99 TTFT (s; log scale)', 'Recorded p99 TPOT (ms)']):
        ax.set_title(title, loc='left', pad=13)
        ax.set_xlabel('Concurrent sessions', labelpad=9)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ns)
        ax.set_xlim(.5, 12.5)
        ax.grid(axis='y', color='#E2E8F0', linewidth=.8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.65)
    axes[2].set_yscale('log')
    axes[2].set_ylim(.45, 60)
    axes[2].set_yticks([.5, 1, 4.5, 10, 40], ['0.5', '1', '4.5', '10', '40'])
    axes[3].set_ylim(0, 150)
    for ax, threshold, text in [(axes[2], 4.5, 'SLO: 4.5 s'), (axes[3], 100, 'SLO: 100 ms')]:
        ax.axhline(threshold, color='#64748B', linestyle='--', linewidth=1.2)
        ax.text(12.3, threshold*1.09, text, ha='right', color='#475569', fontsize=9)
    axes[2].annotate('41.05 s', (12, values['kv-14gib'][-1]['ttft_p99_s']),
                     xytext=(-10, 9), textcoords='offset points', ha='right', color=BLUE, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.075, .88),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.08, .073, '5 turns/session · 9,981 initial tokens + 1,633 tokens/turn · 1 s tool think time · max batch 8', fontsize=10)
    fig.text(.08, .035, 'Historical run, 5N requests/point; no repeated-run uncertainty. Lines join measured points only. p99 is estimated from small samples.', fontsize=9, color='#64748B')
    save(fig, 'session-scaling')

    cases = [('decode', 'Decode'), ('latency', 'Latency'), ('batch8', 'Batch 8'),
             ('throughput', 'Batch 16'), ('prefill', 'Prefill-shaped'), ('longctx', 'Long context')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.76, hspace=.65, wspace=.35)
    fig.suptitle('Output throughput across six fixed workloads', x=.08, y=.965,
                 ha='left', fontsize=21, fontweight='bold')
    fig.text(.08, .908, 'Qwen3-4B BF16  |  Historical serving baseline  |  Higher is better', fontsize=11)
    for ax, (case, label) in zip(axes.flat, cases):
        records = [read(f'bench/results/runtime-vs-vllm/{engine}_{case}.json')
                   for engine in ['qwen3-runtime', 'vllm']]
        for key in ['prompt_tokens', 'output_tokens', 'concurrency']:
            assert records[0]['workload'][key] == records[1]['workload'][key]
        w = records[0]['workload']
        for i, (d, color) in enumerate(zip(records, [BLUE, ORANGE])):
            m = d['metrics']['output_tok_s']
            assert len(d['metrics']['trials']) == 3
            ax.barh(i, m['median'], color=color, height=.52)
            ax.errorbar(m['median'], i,
                        xerr=[[m['median']-m['min']], [m['max']-m['median']]],
                        color=INK, capsize=3, linewidth=1)
            ax.text(m['median'], i-.33, f"{m['median']:,.2f}", ha='right', va='center', fontsize=10)
        values[case] = [d['metrics']['output_tok_s'] for d in records]
        ax.set_title(f"{label}  ·  {w['prompt_tokens']:,} in / {w['output_tokens']} out\nConcurrency {w['concurrency']}", loc='left', fontsize=11, pad=10)
        ax.set_yticks([0,1], ['Runtime', 'vLLM'])
        ax.invert_yaxis()
        ax.set_ylim(1.6, -.65)
        ax.set_xlim(0, max(d['metrics']['output_tok_s']['max'] for d in records)*1.12)
        ax.set_xlabel('Output tokens / s', fontsize=9)
        ax.grid(axis='x', color='#E2E8F0')
        ax.set_axisbelow(True)
    fig.text(.08, .845, 'Blue: qwen3-runtime    Orange: vLLM 0.27.1 (prefix caching off)', fontsize=11)
    fig.text(.08, .065, 'Median of 3 measured trials; whiskers show min–max, not confidence intervals. Wall time includes prefill and decode.', fontsize=9, color='#64748B')
    fig.text(.08, .03, 'Each panel starts at zero and has its own scale. Different workload shapes are not connected into a scaling curve.', fontsize=9, color='#64748B')
    save(fig, 'serving-baseline')
    (out / 'figure-data.json').write_text(json.dumps({'sources': sources, 'plotted_values': values}, indent=2)+'\n')
    print(f'Generated 2 figures (PNG + SVG), audited {len(sources)} clean source files')


if __name__ == '__main__':
    main()
