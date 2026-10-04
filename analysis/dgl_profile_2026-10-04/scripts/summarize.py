"""Recompute every number reported for the 4 October DGL/Binary/communication-range questions.

Read-only with respect to all inputs. Run from the repository root:
    .venv/bin/python analysis/dgl_profile_2026-10-04/scripts/summarize.py
Writes analysis/dgl_profile_2026-10-04/summary.json, including SHA256 hashes of every input read.
"""
import hashlib
import json
import pathlib
import pstats
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
D = ROOT / 'analysis' / 'dgl_profile_2026-10-04'
INPUTS = {}


def data(rel):
    path = ROOT / rel
    raw = path.read_bytes()
    INPUTS[str(rel)] = hashlib.sha256(raw).hexdigest()
    return raw


def text(rel):
    return data(rel).decode('utf-8', errors='replace')


def jsonl(rel):
    return [json.loads(line) for line in text(rel).splitlines() if line.strip()]


def rate(rows):
    return sum(r['steps'] for r in rows) / sum(r['wall_time_seconds'] for r in rows)


# A. In-situ single-collector PCP runs on the Mac (4 complete updates each, seed 991).
in_situ = {}
for tag in ['base_real', 'base_binary', 'ins_real', 'ins_binary', 'det_real', 'cprofile_real', 'cprofile_binary']:
    rows = jsonl(f'analysis/dgl_profile_2026-10-04/raw/{tag}/updates.jsonl')
    steps = sum(r['steps'] for r in rows)
    seconds = sum(r['wall_time_seconds'] for r in rows)
    entry = {'updates': len(rows), 'steps': steps, 'episodes': sum(r['episodes'] for r in rows),
             'update_seconds': seconds, 'steps_per_second': steps / seconds, 'ms_per_step': 1e3 * seconds / steps}
    path = D / 'raw' / tag / 'instrument.json'
    if path.exists():
        timing = json.loads(text(path.relative_to(ROOT)))
        entry['timers'] = {k: {'seconds': v, 'percent_of_update_time': 100 * v / seconds,
                               'ms_per_step': 1e3 * v / steps, 'calls_per_step': timing['calls'][k] / steps}
                           for k, v in sorted(timing['seconds'].items(), key=lambda kv: -kv[1])}
        fwd = sum(v for k, v in timing['seconds'].items() if k.startswith('dgl_fwd') or k == 'dgl_forward_ops')
        dgl_total = fwd + timing['seconds']['dgl_backward_ops'] + timing['seconds']['graph_build']
        entry['dgl_forward_seconds'] = fwd
        entry['dgl_total_seconds'] = dgl_total
        entry['dgl_total_percent'] = 100 * dgl_total / seconds
        entry['dgl_total_ms_per_step'] = 1e3 * dgl_total / steps
    in_situ[tag] = entry
in_situ['instrumentation_overhead_percent'] = {
    v: 100 * (1 - in_situ[f'ins_{v}']['steps_per_second'] / in_situ[f'base_{v}']['steps_per_second'])
    for v in ['real', 'binary']}
in_situ['binary_slower_than_real_percent_uninstrumented'] = 100 * (
    in_situ['base_real']['steps_per_second'] / in_situ['base_binary']['steps_per_second'] - 1)


def semantic(tag, name):
    """Records without clock fields, for checking that timers/profilers did not change the computation."""
    return [{k: v for k, v in row.items() if 'time' not in k and 'seconds' not in k}
            for row in jsonl(f'analysis/dgl_profile_2026-10-04/raw/{tag}/{name}')]


in_situ['identical_to_uninstrumented_run'] = {
    tag: {name: semantic(tag, name) == semantic(f'base_{variant}', name)
          for name in ['updates.jsonl', 'epoch_signatures.jsonl']}
    for variant, tags in [('real', ['ins_real', 'det_real', 'cprofile_real']),
                          ('binary', ['ins_binary', 'cprofile_binary'])] for tag in tags}

# B. cProfile attribution (profiler overhead inflates Python-heavy code; used only for shares of named functions).
cprofile = {}
for v in ['real', 'binary']:
    rel = f'analysis/dgl_profile_2026-10-04/raw/cprofile_{v}/{v}.prof'
    data(rel)
    stats = pstats.Stats(str(ROOT / rel)).stats
    total = sum(s[2] for s in stats.values())
    def cumulative(name, file_part):
        return sum(s[3] for (f, _, n), s in stats.items() if n == name and file_part in f)
    cprofile[v] = {'total_tottime_seconds': total,
                   'gumbel_softmax_cumulative_percent': 100 * cumulative('gumbel_softmax', 'functional.py') / total,
                   'env_wrapper_step_cumulative_percent': 100 * cumulative('step', 'env_wrappers.py') / total,
                   'build_hetgraph_cumulative_percent': 100 * cumulative('build_hetgraph', 'hetgat/utils.py') / total}

# C. Dense-versus-DGL microbenchmark of the replaced message-passing portion.
micro_text = text('analysis/dgl_profile_2026-10-04/raw/dense_vs_dgl.txt')
micro = {'dgl': [], 'dense': []}
for kind, build, fwd, bwd, tot in re.findall(
        r'^(dgl|dense)\s+per step: graph build ([\d.]+) ms, message passing forward ([\d.]+) ms, '
        r'backward ([\d.]+) ms, total ([\d.]+) ms', micro_text, re.M):
    micro[kind].append({'build_ms': float(build), 'forward_ms': float(fwd), 'backward_ms': float(bwd), 'total_ms': float(tot)})
mean = {k: sum(r['total_ms'] for r in v) / len(v) for k, v in micro.items()}
micro['mean_total_ms'] = mean
micro['dgl_over_dense_ratio'] = mean['dgl'] / mean['dense']

# D. Amdahl-style estimate (NOT an end-to-end measurement).
ins = in_situ['ins_real']
fraction = ins['dgl_total_seconds'] / ins['update_seconds']
estimated_ms = ins['ms_per_step'] - ins['dgl_total_ms_per_step'] + mean['dense']
estimate = {'basis': 'ins_real per-step time with measured in-situ DGL time replaced by mean dense microbenchmark time',
            'dgl_fraction': fraction, 'current_ms_per_step': ins['ms_per_step'],
            'estimated_dense_ms_per_step': estimated_ms,
            'estimated_speedup': ins['ms_per_step'] / estimated_ms,
            'ceiling_speedup_if_dgl_cost_zero': 1 / (1 - fraction)}
det = in_situ['det_real']['timers']
estimate['relation_slicing_percent_det_real'] = det['dgl_fwd:graph.__getitem__']['percent_of_update_time']
estimate['graph_build_percent_det_real'] = det['graph_build']['percent_of_update_time']
estimate['r5_plus_r7_max_speedup'] = 1 / (1 - (estimate['relation_slicing_percent_det_real']
                                               + estimate['graph_build_percent_det_real']) / 100)

# E. Archived preflights: Binary versus Real on Mac and Stokes.
platforms = {'stokes_903502': 'stokes_runs/hetnet_preflight_903502',
             'mac_20261003': 'runs/hetnet_preflight_mac_20261003_234455_727228'}
preflight = {}
for label, root in platforms.items():
    out = {}
    for w in ['pp_real', 'pcp_real', 'fc_real', 'pcp_binary']:
        rows = jsonl(f'{root}/{w}/seed991/updates.jsonl')
        status = json.loads(text(f'{root}/{w}/seed991/run_status.json'))
        collectors = status['resources']['collectors']
        cpu = [c['user_cpu_seconds'] + c['system_cpu_seconds'] for c in collectors]
        wall = status['segment_wall_time_seconds']
        steps = sum(r['steps'] for r in rows)
        out[w] = {'steps': steps, 'steps_per_second': rate(rows),
                  'steps_per_second_updates_1_10': rate(rows[:10]),
                  'steps_per_second_updates_46_55': rate(rows[45:55]),
                  'steps_per_second_updates_91_100': rate(rows[-10:]),
                  'segment_wall_seconds': wall, 'collector_cpu_seconds': cpu,
                  'collector_cpu_utilisation': [c / wall for c in cpu],
                  'collector_cpu_seconds_per_step': sum(cpu) / steps,
                  'cpu_affinity': [c['cpu_affinity'] for c in collectors]}
    out['binary_over_real_cpu_seconds_per_step'] = (out['pcp_binary']['collector_cpu_seconds_per_step']
                                                    / out['pcp_real']['collector_cpu_seconds_per_step'])
    out['real_over_binary_steps_per_second'] = out['pcp_real']['steps_per_second'] / out['pcp_binary']['steps_per_second']
    host_hits = []
    for path in sorted((ROOT / root).rglob('*')):
        if path.is_file() and 'source' not in path.relative_to(ROOT / root).parts and path.suffix in {'.json', '.jsonl', '.log', '.txt'}:
            if re.search(r'hostname|nodelist|SLURMD_NODENAME|nodename', path.read_text(errors='replace'), re.I):
                host_hits.append(str(path.relative_to(ROOT)))
    out['files_mentioning_host_or_node'] = host_hits
    preflight[label] = out

# F. Communication range actually configured in current experiments.
comm = {}
values = []
for path in sorted((ROOT / 'stokes_runs/runs/softrole_primary').glob('*/seed*/config.json')):
    values.append(json.loads(text(path.relative_to(ROOT)))['comm_range'])
comm['softrole_primary_config_comm_range'] = {'files': len(values), 'distinct_values': sorted(set(values))}
recon = []
for root in platforms.values():
    for w in ['pp_real', 'pcp_real', 'fc_real', 'pcp_binary']:
        args = json.loads(text(f'{root}/{w}/seed991/resolved_args.json'))
        recon.append((args['comm_range_P'], args['comm_range_A'], args['lossy_comm']))
comm['reconstruction_preflight_comm_range_P_A_lossy'] = {'runs': len(recon), 'distinct': sorted(set(recon))}
frozen = []
for path in sorted((ROOT / 'stokes_runs/frozen_pcp_30m/frozen_pcp_30m_20261002_slurm/results').glob('*.json')):
    frozen += re.findall(r'"comm_range":\s*(-?[\d.]+)', text(path.relative_to(ROOT)))
comm['frozen_pcp_30m_result_reports_comm_range'] = {'occurrences': len(frozen), 'distinct': sorted(set(frozen))}
legacy = []
for path in sorted((ROOT / 'logs_1').rglob('*')):
    if path.is_file():
        legacy += re.findall(r'comm_range_([PA])=(-?\d+)', text(path.relative_to(ROOT)))
comm['legacy_logs_1_namespace_comm_range'] = {'occurrences': len(legacy), 'distinct': sorted(set(legacy))}

# G. Paper and public upstream commands.
paper = text('research/papers/HetNet.txt').split('\n')  # newline-only, matching grep -n
phrases = ['Limited-range communications', 'within communication range', 'Ablation Study #2',
           'Full, Half and No', 'half-communication (i.e., limited range)', '(a) Communication range.',
           'Reported results are Mean']
comm['paper_line_numbers_research_papers_HetNet_txt'] = {
    p: [i + 1 for i, line in enumerate(paper) if p in line] for p in phrases}
supplement = text('research/papers/HetNet_Supplementary.txt')
comm['supplement_mentions'] = {k: len(re.findall(k, supplement, re.I))
                               for k in ['communication range', 'comm_range', 'radius', 'half']}
commands = []
for path in sorted((ROOT / 'analysis/upstream_history_2026-09-30/snapshots').glob('*/README.md')):
    for line in text(path.relative_to(ROOT)).splitlines():
        if 'main.py' in line and 'python' in line:
            commands.append((path.parent.name, 'comm_range' in line))
comm['upstream_readme_training_commands'] = {
    'snapshots': sorted({c[0] for c in commands}), 'command_lines': len(commands),
    'command_lines_setting_comm_range': sum(c[1] for c in commands)}

summary = {'in_situ_mac_single_collector': in_situ, 'cprofile_mac': cprofile, 'dense_vs_dgl_microbenchmark': micro,
           'dense_speedup_estimate': estimate, 'archived_preflights': preflight, 'communication_range': comm,
           'inputs_sha256': INPUTS}
(D / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=False) + '\n')
print(json.dumps({'dgl_percent': ins['dgl_total_percent'], 'estimate': estimate, 'micro_ratio': micro['dgl_over_dense_ratio'],
                  'binary_slower_mac_percent': in_situ['binary_slower_than_real_percent_uninstrumented'],
                  'overhead': in_situ['instrumentation_overhead_percent'],
                  'stokes_cpu_ratio': preflight['stokes_903502']['binary_over_real_cpu_seconds_per_step'],
                  'mac_cpu_ratio': preflight['mac_20261003']['binary_over_real_cpu_seconds_per_step'],
                  'host_hits': {k: v['files_mentioning_host_or_node'] for k, v in preflight.items()},
                  'comm': {k: v for k, v in comm.items()}, 'cprofile': cprofile, 'inputs': len(INPUTS)}, indent=1))
