"""Compute a nominal duration diagnostic, not measured failure performance.

For an event independent of the nominal trajectory with U uniform on 10,...,30,
Pr(U<T | T) = max(0,min(21,T-10))/21. Here T counts executed actions and the
event occurs before action U, using zero-based event times. Only PCP has
implemented physical sensor-failure semantics; other task rows are mathematical
duration statistics. These changing-policy windows are not frozen evaluations.
"""
from pathlib import Path
import collections
import json

DIRECTORY = Path(__file__).parent
summaries = json.loads((DIRECTORY / 'findings.json').read_text())
specifications = {row['name']: row for row in summaries}
histograms = collections.defaultdict(collections.Counter)
for line in (DIRECTORY / 'episode_epoch_sufficient_stats.jsonl').open():
    row = json.loads(line)
    end = specifications[row['name']]['matched_end_epoch']
    if end - 49 <= row['epoch'] <= end:
        histograms[row['name']].update({int(k): v for k, v in row['steps_hist'].items()})
results = []
for name, histogram in histograms.items():
    n = sum(histogram.values())
    probability = sum(count * max(0, min(21, length - 10)) / 21
                      for length, count in histogram.items()) / n
    results.append(dict(name=name,
                        first_epoch=specifications[name]['matched_end_epoch'] - 49,
                        last_epoch=specifications[name]['matched_end_epoch'],
                        episodes=n,
                        nominal_uniform_10_30_event_exposure=probability))
(DIRECTORY / 'nominal_failure_exposure.json').write_text(json.dumps(results, indent=2) + '\n')
