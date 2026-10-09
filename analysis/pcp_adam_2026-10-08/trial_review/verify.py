#!/usr/bin/env python3
"""Reproduce input-immutability validation and independent root-window comparison.

Run after audit.py --episodes and the root compare.py. This rehashes inputs but
never reruns simulation, model execution, or the full JSON episode aggregation.
"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPORT=OUT.parent

def read(name):return json.loads((OUT/name).read_text())
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ast.parse((OUT/'audit.py').read_text())
    episodes=read('episode_audit.json')
    hashes={**read('input_hashes.json'),**{r['path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in episodes}}
    for relative,expected in hashes.items():
        path=ROOT/relative
        assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes'],relative
    review=read('review.json');checkpoints=read('checkpoints.json');old=read('old_terminal_checkpoints.json')
    assert all(r['all_validations_pass'] for r in episodes)
    for row in episodes:
        final=next(r['final'] for r in review['runs'] if r['group']=='adam_shared' and r['seed']==row['seed'])
        for key,value in row['final_epoch_recomputed'].items():
            assert math.isclose(value,final[key],abs_tol=1e-11,rel_tol=1e-10),(row['seed'],key)
    validation={
        'all_input_hashes_rechecked_unchanged':True,'input_files_rechecked':len(hashes),
        'new_checkpoint_count':len(checkpoints),'old_terminal_checkpoint_count':sum(r['available'] for r in old),
        'source_files_verified':sum(r['source_files_verified'] for r in review['runs'] if r['group']=='adam_shared'),
        'new_episode_records':sum(r['episodes'] for r in episodes),'new_steps':sum(r['steps'] for r in episodes),
        'new_updates_reconciled':sum(r['updates_reconciled'] for r in episodes),
        'new_epochs_reconciled':sum(r['epochs_reconciled'] for r in episodes),
        'script_syntax_valid':True,'script_sha256':sha(OUT/'audit.py'),
        'stdout_discrepancy_count':sum(len(r['stdout_discrepancies']) for r in review['runs'] if r['group']=='adam_shared'),
        'no_training_or_evaluation_performed':True,
    }
    save('validation.json',validation)
    remap={'shared_adam':'adam_shared','shared_rmsprop':'rmsprop_shared','banked_rmsprop':'rmsprop_banked'}
    with (OUT/'windows.csv').open() as stream:
        reference={(r['group'],int(r['seed']),r['window']):r for r in csv.DictReader(stream)}
    count=fields=0;exact=True
    with (REPORT/'comparison_windows.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['group'] not in remap:continue
            other=reference[(remap[row['group']],int(row['seed']),row['window'])]
            for key,other_key in [('epochs','epochs'),('episodes','episodes'),('first_epoch','first_epoch'),
                ('last_epoch','last_epoch'),('start_steps','actual_start_steps'),('end_steps','actual_end_steps'),
                ('steps','steps'),('steps_taken','steps_taken'),('success_rate','success_rate'),
                ('mean_agent_return','mean_agent_return')]:
                a,b=float(row[key]),float(other[other_key]);exact &= a==b
                assert math.isclose(a,b,abs_tol=1e-10),(row,key,a,b)
                fields+=1
            assert math.isclose(float(row['epoch_hours'])*3600,float(other['recorded_epoch_seconds']),abs_tol=1e-8)
            count+=1
    save('comparison_verification.json',{'per_seed_windows_checked':count,'metric_count_boundary_fields_checked':fields,
        'all_fields_exact':exact,'timing_fields_match_within_roundoff':True,
        'root_comparison_sha256':sha(REPORT/'comparison_windows.csv'),'reference_windows_sha256':sha(OUT/'windows.csv')})
    print(json.dumps(validation,sort_keys=True))

if __name__=='__main__':main()
