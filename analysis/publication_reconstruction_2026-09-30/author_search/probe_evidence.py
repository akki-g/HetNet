"""Verify archived evidence and execute bounded unchanged predecessor methods."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
PUB=ROOT/'public'

def load_class(path,name):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    ns={'np':np}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]


def main():
    manifest=json.loads((ROOT/'fetch_manifest.json').read_text())
    for row in manifest:
        assert row['status']==200,row
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row
        assert not row['pagination_link'],row
    profiles={}
    for key in ['esiseraj','rohan','wang','daniel','sklar','patel','core']:
        repos=json.loads((PUB/f'{key}_repos.json').read_text())
        profiles[key]={'repository_count':len(repos),'names':[r['full_name'] for r in repos]}
    original=(PUB/'fc_wildfire_oct2021.py').read_bytes()
    git=ROOT.parents[1]/'upstream_history_2026-09-30/history.git'
    bundled=subprocess.check_output(['git',f'--git-dir={git}','show','59d598319d6285acfa42a1d9d8c859305a9bbf1e:WildFire_Simulate_Original.py'])
    assert original==bundled
    wildfire=load_class(PUB/'fc_wildfire_oct2021.py','WildFire')
    w=wildfire(terrain_sizes=[5,5],hotspot_areas=[[0,0,2,2]],num_ign_points=1,duration=300)
    try:
        w.hotspot_init()
    except ValueError as exc:
        degenerate={'exception':type(exc).__name__,'message':str(exc)}
    else:
        raise AssertionError('Expected historical equal-bound failure')
    w.hotspot_areas=[[1,3,1,3]]
    np.random.seed(17)
    hotspot=w.hotspot_init()
    assert hotspot.shape==(1,3)
    boundary=np.array([[0.,2.,1.]])
    front,geo=w.fire_propagation(5,ign_points_all=boundary,geo_phys_info={'spread_rate':np.ones((5,5)),'wind_speed':np.ones((5,1)),'wind_direction':np.ones((5,1))},previous_terrain_map=boundary.copy(),pruned_List=[])
    assert front.tolist()==[[0.,0.,0.]]
    util=load_class(PUB/'fc_utilities_oct2021.py','EnvUtilities')
    agent=np.zeros((5,5));agent[2,2]=1
    undiscovered=np.zeros((5,5));undiscovered[2,2]=1
    discovered=np.zeros((5,5));discovered[2,2]=2
    before=util.pruning(undiscovered,agent,np.array([]),np.array([12]),5)
    after=util.pruning(discovered,agent,np.array([12]),np.array([12]),5)
    assert before[0].tolist()==[] and before[2].tolist()==[12]
    assert after[0].tolist()==[12] and after[2].tolist()==[]
    trees={key:json.loads((PUB/f'{key}.json').read_text()) for key in ['fc_tree_oct2021','fc_tree_nov2020','magic_tree']}
    for d in trees.values():assert not d['truncated']
    result={
      'fetched_resources_verified':len(manifest),'repository_inventories':profiles,
      'firecommander_commit':'b04e8ec702f5c550978a170d9823c2a39de617be',
      'wildfire_exact_match':True,'wildfire_size_bytes':len(original),'wildfire_sha256':hashlib.sha256(original).hexdigest(),
      'degenerate_hotspot':degenerate,'nondegenerate_hotspot':hotspot.tolist(),
      'boundary_point':boundary.tolist(),'returned_front':front.tolist(),
      'base_undiscovered_suppression':{'pruned':before[0].tolist(),'remaining_fires':before[2].tolist()},
      'base_discovered_suppression':{'pruned':after[0].tolist(),'remaining_fires':after[2].tolist()},
      'historical_tree_blob_paths':{k:[i['path'] for i in v['tree'] if i['type']=='blob'] for k,v in trees.items()},
      'gatech_access':'Redirect to LDAP login; source unavailable without institutional access',
      'scope':'Exact archived predecessor methods only; no learning or complete historical FC episode'}
    (ROOT/'evidence_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['repository_inventories','historical_tree_blob_paths']},indent=2))

if __name__=='__main__':main()
