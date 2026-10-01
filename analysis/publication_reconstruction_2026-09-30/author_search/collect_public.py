"""Archive bounded, unauthenticated public GitHub/author-page evidence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime,hashlib,json,urllib.error,urllib.request
ROOT=Path(__file__).resolve().parent
URLS={
'gatech_distributed_control':'https://github.gatech.edu/MCG-Lab/DistributedControl',
'sklar_profile':'https://api.github.com/users/matthewsklar',
'sklar_repos':'https://api.github.com/users/matthewsklar/repos?per_page=100&type=owner',
'patel_profile':'https://api.github.com/users/anirudh2',
'patel_repos':'https://api.github.com/users/anirudh2/repos?per_page=100&type=owner',
'sklar_user_search':'https://api.github.com/search/users?q=Matthew%20Sklar%20in:fullname&per_page=20',
'daniel_profile':'https://api.github.com/users/Daniel-Martin576',
'daniel_repos':'https://api.github.com/users/Daniel-Martin576/repos?per_page=100&type=owner',
'fc_cmplx1_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Cmplx1.py',
'fc_cmplx2_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Cmplx2.py',
'fc_cmplx1_utilities_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Cmplx1_Utilities.py',
'fc_cmplx2_utilities_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Cmplx2_Utilities.py',
'fc_wildfire_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/WildFire_Model.py',
'fc_base_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Base.py',
'fc_utilities_oct2021':'https://raw.githubusercontent.com/EsiSeraj/FireCommander2020/b04e8ec702f5c550978a170d9823c2a39de617be/MARL%20Package/FireCommander_Base_Utilities.py',
'magic_pp_source':'https://raw.githubusercontent.com/CORE-Robotics-Lab/MAGIC/main/envs/ic3net-envs/ic3net_envs/predator_prey_env.py',
'fc_tree_oct2021':'https://api.github.com/repos/EsiSeraj/FireCommander2020/git/trees/b04e8ec702f5c550978a170d9823c2a39de617be?recursive=1',
'fc_tree_nov2020':'https://api.github.com/repos/EsiSeraj/FireCommander2020/git/trees/176a6b151d935769363ef5b0606ad9589eedd614?recursive=1',
'magic_metadata':'https://api.github.com/repos/CORE-Robotics-Lab/MAGIC',
'magic_tree':'https://api.github.com/repos/CORE-Robotics-Lab/MAGIC/git/trees/main?recursive=1',
'esiseraj_profile':'https://api.github.com/users/EsiSeraj',
'esiseraj_repos':'https://api.github.com/users/EsiSeraj/repos?per_page=100&type=owner',
'rohan_profile':'https://api.github.com/users/rohanpaleja27',
'rohan_repos':'https://api.github.com/users/rohanpaleja27/repos?per_page=100&type=owner',
'wang_profile':'https://api.github.com/users/phejohnwang',
'wang_repos':'https://api.github.com/users/phejohnwang/repos?per_page=100&type=owner',
'core_repos':'https://api.github.com/orgs/CORE-Robotics-Lab/repos?per_page=100&type=all',
'firecommander2020':'https://api.github.com/repos/EsiSeraj/FireCommander2020',
'firecommander2020_commits':'https://api.github.com/repos/EsiSeraj/FireCommander2020/commits?per_page=100',
'wang_page':'https://phejohnwang.github.io/',
'daniel_page':'https://dmartinnn.com/',
'rohan_page':'https://www.rohanpaleja.com/',
}
def fetch(item):
 name,url=item
 req=urllib.request.Request(url,headers={'User-Agent':'HetNet-reproducibility-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r: data=r.read();status=r.status;headers=dict(r.headers);final=r.url
 except urllib.error.HTTPError as e: data=e.read();status=e.code;headers=dict(e.headers);final=e.url
 suffix='.json' if 'api.github.com' in url else ('.py' if url.endswith('.py') else '.html')
 p=ROOT/'public'/f'{name}{suffix}';p.parent.mkdir(exist_ok=True);p.write_bytes(data)
 return {'name':name,'url':url,'final_url':final,'status':status,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pagination_link':headers.get('Link'),'rate_remaining':headers.get('X-RateLimit-Remaining')}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as pool:records=list(pool.map(fetch,URLS.items()))
 (ROOT/'fetch_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 for r in records: print(r['name'],r['status'],r['pagination_link'])
