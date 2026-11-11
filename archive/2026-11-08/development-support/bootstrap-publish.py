"""Install portable, official publishing tools inside this workspace only."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import requests
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'.tools'
TOOLS.mkdir(exist_ok=True)
SELECTIONS={
    'git':('git-for-windows/git',lambda n:n.startswith('MinGit-') and n.endswith('-64-bit.zip') and 'busybox' not in n),
    'gh':('cli/cli',lambda n:n.endswith('windows_amd64.zip')),
    'git-xet':('huggingface/xet-core',lambda n:n=='git-xet-windows-x86_64.zip'),
    'git-lfs':('git-lfs/git-lfs',lambda n:'windows-amd64' in n and n.endswith('.zip')),
}

def install(item):
    name,(repo,predicate)=item
    release=requests.get('https://api.github.com/repos/'+repo+'/releases/latest',timeout=30)
    release.raise_for_status();body=release.json()
    assets=[a for a in body['assets'] if predicate(a['name'])]
    if not assets:
        releases=requests.get('https://api.github.com/repos/'+repo+'/releases',params={'per_page':20},timeout=30)
        releases.raise_for_status()
        body=next(r for r in releases.json() if any(predicate(a['name']) for a in r['assets']))
        assets=[a for a in body['assets'] if predicate(a['name'])]
    if len(assets)!=1:raise ValueError((name,[a['name'] for a in body['assets']]))
    asset=assets[0];archive=TOOLS/asset['name'];destination=TOOLS/name
    if not archive.exists() or archive.stat().st_size!=asset['size']:
        response=requests.get(asset['browser_download_url'],timeout=(15,90),stream=True)
        response.raise_for_status()
        with archive.with_suffix('.partial').open('wb') as output:
            for chunk in response.iter_content(2**20):output.write(chunk)
        archive.with_suffix('.partial').replace(archive)
    assert archive.stat().st_size==asset['size']
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target=(destination/entry.filename).resolve()
            assert target.is_relative_to(destination.resolve()),entry.filename
        bundle.extractall(destination)
    return {'name':name,'tag':body['tag_name'],'asset':asset['name'],
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'executables':[str(p.relative_to(ROOT)) for p in destination.rglob('*.exe') if p.stem in ['git','gh','git-xet','git-lfs']]}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(install,SELECTIONS.items()))
(TOOLS/'install-receipt.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results))
