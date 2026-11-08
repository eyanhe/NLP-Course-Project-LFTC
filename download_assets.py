"""Download pinned public assets. No login or upload is performed."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parent
DATASETS = {
    'imdb': 'stanfordnlp/imdb',
    'rotten_tomatoes': 'cornell-movie-review-data/rotten_tomatoes',
    'yelp': 'fancyzhx/yelp_polarity',
    'ag_news': 'fancyzhx/ag_news',
    'mnli': 'nyu-mll/multi_nli',
}
MODEL = 'distilbert/distilbert-base-uncased'

def get_json(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def download(task):
    url, dest, expected = task
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
        if expected and checksum != expected['sha256']:
            raise RuntimeError(f'Cached asset differs from its recorded SHA256: {dest.name}')
        return {'path': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size,
                'sha256': checksum, 'url': url}
    partial = dest.with_suffix(dest.suffix + '.partial')
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                digest = hashlib.sha256()
                total = 0
                with partial.open('wb') as f:
                    for block in r.iter_content(1024 * 1024):
                        if block:
                            f.write(block); digest.update(block); total += len(block)
            partial.replace(dest)
            print(f'Downloaded {dest.name}: {total:,} bytes', flush=True)
            return {'path': str(dest.relative_to(ROOT)), 'bytes': total,
                    'sha256': digest.hexdigest(), 'url': url}
        except Exception:
            if attempt == 2: raise
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.root.resolve() != ROOT: raise ValueError('Use the project-local asset directory')
    lockfile = ROOT / 'assets.lock.json'
    lock = json.loads(lockfile.read_text('utf-8')) if lockfile.exists() else {'datasets': {}, 'model': {}}
    previous = {row['path']: row for row in lock.get('downloads', [])}
    tasks = []
    for name, repo in DATASETS.items():
        revision = lock['datasets'].get(name, {}).get('revision', 'main')
        info = get_json(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}')
        sha = info['sha']
        files = [x['rfilename'] for x in info['siblings']
                 if x['rfilename'].endswith('.parquet') and 'unsupervised' not in x['rfilename']]
        if not files: raise RuntimeError(f'No parquet data found for {repo}')
        lock['datasets'][name] = {'repo': repo, 'revision': sha, 'files': files}
        for filename in files:
            dest = ROOT / 'assets' / name / filename
            tasks.append((f'https://huggingface.co/datasets/{repo}/resolve/{sha}/{filename}?download=true',
                          dest, previous.get(str(dest.relative_to(ROOT)))))
    revision = lock.get('model', {}).get('revision', 'main')
    info = get_json(f'https://huggingface.co/api/models/{MODEL}/revision/{revision}')
    files = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors']
    available = {x['rfilename'] for x in info['siblings']}
    files = [f for f in files if f in available]
    if 'model.safetensors' not in files: raise RuntimeError('Safe tensor checkpoint is required')
    lock['model'] = {'repo': MODEL, 'revision': info['sha'], 'files': files}
    for filename in files:
        dest = ROOT / 'assets' / 'distilbert' / filename
        tasks.append((f'https://huggingface.co/{MODEL}/resolve/{info["sha"]}/{filename}?download=true',
                      dest, previous.get(str(dest.relative_to(ROOT)))))
    lockfile.write_text(json.dumps(lock, indent=2), 'utf-8')
    with ThreadPoolExecutor(max_workers=3) as pool:
        lock['downloads'] = list(pool.map(download, tasks))
    lockfile.write_text(json.dumps(lock, indent=2), 'utf-8')
    print('All assets downloaded and SHA256 recorded.', flush=True)

if __name__ == '__main__': main()
