"""Check every archived file after extraction. Uses only Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

ROOT=Path(__file__).resolve().parent

def safe_path(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Unsafe manifest path: '+name)
    path=root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Path outside package')
    return path

def verify(root):
    manifest=json.loads((root/'FULL-PACKAGE-MANIFEST.json').read_text('utf-8'))
    failures=[];done=0;start=time.monotonic();last=start
    for row in manifest['files']:
        path=safe_path(root,row['path'])
        if not path.is_file():failures.append(row['path']+': missing');continue
        h=hashlib.sha256();size=0
        with path.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block);size+=len(block)
        if size!=row['bytes'] or h.hexdigest()!=row['sha256']:failures.append(row['path']+': size/SHA256 mismatch')
        done+=size
        if time.monotonic()-last>=20:
            print(f'Checked {done/1e9:.2f} GB',flush=True);last=time.monotonic()
    if failures:
        for failure in failures:print(failure)
        raise SystemExit(f'FAILED: {len(failures)} file(s) differ')
    print(f'PASS: {len(manifest["files"])} files, {done:,} bytes, SHA256 verified in {time.monotonic()-start:.1f}s')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    verify(parser.parse_args().root.resolve())
