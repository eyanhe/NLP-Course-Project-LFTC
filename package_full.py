"""Full ZIP64 handoff: include assets, texts, all weights, candidates and support."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parent
MANIFEST='FULL-PACKAGE-MANIFEST.json'

def build(destination):
    destination=destination.resolve()
    if destination.is_relative_to(ROOT):raise ValueError('Archive must be outside the project')
    pending=destination.with_suffix(destination.suffix+'.tmp')
    files=[];excluded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT).as_posix()
        if path.is_symlink():raise ValueError('Unexpected symlink: '+relative)
        if relative==MANIFEST:continue
        if path.name.startswith('~$') or path.suffix in {'.tmp','.partial'}:
            excluded.append({'path':relative,'reason':'Temporary application lock or incomplete file'});continue
        files.append(path)
    total=sum(p.stat().st_size for p in files);print(f'Packing {len(files)} files, {total/1e9:.3f} GB including all weights and candidates.',flush=True)
    rows=[];done=0;last=time.monotonic()
    # Store large binary weights verbatim; deflate documents and code. ZIP64 supports >4 GB.
    with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path in files:
            before=path.stat();h=hashlib.sha256();size=0;relative=path.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo.from_file(path,arcname=ROOT.name+'/'+relative)
            info.compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {'.safetensors','.pt','.parquet'} else zipfile.ZIP_DEFLATED
            info._compresslevel=1
            with path.open('rb') as source,archive.open(info,'w',force_zip64=True) as target:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    h.update(block);size+=len(block);target.write(block)
                    if time.monotonic()-last>=20:
                        print(f'Packed {(done+size)/1e9:.2f}/{total/1e9:.2f} GB',flush=True);last=time.monotonic()
            after=path.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or size!=before.st_size:
                raise RuntimeError('File changed during packaging: '+relative)
            rows.append({'path':relative,'bytes':size,'sha256':h.hexdigest()});done+=size
        manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'package_kind':'Full project handoff with assets, prepared data, final and candidate weights, code, results, reports and supporting materials',
            'project_folder':ROOT.name,'file_count':len(rows),'total_uncompressed_bytes':done,
            'manifest_scope':'All archived project files except this manifest itself. Temporary Office locks and incomplete files excluded.',
            'excluded':excluded,'files':rows}
        payload=json.dumps(manifest,ensure_ascii=False,indent=2)
        archive.writestr(ROOT.name+'/'+MANIFEST,payload)
    (ROOT/MANIFEST).write_text(payload,encoding='utf-8')
    print('Checking every compressed entry (CRC)...',flush=True)
    with zipfile.ZipFile(pending) as archive:
        seen=set();checked=0;last=time.monotonic()
        expected={ROOT.name+'/'+r['path']:r for r in rows}
        for member in archive.infolist():
            if member.filename in seen:raise RuntimeError('Duplicate ZIP entry')
            seen.add(member.filename)
            if member.filename==ROOT.name+'/'+MANIFEST:continue
            row=expected[member.filename];h=hashlib.sha256();size=0
            with archive.open(member) as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block);size+=len(block)
            if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise RuntimeError('Archive content mismatch: '+member.filename)
            checked+=size
            if time.monotonic()-last>=20:print(f'Verified {checked/1e9:.2f}/{done/1e9:.2f} GB',flush=True);last=time.monotonic()
        if seen!=set(expected)|{ROOT.name+'/'+MANIFEST}:raise RuntimeError('Archive file inventory mismatch')
    pending.replace(destination)
    digest=hashlib.sha256()
    with destination.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    checksum=destination.with_suffix(destination.suffix+'.sha256')
    checksum.write_text(digest.hexdigest()+'  '+destination.name+'\n',encoding='ascii')
    destination.with_suffix('.manifest.json').write_text(payload,encoding='utf-8')
    print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,'files':len(rows)+1,'uncompressed_bytes':done,'sha256':digest.hexdigest(),'integrity':'Every archived file SHA256 and CRC verified'},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT.parent/'text-classification-full-handoff.zip')
    build(parser.parse_args().output)
