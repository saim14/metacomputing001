"""Restore the byte-identical study archives and their original directory structure."""
import argparse, hashlib, json, shutil, tempfile, zipfile
from pathlib import Path, PurePosixPath

REPO=Path(__file__).resolve().parents[1]
def restore(studies=None,verify_only=False):
    manifest=json.loads((REPO/'research002/archive_manifest.json').read_text())
    known={a['study'] for a in manifest['archives']}
    if studies and not set(studies)<=known:raise ValueError(f'Unknown studies: {set(studies)-known}')
    for a in manifest['archives']:
        if studies and a['study'] not in studies:continue
        h=hashlib.sha256();total=0
        with tempfile.TemporaryFile() as assembled:
            for part in a['parts']:
                b=(REPO/part['path']).read_bytes()
                if len(b)!=part['size_bytes'] or hashlib.sha256(b).hexdigest()!=part['sha256']:
                    raise ValueError(f'Part integrity failure: {part["path"]}')
                assembled.write(b);h.update(b);total+=len(b)
            if total!=a['size_bytes'] or h.hexdigest()!=a['sha256']:raise ValueError('Archive integrity failure')
            assembled.seek(0)
            with zipfile.ZipFile(assembled) as z:
                bad=z.testzip()
                if bad:raise ValueError(f'ZIP integrity failure: {bad}')
                if not verify_only:
                    base=REPO/a['extract_to'];base.mkdir(parents=True,exist_ok=True)
                    for item in z.infolist():
                        if item.is_dir():continue
                        p=PurePosixPath(item.filename)
                        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
                        if a['strip_prefix']:
                            if p.parts[0]!=a['strip_prefix']:raise ValueError('Unexpected prefix')
                            p=PurePosixPath(*p.parts[1:])
                        dest=base/str(p);dest.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(item) as source,dest.open('wb') as output:shutil.copyfileobj(source,output)
        print(a['study'], 'verified' if verify_only else 'restored')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',nargs='+');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();restore(args.study,args.verify_only)
