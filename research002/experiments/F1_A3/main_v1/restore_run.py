"""Verify complete F1-A3 archive and restore missing evidence without overwrites."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, tempfile, zipfile

ROOT=Path(__file__).resolve().parent
def restore(verify_only=False):
    spec=json.loads((ROOT/'archive_manifest.json').read_text()); h=hashlib.sha256(); total=0
    with tempfile.TemporaryFile() as joined:
        for item in spec['parts']:
            data=(ROOT/item['path']).read_bytes()
            assert len(data)==item['size_bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            joined.write(data); h.update(data); total+=len(data)
        assert total==spec['size_bytes'] and h.hexdigest()==spec['sha256']; joined.seek(0)
        with zipfile.ZipFile(joined) as z:
            assert z.testzip() is None
            # Validate every destination before writing any bytes.
            pending=[]
            for item in z.infolist():
                p=PurePosixPath(item.filename)
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe path')
                if item.is_dir():continue
                dest=ROOT/str(p); data=z.read(item)
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(data).digest():
                        raise ValueError(f'Preserving differing existing file: {p}')
                else:pending.append(item)
            if not verify_only:
                for item in pending:
                    dest=ROOT/item.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
    print(json.dumps({'verified':True,'archive_sha256':spec['sha256'],'restored_files':0 if verify_only else len(pending)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args();restore(args.verify_only)
