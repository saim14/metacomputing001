"""Package this completed study and validate every released archive member."""
from pathlib import Path
import hashlib
import json
import os
import zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'F1_A1_Study.zip'
files=sorted(p for p in ROOT.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.tmp','.zip'}
             and p.name!='SHA256SUMS.txt')
audit=json.loads((ROOT/'run/audit.json').read_text())
assert audit['passed']
manifest=json.loads((ROOT/'run/manifest.json').read_text())
for rel,expected in manifest['hashes'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
for p in files:
    if p.suffix=='.npz':
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None,p
checksums=''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in files)
(ROOT/'SHA256SUMS.txt').write_text(checksums)
files.append(ROOT/'SHA256SUMS.txt')
temp=OUT.with_suffix('.zip.tmp')
with temp.open('wb') as f:
    with zipfile.ZipFile(f,'w',allowZip64=True) as z:
        for p in files:
            method=zipfile.ZIP_STORED if p.suffix in {'.npz','.png'} else zipfile.ZIP_DEFLATED
            z.write(p,p.relative_to(ROOT).as_posix(),compress_type=method)
    f.flush();os.fsync(f.fileno())
with zipfile.ZipFile(temp) as z:
    assert z.testzip() is None
    for line in checksums.splitlines():
        expected,rel=line.split('  ',1)
        assert hashlib.sha256(z.read(rel)).hexdigest()==expected,rel
os.replace(temp,OUT)
print(json.dumps({'archive':str(OUT),'files':len(files),'size_bytes':OUT.stat().st_size,
                  'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
                  'all_npz_and_zip_members_verified':True},indent=2))
