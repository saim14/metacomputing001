"""Package the complete immutable run into integrity-checked GitHub-sized parts."""
from pathlib import Path
import hashlib, json, zipfile

ROOT=Path(__file__).resolve().parent
def package():
    target=ROOT/'F1_A3_Main_Study.zip'
    if target.exists():raise RuntimeError('Existing archive preserved')
    assert json.loads((ROOT/'run/audit.json').read_text())['passed']
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and
        '__pycache__' not in p.parts and 'archive_parts' not in p.parts and
        p.name not in ['archive_manifest.json','F1_A3_Main_Study.zip','SHA256SUMS.txt']]
    checks={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'SHA256SUMS.txt').write_text(''.join(f'{h}  {name}\n' for name,h in checks.items()))
    files.append(ROOT/'SHA256SUMS.txt')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    directory=ROOT/'archive_parts'; directory.mkdir(exist_ok=False)
    parts=[]; whole=hashlib.sha256(); total=0
    with target.open('rb') as f:
        i=0
        while data:=f.read(4*1024*1024):
            p=directory/f'part-{i:04d}.bin';p.write_bytes(data);whole.update(data);total+=len(data)
            parts.append({'path':str(p.relative_to(ROOT)),'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});i+=1
    manifest={'study':'F1-A3 main-v1','archive_name':target.name,'size_bytes':total,'sha256':whole.hexdigest(),
        'parts':parts,'member_count':len(files),'source_commit':'b549785c9a1552b9c071137b860f1791d1a1eee8'}
    (ROOT/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bytes':total,'parts':len(parts),'members':len(files),'sha256':whole.hexdigest()}))

if __name__=='__main__':package()
