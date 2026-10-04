#!/usr/bin/env python3
"""Package the closed T1-U7C record; never launches an experiment."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    root = Path(__file__).resolve().parent
    output = root / 'outputs/t1_u7c'
    audit = json.loads((output / 'recovery_audit.json').read_text())
    assert audit['status'] == 'recovered_complete_outputs_verified'
    assert audit['direct_test_count'] == 16
    assert not audit['decision']['advance_to_t1_u8']
    for relative, expected in audit['original_output_sha256'].items():
        assert digest(root / relative) == expected, relative

    archive_path = output / 'T1_U7C_PORTABLE_BUNDLE.zip'
    manifest_path = output / 'T1_U7C_INNER_SHA256SUMS.txt'
    external_hash_path = output / 'T1_U7C_BUNDLE_SHA256.txt'
    verification_path = output / 'T1_U7C_BUNDLE_VERIFICATION.json'
    excluded = {archive_path, manifest_path, external_hash_path, verification_path}
    files = []
    for name in [
        'run_t1_u7.py', 'run_t1_u7b.py', 'run_t1_u7c.py',
        'audit_t1_u7c.py', 'build_t1_u7c_bundle.py', 'requirements.txt',
        'T1_U7C_REPRODUCE.md', 'T1_U7C_Audit_Colab.ipynb',
    ]:
        files.append(root / name)
    for directory in ['src', 'tests']:
        files.extend(sorted((root / directory).glob('*.py')))
    for name in [
        't1_u7_two_hop_gate.yaml', 't1_u7_smoke.yaml',
        't1_u7b_fresh_stream_extension.yaml', 't1_u7b_smoke.yaml',
        't1_u7c_training_signal.yaml', 't1_u7c_smoke.yaml',
    ]:
        files.append(root / 'configs' / name)
    files.extend(sorted((root / 'notes').glob('Research_002_Direction_Freeze_*.md')))
    files.extend(sorted(p for p in output.rglob('*') if p.is_file() and p not in excluded))
    for experiment in ['t1_u7', 't1_u7b']:
        reference = root / 'outputs' / experiment
        files.extend(sorted(p for p in reference.iterdir()
                            if p.is_file() and p.suffix in {'.csv', '.json', '.yaml', '.md'}))
    files = sorted(set(files))
    assert all(p.is_file() for p in files)
    manifest = {p.relative_to(root).as_posix(): digest(p) for p in files}
    manifest_path.write_text(''.join(f'{sha}  {path}\n' for path, sha in manifest.items()))
    prefix = root.name + '/'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as bundle:
        for path in [*files, manifest_path]:
            bundle.write(path, prefix + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive_path) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == len(manifest) + 1
        for relative, expected in manifest.items():
            assert hashlib.sha256(bundle.read(prefix + relative)).hexdigest() == expected, relative
        assert bundle.read(prefix + manifest_path.relative_to(root).as_posix()) == manifest_path.read_bytes()
    bundle_digest = digest(archive_path)
    external_hash_path.write_text(f'{bundle_digest}  {archive_path.name}\n')
    verification = {
        'archive': archive_path.name,
        'archive_sha256': bundle_digest,
        'archive_size_bytes': archive_path.stat().st_size,
        'payload_files_verified': len(manifest),
        'archive_entries': len(manifest) + 1,
        'zip_crc_integrity_passed': True,
        'all_internal_sha256_matched': True,
        'original_results_unchanged': True,
        'training_performed': False,
        'trajectory_probes_fitted': False,
        'fresh_confirmation_opened': False,
    }
    verification_path.write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    build()
