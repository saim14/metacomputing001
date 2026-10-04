#!/usr/bin/env python3
"""Audit recovered T1-U7C results without training, tuning, or fitting probes.

Run from the project root. Original outputs are read only. Checkpoint inference
recomputes the original development metrics; no reserved seed is instantiated.
"""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_t1_u7 import (
    causal_metrics, evaluate_seed_gate, generate_clean_evaluation,
    load_config, trajectory_value_statistics,
)
from run_t1_u7c import load_reference_results, paired_development_effects, validate_protocol
from src.t1_experiment import extract_trajectory
from src.tiny_transformer import ModelConfig, TinyTransformer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-8,
    )


def run_audit() -> dict:
    root = Path('outputs/t1_u7c')
    original_files = sorted(
        p for p in root.rglob('*') if p.is_file()
        and p.name not in {'recovery_audit.json', 'recovery_audit.log'}
        and not p.name.startswith('T1_U7C_')
    )
    hashes_before = {str(p): sha256(p) for p in original_files}
    study = load_config(root / 'preregistered_protocol.yaml')
    assert study == load_config(Path('configs/t1_u7c_training_signal.yaml'))
    assert study['experiment']['version'] == '0.11.0'
    assert study['development_seeds'] == [503, 607, 709, 811, 907, 1009]
    assert study['fresh_confirmation_seeds'] == [1103, 1201, 1301]
    reference = load_config(Path(study['closed_reference']['config']))
    verification = validate_protocol(study, reference)
    assert verification == json.loads((root / 'closed_reference_verification.json').read_text())
    assert verification['intervention_auxiliary_loss_weight'] == 0.005

    tests = []
    for test_path in sorted(Path('tests').glob('test_*.py')):
        namespace = runpy.run_path(str(test_path))
        for name, function in sorted(namespace.items()):
            if name.startswith('test_') and callable(function):
                function()
                tests.append(f'{test_path.name}::{name}')
    print(f'Existing direct tests passed: {len(tests)}', flush=True)

    history = pd.read_csv(root / 'development_training_history.csv')
    base_root = Path(study['closed_reference']['output'])
    base_history = pd.concat([
        pd.read_csv(base_root / 'development_training_history.csv'),
        pd.read_csv(base_root / 'fresh_training_history.csv'),
    ], ignore_index=True)
    gate = pd.read_csv(root / 'development_seed_gate.csv')
    gate['valid_value_steps'] = gate['valid_value_steps'].map(ast.literal_eval)
    causal = pd.read_csv(root / 'development_causal_metrics.csv')
    values = pd.read_csv(root / 'development_trajectory_value_statistics.csv')
    effects = pd.read_csv(root / 'paired_development_effects.csv')
    summary = json.loads((root / 'summary.json').read_text())
    decision = json.loads((root / 'advancement_decision.json').read_text())
    assert sorted(history.seed.unique().tolist()) == study['development_seeds']
    assert gate.seed.tolist() == study['development_seeds']
    assert len(causal) == 6 and len(values) == 30

    seed_audits = []
    for seed in study['development_seeds']:
        h = history[history.seed == seed].copy()
        b = base_history[base_history.seed == seed].copy()
        assert h.step.is_unique and h.step.is_monotonic_increasing
        expected_updates = []
        update_offset = 0
        for phase in study['training']['curriculum']:
            n = int(phase['steps'])
            logged_steps = sorted({1, n, *range(study['training']['log_every'], n + 1,
                                               study['training']['log_every'])})
            expected_updates.extend(update_offset + k for k in logged_steps)
            phase_history = h[h.phase == phase['name']]
            assert phase_history.phase_step.tolist() == logged_steps
            assert (phase_history.auxiliary_loss_weight == phase['auxiliary_loss_weight']).all()
            assert (phase_history.learning_rate == phase['learning_rate']).all()
            update_offset += n
        assert h.step.tolist() == expected_updates and update_offset == 6000
        assert np.isfinite(h.select_dtypes('number')).all().all()
        prefix_columns = [c for c in h if c not in ['evaluation_phase', 'elapsed_seconds']]
        pd.testing.assert_frame_equal(
            h.loc[h.step <= 4000, prefix_columns].reset_index(drop=True),
            b.loc[b.step <= 4000, prefix_columns].reset_index(drop=True),
            check_exact=True,
        )

        seed_dir = root / 'development' / f'seed_{seed}'
        checkpoint = seed_dir / 'checkpoints' / f't1_transformer_seed_{seed}.npz'
        trajectory_path = seed_dir / 'trajectory' / f't1_u7_development_trajectory_seed_{seed}.npz'
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: archive[key] for key in archive.files}
        expected_data = generate_clean_evaluation(study, seed, study['data']['development_sample_seed_offset'])
        for key in ['tokens', 'labels']:
            np.testing.assert_array_equal(trajectory[key], expected_data[key])
        assert trajectory['tokens'].shape == (4000, 12)
        assert trajectory['probabilities'].shape == (4000, 7, 2)
        for array in trajectory.values():
            assert np.isfinite(array).all()

        model = TinyTransformer(ModelConfig(
            vocab_size=study['data']['key_count'] + 3,
            sequence_length=2 + 2 * study['data']['pair_count'], **study['model'],
        ), seed=seed)
        with np.load(checkpoint, allow_pickle=False) as archive:
            parameters = {parameter.name: parameter for parameter in model.parameters()}
            assert set(parameters) == set(archive.files)
            for name, parameter in parameters.items():
                assert parameter.data.shape == archive[name].shape
                parameter.data[...] = archive[name]
        regenerated = extract_trajectory(
            model, trajectory['tokens'], trajectory['labels'],
            batch_size=study['training']['validation_batch_size'],
        )
        max_difference = 0.0
        for name in trajectory:
            max_difference = max(max_difference, float(np.max(np.abs(regenerated[name] - trajectory[name]))))
            np.testing.assert_allclose(regenerated[name], trajectory[name], rtol=1e-5, atol=2e-6)
        value_rows = trajectory_value_statistics(
            trajectory, seed=seed, phase='development',
            positive_epsilon=study['development_gate']['value_positive_epsilon'],
        )
        causal_row = causal_metrics(
            model, study=study, seed=seed, phase='development',
            causal_offset=study['data']['development_causal_seed_offset'],
        )
        gate_row = evaluate_seed_gate(value_rows, causal_row, study['development_gate'])
        assert_frame_equal(pd.DataFrame(value_rows), values[values.seed == seed])
        assert_frame_equal(pd.DataFrame([causal_row]), causal[causal.seed == seed])
        assert_frame_equal(pd.DataFrame([gate_row]), gate[gate.seed == seed])
        seed_audits.append({
            'seed': seed, 'logged_rows': len(h), 'final_update': int(h.step.max()),
            'prefix_logged_rows_exactly_equal_to_u7b': int((h.step <= 4000).sum()),
            'checkpoint_sha256': sha256(checkpoint),
            'trajectory_sha256': sha256(trajectory_path),
            'checkpoint_trajectory_max_absolute_difference': max_difference,
            'original_clean_examples_reproduced': True,
            'original_causal_metrics_reproduced': True,
            'gate_recomputed': gate_row,
        })
        print(f'Verified seed {seed}: update 6000, clean={gate_row["final_accuracy"]:.5f}, '
              f'paired_cf={gate_row["counterfactual_pair_accuracy"]:.4f}, '
              f'gate={gate_row["passed"]}', flush=True)

    baseline_gate, baseline_causal = load_reference_results(study)
    assert_frame_equal(paired_development_effects(baseline_gate, gate, baseline_causal, causal), effects)
    assert_frame_equal(pd.DataFrame(summary['development']['seed_gate']), gate)
    assert_frame_equal(pd.DataFrame(summary['development']['causal_metrics']), causal)
    assert_frame_equal(pd.DataFrame(summary['development']['trajectory_value_statistics']), values)
    assert_frame_equal(pd.DataFrame(summary['paired_development_effects_vs_t1_u7b']), effects)
    expected_decision = {
        'development_passed': False, 'development_passing_fraction': 0.5,
        'fresh_confirmation_opened': False, 'fresh_confirmation_passed': False,
        'trajectory_probes_fitted': False, 'advance_to_t1_u8': False,
    }
    assert gate.passed.tolist() == [True, True, True, False, False, False]
    assert decision == expected_decision == summary['advancement_decision']
    assert summary['status'] == 'failed_development_gate_new_confirmation_seeds_unopened'
    assert summary['fresh_confirmation'] == {
        'opened': False, 'passed': False, 'seed_gate': [], 'causal_metrics': [],
        'trajectory_value_statistics': [],
    }
    assert not (root / 'fresh_confirmation').exists()
    assert not list(root.glob('fresh_*.csv'))
    assert not any(root.rglob('seed_1103'))
    assert not any(root.rglob('seed_1201'))
    assert not any(root.rglob('seed_1301'))
    assert {str(p): sha256(p) for p in original_files} == hashes_before

    result = {
        'audit_date_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'recovered_complete_outputs_verified',
        'training_performed_during_recovery': False,
        'probes_fitted_during_recovery': False,
        'confirmation_seeds_instantiated_during_recovery': [],
        'direct_tests_passed': tests, 'direct_test_count': len(tests),
        'original_output_files_unchanged': True,
        'original_output_sha256': hashes_before,
        'protocol_configuration_verification': verification,
        'seeds': seed_audits, 'decision': decision,
        'interpretation_cautions': [
            'Protocol validator preserves data fields, but moving 811/907/1009 from '
            'fresh confirmation to development changes the active evaluation offsets '
            'from 83/89 to 37/43. Their historical differences are paired by model '
            'seed, not by evaluation example. Original measurements are preserved.',
            'Model seed also controls training data and minibatch randomness; '
            'between-seed differences do not isolate weight initialization.',
            'Checkpoints contain final model weights, not optimizer or RNG states. '
            'No mid-training resume was needed or attempted.',
            'Corruption controls test the specified edge dependencies; chance '
            'performance does not exclude every possible shortcut.',
        ],
    }
    (root / 'recovery_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('AUDIT PASSED: six original endpoints verified; no training or probes.', flush=True)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        run_audit()
