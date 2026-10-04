"""Replay every panel decision from saved dataset scores and bootstrap draws."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(folder):
    datasets = pd.read_csv(folder/'dataset_metrics.csv')
    panels = pd.read_csv(folder/'panel_metrics.csv')
    summary = json.loads((folder/'detection_summary.json').read_text())
    result = json.loads((folder/'calibration_result.json').read_text())
    choices = json.loads((folder/'validation_choices.json').read_text())
    generation = json.loads((folder/'generation_checks.json').read_text())
    assert len(datasets) == 576 and len(panels) == 96 and len(choices) == 432 and len(generation) == 144
    assert not datasets.duplicated(['replicate', 'slot', 'planted_delta']).any()
    assert not panels.duplicated(['replicate', 'planted_delta']).any()
    assert set(datasets.replicate) == set(range(24)) and set(datasets.slot) == set(range(6))
    for name in ['linear', 'rff']:
        np.testing.assert_allclose(datasets[f'delta_vs_{name}'], datasets.history_r2-datasets[f'current_{name}_r2'], atol=1e-12)
    np.testing.assert_allclose(datasets.realized_oracle_delta, datasets.oracle_history_r2-datasets.oracle_current_r2, atol=1e-12)
    assert (datasets.loc[datasets.planted_delta == 0, 'realized_oracle_delta'] == 0).all()
    for _, row in panels.iterrows():
        cells = datasets[(datasets.replicate == row.replicate)&(datasets.planted_delta == row.planted_delta)]
        assert len(cells) == 6
        draws = np.load(folder/'bootstrap_panels'/f'replicate_{int(row.replicate):02d}.npz')['delta_r2']
        assert draws.shape == (1000, 4, 2) and np.isfinite(draws).all()
        di = [0, .01, .05, .1].index(row.planted_delta)
        passed = True
        for bi, name in enumerate(['linear', 'rff']):
            delta = cells[f'delta_vs_{name}']
            effect = float(delta.mean())
            mae = float(cells[f'mae_advantage_vs_{name}'].mean())
            lo, hi = np.percentile(draws[:, di, bi], [2.5, 97.5])
            for column, expected in [(f'mean_delta_vs_{name}', effect), (f'mae_advantage_vs_{name}', mae),
                                     (f'ci_low_vs_{name}', lo), (f'ci_high_vs_{name}', hi)]:
                np.testing.assert_allclose(row[column], expected, atol=1e-12)
            positive = int((delta > 0).sum())
            assert row[f'positive_datasets_vs_{name}'] == positive
            criteria = {'effect_threshold': effect >= .01, 'ci_above_zero': lo > 0,
                        'all_six_positive': positive == 6, 'mae_not_worse': mae >= 0}
            for criterion, expected in criteria.items():
                assert bool(row[f'{criterion}_vs_{name}']) == bool(expected)
            passed = passed and all(criteria.values())
        assert bool(row.screen_passed) == bool(passed)
    for item in summary:
        group = panels[panels.planted_delta == item['planted_population_delta_r2']]
        assert item['detected_panels'] == int(group.screen_passed.sum())
        assert item['panels'] == len(group) == 24
        assert item['detection_fraction'] == item['detected_panels']/24
        for name in ['linear', 'rff']:
            np.testing.assert_allclose(item[f'mean_observed_delta_vs_{name}'], group[f'mean_delta_vs_{name}'].mean(), atol=1e-12)
            for criterion in ['effect_threshold', 'ci_above_zero', 'all_six_positive', 'mae_not_worse']:
                assert item[f'panels_passing_{criterion}_vs_{name}'] == int(group[f'{criterion}_vs_{name}'].sum())
    alpha = [.01, .1, 1, 10, 100]
    for choice in choices:
        assert len(choice['selected_alpha']) == 4 and len(choice['validation_mse_by_alpha']) == 4
        for selected, losses in zip(choice['selected_alpha'], choice['validation_mse_by_alpha']):
            assert len(losses) == 5
            best = min(range(5), key=lambda j: (losses[j], -alpha[j]))
            assert selected == alpha[best]
        assert choice['fitted_coefficients'] == (389 if choice['arm'] == 'current_linear' else 425)
    assert len({g['x_sha256'] for g in generation}) == 144
    assert len({g['h_sha256'] for g in generation}) == 144
    assert result['calibration_checks_passed'] == (summary[0]['detected_panels'] <= 1 and summary[-1]['detected_panels'] >= 22)
    assert result['base_models_loaded'] == result['base_model_training_updates'] == 0
    assert result['reserved_model_seeds_opened'] == [] and not result['natural_history_evidence_added']
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256((root/'CALIBRATION_PROTOCOL.json').read_bytes()).hexdigest()
    assert digest == result['protocol_sha256']
    return {'status': 'passed', 'dataset_score_rows_verified': len(datasets), 'panel_decisions_reproduced': len(panels),
            'bootstrap_draws_checked': 24000, 'target_alpha_selections_verified': 1728,
            'shared_feature_tuning_records': len(choices), 'distinct_dataset_hashes': len(generation),
            'summary_scenarios_verified': len(summary), 'refitting_required': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    checked = verify(args.results)
    (args.results/'result_verification.json').write_text(json.dumps(checked, indent=2)+'\n')
    print(json.dumps(checked, indent=2))
