from pathlib import Path

from eval.run_eval import run_eval


def test_run_eval_offline_toy_dataset():
    out = run_eval(Path('eval/datasets/toy_papers'))
    assert 'metrics' in out and 'results' in out
    assert out['metrics']['schema_valid_rate'] >= 0.0
    assert len(out['results']) == 4
