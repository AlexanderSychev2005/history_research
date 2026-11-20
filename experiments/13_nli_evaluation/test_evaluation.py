"""Tests for NLI evaluation; no precomputed output files are required."""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluation as m


class EvaluationTest(unittest.TestCase):
    def test_gold_scope(self):
        ev = m.load_evaluator()
        gold = ev.load_dir(m.ROOT / 'archive/eval/gold')
        self.assertEqual(sum(len(d['scope']['paragraphs']) for d in gold.values()), 14)
        self.assertEqual(sum(len(d['facts']) for d in gold.values()), 54)

    def test_prediction_cache_available(self):
        for source in ('gold', 'lela'):
            score_file = m.ROOT / 'experiments/09_nli_extended/results' / f'{source}_sentence_sentence_scores.json'
            paired_file = m.ROOT / 'experiments/11_nli_error_driven/results/reverse_probe' / f'{source}_ensemble_t0p9_paired_scores.json'
            self.assertTrue(score_file.exists(), f'Missing {score_file}; run earlier NLI inference first')
            self.assertTrue(paired_file.exists(), f'Missing {paired_file}; run reverse_probe.py first')

    def test_evaluate_end_to_end_from_inputs(self):
        # Use a temporary directory to prove the script creates outputs from source caches.
        with tempfile.TemporaryDirectory() as td:
            rows = m.main(['--out', td, '--bootstrap', '20'])
            by_name = {r['run']: r for r in rows}
            self.assertIn('nli_gold_reverse_d0p2', by_name)
            self.assertIn('nli_lela_reverse_d0p2', by_name)
            self.assertEqual(by_name['nli_gold_reverse_d0p2']['gold'], 54)
            self.assertEqual(by_name['nli_gold_reverse_d0p2']['issues'], 0)
            for name in ('comparison.csv', 'per_relation.csv', 'errors.csv', 'validation.csv', 'RESULTS.md'):
                self.assertTrue((Path(td) / name).exists())


if __name__ == '__main__':
    unittest.main()
