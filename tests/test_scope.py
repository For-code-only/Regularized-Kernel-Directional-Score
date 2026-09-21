"""Frozen paper scope and data inventory checks, without model fitting."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with gzip.open(ROOT / 'config/manifests' / (name + '.csv.gz'), 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class ScopeTests(unittest.TestCase):
    def test_suite_counts(self):
        expected = {'anm': 58320, 'sensitivity': 1944, 'tcep': 1122, 'nonanm': 3600}
        ids = set()
        for suite, count in expected.items():
            data = rows(suite)
            self.assertEqual(len(data), count)
            self.assertEqual(len({r['case_id'] for r in data}), count)
            self.assertFalse(ids & {r['case_id'] for r in data})
            ids.update(r['case_id'] for r in data)
        self.assertEqual(len(ids), 64986)
        self.assertEqual(6 * (58320 + 1122 + 3600) + 12 * 1944, 401580)

    def test_anm_and_sensitivity(self):
        anm, sensitivity = rows('anm'), rows('sensitivity')
        self.assertEqual(Counter(int(r['n']) for r in anm), {200: 19440, 500: 19440, 1000: 19440})
        self.assertEqual({r['n'] for r in sensitivity}, {'1000'})
        self.assertEqual(len({r['cell_id'] for r in sensitivity}), 972)
        self.assertEqual(Counter(Counter(r['cell_id'] for r in sensitivity).values()), {2: 972})
        self.assertEqual(len({r['function_name'] for r in anm}), 12)
        self.assertEqual(len({r['cause_name'] for r in anm}), 3)
        self.assertEqual(len({r['noise_name'] for r in anm}), 9)
        self.assertEqual({float(r['rho']) for r in anm}, {.5, 1, 2})

    def test_tcep_indices_and_sources(self):
        data = rows('tcep')
        self.assertEqual(Counter(r['cap'] for r in data), {'1000': 1020, 'full': 102})
        self.assertEqual(len({r['pair_id'] for r in data}), 102)
        with gzip.open(ROOT / 'data/tcep/indices.json.gz', 'rt') as handle:
            index = json.load(handle)
        self.assertEqual(index['index_base'], 1)
        for row in data:
            entry = index['cases'][row['case_id']]
            self.assertEqual(entry['case_id'], row['case_id'])
            self.assertEqual(entry['row_seed'], int(row['row_seed']))
            self.assertEqual(len(entry['original_rows']), int(row['n']))
            self.assertEqual(len(set(entry['original_rows'])), int(row['n']))
            self.assertGreaterEqual(min(entry['original_rows']), 1)
            self.assertLessEqual(max(entry['original_rows']), int(row['original_n']))
        with (ROOT / 'data/tcep/pairs.csv').open(newline='') as handle:
            pairs = list(csv.DictReader(handle))
        self.assertEqual(len(pairs), 102)
        self.assertAlmostEqual(sum(float(r['weight']) for r in pairs), 38.4979, 10)
        continuous = [r for r in pairs if r['continuous_continuous'] == 'yes']
        self.assertEqual(len(continuous), 67)
        self.assertAlmostEqual(sum(float(r['weight']) for r in continuous), 22.7063, 10)
        with zipfile.ZipFile(ROOT / 'data/tcep/observations.zip') as archive:
            self.assertEqual(len(archive.namelist()), 102)
            for pair in pairs:
                raw = archive.read('pair%04d.csv' % int(pair['pair_id']))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), pair['cache_sha256'])

    def test_nonanm_pairing(self):
        data = rows('nonanm')
        worlds = {}
        for row in data:
            worlds.setdefault(row['world_id'], []).append(row)
            self.assertEqual(float(row['rho']), 1)
            self.assertEqual(float(row['delta']), .2)
            self.assertEqual(int(row['generation_attempts']), 1)
        self.assertEqual(len(worlds), 900)
        self.assertEqual(Counter(int(v[0]['n']) for v in worlds.values()), {200: 300, 500: 300, 1000: 300})
        expected_models = {'ANM_control', 'heteroscedastic', 'post_nonlinear', 'multiplicative'}
        for group in worlds.values():
            self.assertEqual({r['model'] for r in group}, expected_models)
            for name in ['function_name', 'cause_name', 'noise_name', 'outer_id', 'data_seed', 'design_seed', 'pnl_seed']:
                self.assertEqual(len({r[name] for r in group}), 1)
        distributions = {'Gaussian', 'Laplace', 'Gamma_shape2_scale1', 'bimodal'}
        self.assertEqual({r['cause_name'] for r in data}, distributions)
        self.assertEqual({r['noise_name'] for r in data}, distributions)
        self.assertEqual(len({r['function_name'] for r in data}), 6)

    def test_engineering_cases_are_isolated(self):
        production = {r['case_id'] for suite in ['anm', 'sensitivity', 'tcep', 'nonanm'] for r in rows(suite)}
        smoke = rows('smoke')
        self.assertEqual(len(smoke), 7)
        self.assertFalse(production & {r['case_id'] for r in smoke})
        self.assertEqual(Counter(r['smoke_suite'] for r in smoke), {'anm': 1, 'tcep': 1, 'sensitivity': 1, 'nonanm': 4})


if __name__ == '__main__':
    unittest.main()
