import unittest

import numpy as np

from falcon.deadline_metric import pair_covariances, psd_metric
from falcon.evidence_ledger import unit
from falcon.pixel_certificate import student_from_raw
from train_global_metric import self_test


class RuntimeMathTests(unittest.TestCase):
    def test_pair_covariances_match_explicit_ordered_pairs(self):
        values = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
        labels = np.array([0, 0, 1, 1])
        same, different = pair_covariances(values, labels)
        explicit = [[], []]
        for i in range(len(values)):
            for j in range(len(values)):
                if i == j:
                    continue
                delta = values[i] - values[j]
                explicit[int(labels[i] != labels[j])].append(np.outer(delta, delta))
        np.testing.assert_allclose(same, np.mean(explicit[0], axis=0), atol=1e-12)
        np.testing.assert_allclose(different, np.mean(explicit[1], axis=0), atol=1e-12)
        transform, report = psd_metric(same, different, .1)
        self.assertGreater(report['rank'], 0)
        np.testing.assert_allclose(transform, transform.T, atol=1e-12)

    def test_student_is_unit_1024_for_production_dimensions(self):
        rng = np.random.default_rng(20260928)
        def rand(*shape): return rng.normal(0, .2, shape).astype(np.float32)
        def model():
            return dict(utility_projection=rand(768, 16), hidden_weights=rand(19, 8),
                        hidden_bias=rand(8), predictor_mean_h=rand(8),
                        predictor_mean_y=np.array(.02, np.float32), predictor_beta=rand(8),
                        temperature=np.array(.3, np.float32), support_projection=rand(768, 192),
                        support_center=rand(192), support_transform=rand(192, 192),
                        global_center=rand(768), global_transform=np.eye(768, dtype=np.float32),
                        orthogonal_support=np.array(0))
        first = model(); second = model()
        second['global_center'] = first['global_center'].copy()
        second['global_transform'] = first['global_transform'].copy()
        compression = dict(v1_mean=rand(192), v1_components=rand(192, 128),
                           v2_mean=rand(192), v2_components=rand(192, 128))
        result = student_from_raw(rand(2, 768), rand(2, 64, 768), first, second, compression)
        self.assertEqual(result.shape, (2, 1024))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=2e-6)

    def test_training_cli_synthetic_controls(self):
        report = self_test()
        self.assertTrue(report['passed'])
        self.assertFalse(report['real_training_executed'])
        self.assertEqual(report['negative_controls'], 6)


if __name__ == '__main__':
    unittest.main()
