import unittest

import numpy as np

from falcon.compress_ledger import (compressed_embedding, exact_compact,
                                    project_support, promotion_decision,
                                    support_stack)
from falcon.evidence_ledger import unit
from falcon.ledger_fusion import fused


class CompressLedgerTests(unittest.TestCase):
    def test_exact_compact_preserves_teacher_cosine(self):
        rng = np.random.default_rng(7)
        global_embedding = unit(rng.normal(size=(12, 8)))
        first = unit(rng.normal(size=(12, 3)))
        second = unit(rng.normal(size=(12, 5)))
        v1 = unit(np.concatenate([global_embedding*np.sqrt(.75), first*np.sqrt(.25)], axis=1))
        v2 = unit(np.concatenate([global_embedding*np.sqrt(.75), second*np.sqrt(.25)], axis=1))
        teacher = fused(v1, v2, .75)
        compact = exact_compact(global_embedding, first, second)
        np.testing.assert_allclose(teacher @ teacher.T, compact @ compact.T, atol=2e-6)

    def test_damaged_support_breaks_exact_equivalence(self):
        rng = np.random.default_rng(11)
        global_embedding = unit(rng.normal(size=(9, 6)))
        first = unit(rng.normal(size=(9, 4)))
        second = unit(rng.normal(size=(9, 4)))
        reference = exact_compact(global_embedding, first, second)
        damaged = exact_compact(global_embedding, first[::-1], second)
        self.assertGreater(np.max(np.abs(reference@reference.T-damaged@damaged.T)), 1e-3)

    def test_projection_is_unit_and_rejects_zero_map(self):
        rng = np.random.default_rng(3); features = unit(rng.normal(size=(10, 6)))
        result = project_support(features, np.zeros(6), np.eye(6, 3))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            project_support(features, np.zeros(6), np.zeros((6, 3)))

    def test_joint_support_projection_keeps_global_and_target_dimension(self):
        rng = np.random.default_rng(13)
        global_embedding = unit(rng.normal(size=(10, 8)))
        first = unit(rng.normal(size=(10, 3)))
        second = unit(rng.normal(size=(10, 5)))
        stacked = support_stack(first, second)
        np.testing.assert_allclose(np.linalg.norm(stacked, axis=1), 1, atol=1e-6)
        model = {"joint_mean": np.zeros(8), "joint_components": np.eye(8, 4)}
        result = compressed_embedding(global_embedding, first, second, model)
        self.assertEqual(result.shape, (10, 12))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)

    def test_promotion_validator_passes_known_good_and_rejects_bad_controls(self):
        good = dict(dimension=1024, teacher_gain=.010, student_gain=.0095, retained=.95,
                    teacher_map=.246, student_map=.245, student_ci_lower=.001,
                    global_f1=.20, student_f1=.205, global_tnr=.225, student_tnr=.23)
        self.assertEqual(promotion_decision(**good)[0], "promote")

        too_wide = {**good, "dimension": 2496}
        no_gain = {**good, "student_gain": 0.0, "retained": 0.0,
                   "student_ci_lower": 0.0}
        lazy = {**good, "student_f1": .18, "student_tnr": .40}
        self.assertEqual(promotion_decision(**too_wide)[0], "reject")
        self.assertEqual(promotion_decision(**no_gain)[0], "reject")
        self.assertEqual(promotion_decision(**lazy)[0], "reject")


if __name__ == "__main__": unittest.main()
