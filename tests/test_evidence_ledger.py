import unittest

import numpy as np

from falcon.evidence_ledger import (aggregate_tokens, choose_contrastive_sets,
                                    choose_counterfactual_pairs, counterfactual_targets, unit)


class EvidenceLedgerTests(unittest.TestCase):
    def test_pairs_use_cross_camera_positive_and_different_id_negative(self):
        features = unit(np.array([[1, 0], [.9, .1], [0, 1], [.1, .9]], np.float32))
        ids = np.array(["a", "a", "b", "b"])
        cameras = np.array(["1", "2", "1", "2"])
        positive, negative = choose_counterfactual_pairs(features, ids, cameras)
        self.assertTrue(np.all(ids[positive] == ids))
        self.assertTrue(np.all(cameras[positive] != cameras))
        self.assertTrue(np.all(ids[negative] != ids))

    def test_counterfactual_target_rewards_persistent_unique_token(self):
        tokens = unit(np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [1, 1]],
            [[0, 1], [-1, 0]],
        ], np.float32))
        target, positive, negative = counterfactual_targets(tokens, np.array([1, 0, 0]), np.array([2, 2, 1]))
        self.assertGreater(target[0, 0], target[0, 1])
        self.assertGreater(positive[0, 0], negative[0, 0])

    def test_robust_sets_have_requested_shape_and_valid_labels(self):
        features = unit(np.array([[1, 0], [.9, .1], [.8, .2], [0, 1], [.1, .9], [.2, .8]], np.float32))
        ids = np.array(["a", "a", "a", "b", "b", "b"])
        cameras = np.array(["1", "2", "3", "1", "2", "3"])
        positive, negative = choose_contrastive_sets(features, ids, cameras, 2, 3)
        self.assertEqual(positive.shape, (6, 2))
        self.assertEqual(negative.shape, (6, 3))
        self.assertTrue(np.all(ids[positive] == ids[:, None]))
        self.assertTrue(np.all(cameras[positive] != cameras[:, None]))
        self.assertTrue(np.all(ids[negative] != ids[:, None]))

    def test_high_utility_token_dominates_aggregate(self):
        tokens = np.zeros((1, 64, 2), np.float32)
        tokens[0, :, 1] = 1
        tokens[0, 7] = [1, 0]
        utility = np.zeros((1, 64), np.float32)
        utility[0, 7] = 1
        aggregate, weights, entropy = aggregate_tokens(tokens, utility, temperature=.03)
        self.assertGreater(weights[0, 7], .99)
        self.assertGreater(aggregate[0, 0], .99)
        self.assertLess(entropy[0], .1)


if __name__ == "__main__":
    unittest.main()
