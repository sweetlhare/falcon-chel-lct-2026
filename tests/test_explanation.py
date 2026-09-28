import math
import unittest
from pathlib import Path

import numpy as np
import torch

from explanation import ExplanationError, explain_query_pixels, gallery_state


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


class DeadlineExplanationTests(unittest.TestCase):
    def test_good_batched_full_gallery_explanation_and_fill_sensitivity(self):
        calls = []

        def encoder(batch):
            calls.append(len(batch))
            result = []
            for image in batch:
                # Pixel-only toy encoder with different responses to gray and blur.
                result.append([1.0, float(image[:, :128, :128].mean()),
                               float(image[:, :, :128].std())])
            return normalized(result)

        ramp = torch.linspace(-2, 2, 256).view(1, 1, 256).expand(3, 256, 256).clone()
        gallery = normalized([[1, .2, .1], [1, -.5, .3], [.8, .6, -.2]])
        result = explain_query_pixels(encoder, ramp, gallery, ["target", "near", "third"], "target")
        self.assertEqual(calls, [2, 34])
        self.assertEqual(result["status"], "explained")
        self.assertEqual(np.asarray(result["arms"]["gray"]["target_margin_delta"]).shape, (4, 4))
        self.assertEqual(len(result["regions_xyxy"]), 16)
        self.assertEqual(result["controls"]["intervention_variants"], 32)
        self.assertEqual(result["controls"]["encoder_batch_size"], 34)
        self.assertGreater(result["agreement"]["mean_absolute_margin_delta_difference"], 0)

    def test_no_effect_has_zero_signed_maps_and_no_winner_change(self):
        def encoder(batch):
            return np.repeat([[1.0, 0.0]], len(batch), axis=0)

        gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
        for size in (256, 384):
            result = explain_query_pixels(encoder, torch.zeros(3, size, size), gallery,
                                          ["target", "other"], "target")
            for arm in result["arms"].values():
                np.testing.assert_array_equal(arm["target_score_delta"], np.zeros((4, 4)))
                np.testing.assert_array_equal(arm["target_margin_delta"], np.zeros((4, 4)))
                self.assertFalse(np.asarray(arm["changed_winner"]).any())
            self.assertEqual(result["agreement"]["margin_delta_comparable_count"], 0)
            self.assertIsNone(result["agreement"]["margin_delta_same_sign_fraction"])
            self.assertTrue(np.asarray(result["agreement"]["winner_id_same"]).all())

    def test_corrupt_zero_change_repeat_is_rejected(self):
        def corrupt(batch):
            result = np.repeat([[1.0, 0.0]], len(batch), axis=0)
            if len(batch) == 2:
                result[1] = [0.0, 1.0]
            return result

        with self.assertRaisesRegex(ExplanationError, "baseline repeat"):
            explain_query_pixels(corrupt, torch.zeros(3, 256, 256),
                                 np.eye(2), ["target", "other"], "target")

    def test_batch_numeric_drift_sets_noise_floor_and_hides_microeffects(self):
        calls = []

        def encoder(batch):
            calls.append(len(batch))
            base = np.repeat([[1.0, 0.0]], len(batch), axis=0)
            if len(batch) == 34:
                base[:, 1] = 5e-6
                # A raw intervention effect smaller than the measured floor.
                base[2, 1] += 1e-8
            return base

        result = explain_query_pixels(encoder, torch.zeros(3, 256, 256),
                                      np.eye(2), ["target", "other"], "target")
        self.assertEqual(calls, [2, 34])
        self.assertGreaterEqual(result["controls"]["measured_score_noise"], 4.9e-6)
        self.assertGreaterEqual(result["controls"]["effect_signal_boundary"], 1.9e-5)
        self.assertFalse(np.asarray(result["arms"]["gray"]["target_margin_signal_above_noise"]).any())
        self.assertNotEqual(result["arms"]["gray"]["target_margin_delta"][0][0], 0.0)

    def test_full_gallery_reselects_third_competitor(self):
        gallery = normalized([[1, 0], [.9, .1], [0, 1]])
        first = gallery_state(normalized([[1, .05]])[0], gallery,
                              ["target", "second", "third"], "target")
        changed = gallery_state(normalized([[.1, 1]])[0], gallery,
                                ["target", "second", "third"], "target")
        self.assertEqual(first.competitor_id, "second")
        self.assertEqual(changed.competitor_id, "third")
        self.assertLess(changed.margin, first.margin)

    def test_default_target_and_search_reference_are_same_batch_bound(self):
        def encoder(batch):
            return np.repeat([[1.0, .2]], len(batch), axis=0)

        gallery = normalized([[1, 0], [0, 1], [.5, .5]])
        reference = normalized([[1, .2]])[0]
        scores = gallery @ reference
        result = explain_query_pixels(encoder, torch.zeros(3, 256, 256), gallery,
                                      ["winner", "other", "third"], None,
                                      reference_embedding=reference,
                                      reference_scores=scores)
        self.assertEqual(result["target_id"], "winner")
        control = result["controls"]["search_reference"]
        self.assertLess(control["embedding_max_abs_error"], 1e-12)
        self.assertTrue(control["winner_id_agrees"])
        self.assertTrue(control["competitor_id_agrees"])

    def test_search_reference_drift_fails_closed(self):
        encoder = lambda batch: np.repeat([[1.0, 0.0]], len(batch), axis=0)
        with self.assertRaisesRegex(ExplanationError, "search-reference drift"):
            explain_query_pixels(encoder, torch.zeros(3, 256, 256), np.eye(2),
                                 ["target", "other"], "target",
                                 reference_embedding=np.array([0.0, 1.0]),
                                 reference_scores=np.array([0.0, 1.0]))

    def test_target_acceptance_is_separate_from_overall_refusal(self):
        encoder = lambda batch: np.repeat([[1.0, 0.0]], len(batch), axis=0)
        gallery = normalized([[1, 0], [.5, math.sqrt(.75)]])
        result = explain_query_pixels(encoder, torch.zeros(3, 256, 256), gallery,
                                      ["winner", "clicked"], "clicked", threshold=.75)
        self.assertEqual(result["status"], "explained")
        self.assertFalse(result["baseline"]["overall_refused"])
        self.assertFalse(result["baseline"]["target_accepted"])

    def test_unknown_refused_and_unsupported_multiview_semantics(self):
        calls = []
        encoder = lambda batch: calls.append(len(batch)) or np.repeat([[1., 0.]], len(batch), axis=0)
        unknown = explain_query_pixels(encoder, torch.zeros(3, 256, 256), np.eye(2),
                                       ["a", "b"], "missing")
        self.assertEqual(unknown["status"], "unknown_target")
        self.assertEqual(calls, [])
        refused = explain_query_pixels(encoder, torch.zeros(3, 256, 256), np.eye(2),
                                       ["a", "b"], "a", threshold=1.01)
        self.assertEqual(refused["status"], "refused")
        with self.assertRaisesRegex(ExplanationError, "multiview"):
            explain_query_pixels(encoder, torch.zeros(2, 3, 256, 256), np.eye(2),
                                 ["a", "b"], "a")

    def test_ui_binds_successful_search_and_displays_noise_contract(self):
        root = Path(__file__).parents[1]
        index = (root / "index.html").read_text()
        ui = (root / "explanation_ui.js").read_text()
        self.assertIn("falcon-search-start", index)
        self.assertIn("lastInput=inputSnapshot", index)
        self.assertIn("search_token:answer.search_token", index)
        self.assertIn("const input=lastInput", ui)
        self.assertIn("search_token:input.search_token", ui)
        self.assertIn("target_margin_signal_above_noise", ui)
        self.assertIn("margin_delta_comparable_count", ui)
        self.assertIn("changed_winner_uncertain", ui)
        self.assertNotIn("c.score.toFixed(3)", index)


if __name__ == "__main__":
    unittest.main()
