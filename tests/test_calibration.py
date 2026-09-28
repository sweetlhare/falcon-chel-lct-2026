"""Exact calibration agrees with an independent exhaustive threshold oracle."""
import unittest
import numpy as np
from falcon.evaluate import calibrate,candidate_metrics


class CalibrationTests(unittest.TestCase):
    def test_tied_scores_and_top10_against_exhaustive_thresholds(self):
        rng=np.random.default_rng(20260915)
        for _ in range(20):
            scores=np.round(rng.random((12,17)),2)
            valid=rng.random(scores.shape)>.2
            positive=np.zeros_like(valid)
            known=np.array([True]*8+[False]*4)
            for i in range(8):positive[i,rng.choice(np.flatnonzero(valid[i]),2,replace=False)]=True
            thresholds=np.r_[np.nextafter(scores[valid].max(),np.inf),np.unique(scores[valid])]
            expected=max((candidate_metrics(scores,valid,positive,known,t) for t in thresholds),
                         key=lambda r:(r['pair_F1'],r['unknown_query_TNR'],r['threshold']))
            actual=calibrate(scores,valid,positive,known)
            for key in ('pair_F1','unknown_query_TNR','threshold','tp','fp','fn'):
                self.assertEqual(actual[key],expected[key])
