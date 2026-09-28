"""Production NumPy CHEL embedding path used by offline and API inference."""
import numpy as np

from .compress_ledger import exact_compact
from .evidence_ledger import (aggregate_tokens, apply_metric, hidden_features,
                              predict_utility, token_inputs, unit)
from .score_ledger import _forward_support


def student_from_raw(global_raw, local_tokens, first_model, second_model, compression):
    global_raw = unit(np.asarray(global_raw, dtype=np.float32))
    local_tokens = unit(np.asarray(local_tokens, dtype=np.float32))

    def branch(model, mean, components):
        projected = unit(local_tokens @ model["utility_projection"])
        hidden = hidden_features(token_inputs(projected), model["hidden_weights"],
                                 model["hidden_bias"])
        predictor = {"mean_h": model["predictor_mean_h"],
                     "mean_y": float(model["predictor_mean_y"]),
                     "beta": model["predictor_beta"]}
        utility = predict_utility(hidden, predictor, len(local_tokens), local_tokens.shape[1])
        _, weights, _ = aggregate_tokens(local_tokens, utility, float(model["temperature"]))
        support, _ = _forward_support(global_raw, local_tokens, weights, model, mean, components)
        global_branch = apply_metric(global_raw, model["global_center"],
                                     model["global_transform"])
        return global_branch, support

    global_first, support_first = branch(
        first_model, compression["v1_mean"], compression["v1_components"])
    global_second, support_second = branch(
        second_model, compression["v2_mean"], compression["v2_components"])
    if np.max(np.abs(global_first - global_second)) > 1e-7:
        raise AssertionError("Frozen global branches differ")
    return exact_compact(global_first, support_first, support_second)
