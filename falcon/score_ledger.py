"""Production support-branch forward pass shared by CHEL inference."""
import numpy as np


def _norm(x):
    return np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _forward_support(raw_global, tokens, weights, model, compression_mean, components):
    aggregate_pre = np.einsum("nt,ntd->nd", weights, tokens, optimize=True)
    aggregate = aggregate_pre / _norm(aggregate_pre)
    orthogonal = bool(int(model["orthogonal_support"])) if "orthogonal_support" in model else False
    if orthogonal:
        prepared_pre = aggregate - np.sum(aggregate * raw_global, axis=1, keepdims=True) * raw_global
        prepared = prepared_pre / _norm(prepared_pre)
    else:
        prepared_pre = aggregate
        prepared = aggregate
    projected_pre = prepared @ model["support_projection"]
    projected = projected_pre / _norm(projected_pre)
    metric_pre = (projected - model["support_center"]) @ model["support_transform"]
    metric = metric_pre / _norm(metric_pre)
    compressed_pre = (metric - compression_mean) @ components
    compressed = compressed_pre / _norm(compressed_pre)
    norms = {
        "aggregate": _norm(aggregate_pre), "prepared": _norm(prepared_pre),
        "projected": _norm(projected_pre), "metric": _norm(metric_pre),
        "compressed": _norm(compressed_pre),
    }
    return compressed.astype(np.float32), norms
