"""Pixel-intervention explanations for a single-view retrieval query.

The returned maps are exact full-gallery score responses to artificial input
changes.  They are not semantic part maps, identity confidence, or proof that a
pixel region causes an identity.  Occlusion and blur are out-of-distribution
interventions and effects from several cells need not add.
"""
from dataclasses import dataclass
import math

import numpy as np


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
LIMITATIONS = [
    "Artificial gray and Gaussian-blur interventions can be out of distribution.",
    "Maps show retrieval-score sensitivity, not semantic parts or identity causality.",
    "Signed effects from separate cells are not assumed to be additive.",
    "Map magnitude is not match confidence or a calibrated probability.",
    "Only one prepared 256x256 or 384x384 view is supported.",
]


class ExplanationError(ValueError):
    pass


@dataclass(frozen=True)
class GalleryState:
    target_score: float
    competitor_score: float
    margin: float
    competitor_id: object
    winner_id: object
    winner_score: float
    winner_gap: float


def _as_embeddings(value, expected_rows):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, (tuple, list)) and len(value) and not isinstance(value, np.ndarray):
        # Current delivery encoder returns (embeddings, pooling_weights).
        first = value[0]
        if hasattr(first, "shape") and len(first.shape) == 2:
            value = first.detach().cpu().numpy() if hasattr(first, "detach") else np.asarray(first)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != expected_rows or not np.isfinite(array).all():
        raise ExplanationError("encode_batch must return a finite [batch,dimension] matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ExplanationError("encode_batch returned a zero embedding")
    return array / norms


def _gallery(gallery_unit_vectors, gallery_ids):
    gallery = np.asarray(gallery_unit_vectors, dtype=np.float64)
    ids = list(gallery_ids)
    if gallery.ndim != 2 or len(ids) != len(gallery) or not len(ids):
        raise ExplanationError("gallery vectors and IDs must be aligned and nonempty")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ExplanationError("gallery IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ExplanationError("gallery IDs must be unique")
    if not np.isfinite(gallery).all():
        raise ExplanationError("gallery vectors must be finite")
    norms = np.linalg.norm(gallery, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-4):
        raise ExplanationError("gallery vectors must be unit normalized")
    return gallery, ids


def gallery_state(query_unit_vector, gallery_unit_vectors, gallery_ids, target_id):
    """Score the full gallery and select the strongest non-target competitor."""
    gallery, ids = _gallery(gallery_unit_vectors, gallery_ids)
    if target_id not in ids:
        raise ExplanationError("target_id is absent from gallery")
    query = np.asarray(query_unit_vector, dtype=np.float64)
    if query.shape != (gallery.shape[1],) or not np.isfinite(query).all():
        raise ExplanationError("query embedding does not match gallery dimension")
    norm = np.linalg.norm(query)
    if norm <= 0:
        raise ExplanationError("query embedding is zero")
    scores = gallery @ (query / norm)
    return _state_from_scores(scores, ids, target_id)


def _state_from_scores(scores, ids, target_id):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ExplanationError("reference scores must be one finite score per gallery ID")
    target = ids.index(target_id)
    competitors = np.arange(len(ids)) != target
    if not competitors.any():
        raise ExplanationError("a target explanation needs at least one competitor")
    competitor = int(np.flatnonzero(competitors)[np.argmax(scores[competitors])])
    order = np.argsort(-scores, kind="stable")
    winner = int(order[0])
    return GalleryState(float(scores[target]), float(scores[competitor]),
                        float(scores[target] - scores[competitor]), ids[competitor],
                        ids[winner], float(scores[winner]),
                        float(scores[order[0]] - scores[order[1]]))


def _blur(tensor, kernel_size=31, sigma=10.0):
    import torch
    import torch.nn.functional as functional
    radius = kernel_size // 2
    coordinate = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    kernel = torch.exp(-(coordinate * coordinate) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).expand(3, 1, kernel_size, kernel_size)
    padded = functional.pad(tensor[None], (radius,) * 4, mode="reflect")
    return functional.conv2d(padded, kernel, groups=3)[0]


def _same_sign(a, b, tolerance):
    a, b = np.asarray(a), np.asarray(b)
    return ((np.abs(a) <= tolerance) & (np.abs(b) <= tolerance)) | ((a > tolerance) & (b > tolerance)) | ((a < -tolerance) & (b < -tolerance))


def explain_query_pixels(encode_batch, original_tensor, gallery_unit_vectors,
                         gallery_ids, target_id, grid=4, mean=DEFAULT_MEAN,
                         std=DEFAULT_STD, gray_rgb=127, threshold=None,
                         numeric_tolerance=1e-4, minimum_effect=1e-6,
                         reference_embedding=None, reference_scores=None,
                         reference_tolerance=1e-4):
    """Return signed 4x4 response maps for gray and Gaussian-blur interventions.

    Raw deltas are ``intervened - original``. Positive margin delta means that
    the intervention increased target score relative to the strongest
    competitor, which is reselected from the full gallery for every
    intervention. Separate masks identify effects above the measured numeric
    noise floor; raw sub-floor values remain available for exact accounting.
    """
    import torch
    if not isinstance(original_tensor, torch.Tensor) or original_tensor.device.type != "cpu":
        raise ExplanationError("original_tensor must be a CPU torch tensor")
    if original_tensor.ndim != 3 or original_tensor.shape[0] != 3:
        raise ExplanationError("multiview/batched input is unsupported; expected [3,H,W]")
    _, height, width = original_tensor.shape
    if height != width or height not in (256, 384) or height % grid or grid != 4:
        raise ExplanationError("supported shape is one [3,256,256] or [3,384,384] view with grid=4")
    if not original_tensor.is_floating_point() or not torch.isfinite(original_tensor).all():
        raise ExplanationError("original_tensor must be finite floating point")
    gallery, ids = _gallery(gallery_unit_vectors, gallery_ids)
    if target_id is not None and target_id not in ids:
        return {"status": "unknown_target", "target_id": target_id, "refused": True,
                "limitations": LIMITATIONS.copy()}
    if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(v) for v in (*mean, *std)) or any(v <= 0 for v in std):
        raise ExplanationError("mean/std must contain three finite values and std must be positive")
    if not isinstance(gray_rgb, (int, float)) or not math.isfinite(gray_rgb) or not 0 <= gray_rgb <= 255:
        raise ExplanationError("gray_rgb must be finite and between 0 and 255")
    if threshold is not None and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ExplanationError("threshold must be finite")
    if (not math.isfinite(numeric_tolerance) or numeric_tolerance <= 0 or
            not math.isfinite(minimum_effect) or minimum_effect <= 0 or
            not math.isfinite(reference_tolerance) or reference_tolerance <= 0):
        raise ExplanationError("numeric and reference tolerances must be positive and finite")

    original = original_tensor.detach().clone()
    repeated = _as_embeddings(encode_batch([original, original.clone()]), 2)
    batch2_scores = repeated @ gallery.T
    repeat_embedding_error = float(np.max(np.abs(repeated[0] - repeated[1])))
    repeat_score_error = float(np.max(np.abs(batch2_scores[0] - batch2_scores[1])))
    if max(repeat_embedding_error, repeat_score_error) > numeric_tolerance:
        raise ExplanationError("zero-change baseline repeat exceeded tolerance")

    gray = torch.tensor([(gray_rgb / 255.0 - m) / s for m, s in zip(mean, std)],
                        dtype=original.dtype).view(3, 1, 1)
    blurred = _blur(original)
    variants = []
    regions = []
    for fill in ("gray", "gaussian_blur"):
        for row in range(grid):
            for column in range(grid):
                top, bottom = row * height // grid, (row + 1) * height // grid
                left, right = column * width // grid, (column + 1) * width // grid
                changed = original.clone()
                changed[:, top:bottom, left:right] = (gray if fill == "gray" else blurred[:, top:bottom, left:right])
                variants.append(changed)
                if fill == "gray":
                    regions.append([left, top, right, bottom])
    if len(variants) > 32:
        raise AssertionError("intervention batch exceeds 32 variants")
    # Two originals measure both same-batch repeat noise and batch2-vs-batch34
    # numeric drift. Interventions use the first same-batch original as baseline.
    intervention_batch = [original.clone(), original.clone()] + variants
    encoded = _as_embeddings(encode_batch(intervention_batch), len(intervention_batch))
    controls, embeddings = encoded[:2], encoded[2:]
    control_scores = controls @ gallery.T
    same_batch_embedding_error = float(np.max(np.abs(controls[0] - controls[1])))
    same_batch_score_error = float(np.max(np.abs(control_scores[0] - control_scores[1])))
    cross_batch_embedding_error = float(np.max(np.abs(
        controls[:, None, :] - repeated[None, :, :])))
    cross_batch_score_error = float(np.max(np.abs(
        control_scores[:, None, :] - batch2_scores[None, :, :])))
    measured_embedding_noise = max(repeat_embedding_error, same_batch_embedding_error,
                                   cross_batch_embedding_error)
    measured_score_noise = max(repeat_score_error, same_batch_score_error,
                               cross_batch_score_error)
    if max(measured_embedding_noise, measured_score_noise) > numeric_tolerance:
        raise ExplanationError("original-control numeric drift exceeded tolerance")
    effect_boundary = max(float(minimum_effect), 4.0 * measured_score_noise)
    if target_id is None:
        target_id = ids[int(np.argmax(control_scores[0]))]
    baseline = gallery_state(controls[0], gallery, ids, target_id)

    reference = None
    if reference_embedding is not None or reference_scores is not None:
        if reference_embedding is None or reference_scores is None:
            raise ExplanationError("search reference requires both embedding and full-gallery scores")
        raw_reference = np.asarray(reference_embedding)
        if raw_reference.ndim != 1:
            raise ExplanationError("reference_embedding must be one vector")
        reference_vector = _as_embeddings(raw_reference[None, :], 1)[0]
        supplied_scores = np.asarray(reference_scores, dtype=np.float64)
        reference_state = _state_from_scores(supplied_scores, ids, target_id)
        reconstructed_scores = gallery @ reference_vector
        reference_embedding_error = float(np.max(np.abs(reference_vector - controls[0])))
        reference_score_error = float(np.max(np.abs(supplied_scores - control_scores[0])))
        reference_self_error = float(np.max(np.abs(supplied_scores - reconstructed_scores)))
        if max(reference_embedding_error, reference_score_error,
               reference_self_error) > reference_tolerance:
            raise ExplanationError("search-reference drift exceeded tolerance")
        reference = {
            "embedding_max_abs_error": reference_embedding_error,
            "full_gallery_score_max_abs_error": reference_score_error,
            "embedding_score_self_consistency_max_abs_error": reference_self_error,
            "target_score_abs_error": abs(reference_state.target_score - baseline.target_score),
            "winner_id_agrees": reference_state.winner_id == baseline.winner_id,
            "competitor_id_agrees": reference_state.competitor_id == baseline.competitor_id,
            "tolerance": reference_tolerance,
        }

    arms = {}
    states = []
    for embedding in embeddings:
        states.append(gallery_state(embedding, gallery, ids, target_id))
    for arm_index, fill in enumerate(("gray", "gaussian_blur")):
        selected = states[arm_index * grid * grid:(arm_index + 1) * grid * grid]
        score_delta = np.array([s.target_score - baseline.target_score for s in selected]).reshape(grid, grid)
        margin_delta = np.array([s.margin - baseline.margin for s in selected]).reshape(grid, grid)
        changed_winner = np.array([s.winner_id != baseline.winner_id for s in selected], dtype=bool).reshape(grid, grid)
        uncertain_winner = np.array([
            (s.winner_id != baseline.winner_id) and
            min(s.winner_gap, baseline.winner_gap) <= effect_boundary
            for s in selected], dtype=bool).reshape(grid, grid)
        arms[fill] = {
            "target_score_delta": score_delta.tolist(),
            "target_margin_delta": margin_delta.tolist(),
            "target_score_signal_above_noise": (np.abs(score_delta) > effect_boundary).tolist(),
            "target_margin_signal_above_noise": (np.abs(margin_delta) > effect_boundary).tolist(),
            "changed_winner": changed_winner.tolist(),
            "changed_winner_uncertain": uncertain_winner.tolist(),
            "winner_id": [[selected[r * grid + c].winner_id for c in range(grid)] for r in range(grid)],
            "competitor_id": [[selected[r * grid + c].competitor_id for c in range(grid)] for r in range(grid)],
            "target_score": [[selected[r * grid + c].target_score for c in range(grid)] for r in range(grid)],
            "margin": [[selected[r * grid + c].margin for c in range(grid)] for r in range(grid)],
        }
    gray_delta = np.asarray(arms["gray"]["target_margin_delta"])
    blur_delta = np.asarray(arms["gaussian_blur"]["target_margin_delta"])
    gray_signal = np.abs(gray_delta) > effect_boundary
    blur_signal = np.abs(blur_delta) > effect_boundary
    comparable = gray_signal & blur_signal
    agreement = comparable & _same_sign(gray_delta, blur_delta, effect_boundary)
    comparable_count = int(comparable.sum())
    agreement_count = int(agreement.sum())
    gray_states, blur_states = states[:grid * grid], states[grid * grid:]
    winner_id_same = np.array([a.winner_id == b.winner_id
                               for a, b in zip(gray_states, blur_states)]).reshape(grid, grid)
    refused = bool(threshold is not None and baseline.winner_score < threshold)
    target_accepted = None if threshold is None else bool(baseline.target_score >= threshold)
    overall_refusal_uncertain = bool(
        threshold is not None and abs(baseline.winner_score - threshold) <= effect_boundary)
    target_acceptance_uncertain = bool(
        threshold is not None and abs(baseline.target_score - threshold) <= effect_boundary)
    return {
        "status": "refused" if refused else "explained",
        "refused": refused,
        "target_id": target_id,
        "grid": [grid, grid],
        "regions_xyxy": regions,
        "interventions": {
            "gray": {"rgb": [gray_rgb] * 3, "normalization_mean": list(mean),
                     "normalization_std": list(std)},
            "gaussian_blur": {"kernel_size": 31, "sigma": 10.0},
        },
        "delta_definition": "intervened minus original; competitor is reselected from full gallery",
        "baseline": {
            "target_score": baseline.target_score,
            "competitor_score": baseline.competitor_score,
            "margin": baseline.margin,
            "competitor_id": baseline.competitor_id,
            "winner_id": baseline.winner_id,
            "winner_score": baseline.winner_score,
            "winner_gap": baseline.winner_gap,
            "threshold": threshold,
            "target_accepted": target_accepted,
            "target_threshold_gap": None if threshold is None else baseline.target_score - threshold,
            "target_acceptance_uncertain": target_acceptance_uncertain,
            "overall_refused": refused,
            "overall_threshold_gap": None if threshold is None else baseline.winner_score - threshold,
            "overall_refusal_uncertain": overall_refusal_uncertain,
        },
        "arms": arms,
        "agreement": {
            "margin_delta_same_sign": agreement.tolist(),
            "margin_delta_comparable": comparable.tolist(),
            "margin_delta_comparable_count": comparable_count,
            "margin_delta_same_sign_count": agreement_count,
            "margin_delta_same_sign_fraction": (agreement_count / comparable_count
                                                 if comparable_count else None),
            "winner_id_same": winner_id_same.tolist(),
            "winner_id_same_fraction": float(winner_id_same.mean()),
            "mean_absolute_margin_delta_difference": float(np.mean(np.abs(gray_delta - blur_delta))),
        },
        "controls": {
            "baseline_repeat_embedding_max_abs_error": repeat_embedding_error,
            "baseline_repeat_score_max_abs_error": repeat_score_error,
            "same_batch_original_pair_embedding_max_abs_error": same_batch_embedding_error,
            "same_batch_original_pair_score_max_abs_error": same_batch_score_error,
            "batch2_vs_batch34_embedding_max_abs_error": cross_batch_embedding_error,
            "batch2_vs_batch34_score_max_abs_error": cross_batch_score_error,
            "measured_embedding_noise": measured_embedding_noise,
            "measured_score_noise": measured_score_noise,
            "effect_signal_boundary": effect_boundary,
            "numeric_tolerance": numeric_tolerance,
            "intervention_baseline": "first original control in the same batch34",
            "full_gallery_rescored_for_every_variant": True,
            "intervention_variants": len(variants),
            "encoder_batch_size": len(intervention_batch),
            "model_input_metadata": "pixels only; gallery IDs are used after encoding",
            "search_reference": reference,
        },
        "limitations": LIMITATIONS.copy(),
    }
