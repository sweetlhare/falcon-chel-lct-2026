"""Contrastive hard-negative evidence ledger for fixed vehicle embeddings.

The backbone stays frozen.  Training labels teach a small token-utility model:
a useful region must match a cross-camera image of the same vehicle while not
matching the nearest different-ID vehicle.  The predicted utilities are saved
as an inspectable 8x8 evidence map and aggregate local tokens into one fixed
embedding branch.  Model selection uses calibration IDs only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import calibrate, candidate_metrics, load_scores, read_rows, retrieval_metrics


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def orthogonal_projection(source_dim, target_dim, seed):
    if target_dim > source_dim:
        raise ValueError("Projection cannot increase dimension")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((source_dim, target_dim)))
    return q.astype(np.float32)


def choose_counterfactual_pairs(global_features, vehicle_ids, camera_ids):
    """Choose easiest cross-camera positive and nearest different-ID negative."""
    features = unit(global_features)
    vehicle_ids = np.asarray(vehicle_ids)
    camera_ids = np.asarray(camera_ids)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    positives = np.empty(len(features), dtype=np.int64)
    negatives = np.empty(len(features), dtype=np.int64)
    for i in range(len(features)):
        positive = (vehicle_ids == vehicle_ids[i]) & (camera_ids != camera_ids[i])
        negative = vehicle_ids != vehicle_ids[i]
        if not positive.any():
            raise ValueError(f"Training row {i} has no cross-camera positive")
        positives[i] = np.flatnonzero(positive)[np.argmax(similarity[i, positive])]
        negatives[i] = np.flatnonzero(negative)[np.argmax(similarity[i, negative])]
    return positives, negatives


def choose_contrastive_sets(global_features, vehicle_ids, camera_ids, positive_k, negative_k):
    """Choose small similarity-ranked positive and negative sets, padding only if needed."""
    features = unit(global_features)
    vehicle_ids = np.asarray(vehicle_ids)
    camera_ids = np.asarray(camera_ids)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    positives = np.empty((len(features), positive_k), dtype=np.int64)
    negatives = np.empty((len(features), negative_k), dtype=np.int64)
    for i in range(len(features)):
        positive = np.flatnonzero((vehicle_ids == vehicle_ids[i]) & (camera_ids != camera_ids[i]))
        negative = np.flatnonzero(vehicle_ids != vehicle_ids[i])
        if not len(positive):
            raise ValueError(f"Training row {i} has no cross-camera positive")
        positive = positive[np.argsort(-similarity[i, positive], kind="stable")]
        negative = negative[np.argsort(-similarity[i, negative], kind="stable")]
        positives[i] = np.resize(positive[:positive_k], positive_k)
        negatives[i] = np.resize(negative[:negative_k], negative_k)
    return positives, negatives


def counterfactual_targets(projected_tokens, positive_index, negative_index, batch_size=64):
    """Per-token persistence minus nearest-impostor similarity."""
    tokens = unit(projected_tokens)
    targets = np.empty(tokens.shape[:2], dtype=np.float32)
    positive_match = np.empty_like(targets)
    negative_match = np.empty_like(targets)
    positive_index = np.asarray(positive_index)
    negative_index = np.asarray(negative_index)
    if positive_index.ndim == 1:
        positive_index = positive_index[:, None]
    if negative_index.ndim == 1:
        negative_index = negative_index[:, None]
    for start in range(0, len(tokens), batch_size):
        stop = min(start + batch_size, len(tokens))
        anchor = tokens[start:stop]
        pos_values = []
        neg_values = []
        for column in range(positive_index.shape[1]):
            positive = tokens[positive_index[start:stop, column]]
            pos_values.append(np.einsum("btd,bsd->bts", anchor, positive, optimize=True).max(axis=2))
        for column in range(negative_index.shape[1]):
            negative = tokens[negative_index[start:stop, column]]
            neg_values.append(np.einsum("btd,bsd->bts", anchor, negative, optimize=True).max(axis=2))
        pos = np.mean(pos_values, axis=0)
        neg = np.mean(neg_values, axis=0)
        positive_match[start:stop] = pos
        negative_match[start:stop] = neg
        targets[start:stop] = np.clip(pos - neg, -0.5, 0.5)
    return targets, positive_match, negative_match


def token_coordinates(side=8):
    axis = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([xx, yy, xx * xx + yy * yy], axis=-1).reshape(side * side, 3)


def token_inputs(projected_tokens):
    projected_tokens = unit(projected_tokens)
    coords = token_coordinates(int(round(projected_tokens.shape[1] ** 0.5)))
    if len(coords) != projected_tokens.shape[1]:
        raise ValueError("Token grid is not square")
    tiled = np.broadcast_to(coords, (len(projected_tokens),) + coords.shape)
    return np.concatenate([projected_tokens, tiled], axis=2).astype(np.float32)


def hidden_features(inputs, weights, bias):
    flat = inputs.reshape(-1, inputs.shape[-1])
    return np.tanh(flat @ weights + bias).astype(np.float32)


def fit_predictor_at_ridge(hidden, targets, image_mask, ridge):
    train = np.repeat(np.asarray(image_mask, bool), targets.shape[1])
    y = targets.reshape(-1).astype(np.float64)
    h = hidden.astype(np.float64)
    mean_h = h[train].mean(axis=0)
    mean_y = float(y[train].mean())
    x = h[train] - mean_h
    xtx = x.T @ x
    scale = float(np.trace(xtx) / len(mean_h))
    beta = np.linalg.solve(xtx + np.eye(len(mean_h)) * ridge * scale, x.T @ (y[train] - mean_y))
    return {"ridge": float(ridge), "mean_h": mean_h.astype(np.float32), "mean_y": mean_y,
            "beta": beta.astype(np.float32)}


def fit_ridge_predictor(hidden, targets, train_mask, validation_mask, ridges):
    validation = np.repeat(np.asarray(validation_mask, bool), targets.shape[1])
    y = targets.reshape(-1).astype(np.float64)
    h = hidden.astype(np.float64)
    best = None
    for ridge in ridges:
        predictor = fit_predictor_at_ridge(hidden, targets, train_mask, ridge)
        prediction = ((h[validation] - predictor["mean_h"]) @ predictor["beta"] +
                      predictor["mean_y"])
        mse = float(np.mean((prediction - y[validation]) ** 2))
        candidate = (mse, predictor)
        if best is None or candidate[0] < best[0]:
            best = candidate
    best[1]["validation_mse"] = float(best[0])
    return best[1]


def predict_utility(hidden, predictor, image_count, token_count):
    result = (hidden - predictor["mean_h"]) @ predictor["beta"] + predictor["mean_y"]
    return result.reshape(image_count, token_count).astype(np.float32)


def aggregate_tokens(tokens, utility=None, temperature=None, remove="none", remove_count=8):
    tokens = unit(tokens)
    if utility is None:
        weights = np.full(tokens.shape[:2], 1.0 / tokens.shape[1], dtype=np.float32)
    else:
        scores = np.asarray(utility, dtype=np.float32).copy()
        if remove != "none":
            order = np.argsort(scores, axis=1)
            selected = order[:, -remove_count:] if remove == "top" else order[:, :remove_count]
            np.put_along_axis(scores, selected, -np.inf, axis=1)
        scaled = scores / float(temperature)
        scaled -= np.max(scaled, axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    aggregate = unit(np.einsum("nt,ntd->nd", weights, tokens, optimize=True))
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=1) / np.log(tokens.shape[1])
    return aggregate, weights.astype(np.float32), entropy.astype(np.float32)


def covariance_basis(features, vehicle_ids):
    features = unit(features).astype(np.float64)
    ids = np.asarray(vehicle_ids)
    center = features.mean(axis=0)
    residual = features.copy()
    for vehicle_id in np.unique(ids):
        residual[ids == vehicle_id] -= features[ids == vehicle_id].mean(axis=0)
    covariance = residual.T @ residual / max(1, len(features) - len(np.unique(ids)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.trace(covariance) / covariance.shape[0])
    return center.astype(np.float32), eigenvalues, eigenvectors, scale


def covariance_transform(basis, ridge):
    center, eigenvalues, eigenvectors, scale = basis
    transform = (eigenvectors / np.sqrt(np.maximum(eigenvalues, 0) + ridge * scale)) @ eigenvectors.T
    return center, transform.astype(np.float32)


def apply_metric(features, center, transform):
    return unit((unit(features) - center) @ transform)


def split_arrays(embedding, rows, splits, split):
    query = read_rows(splits / f"{split}_query.jsonl")
    gallery = read_rows(splits / f"{split}_gallery.jsonl")
    return query, gallery, load_scores(embedding, rows, query, gallery)


def evaluate_calibration(embedding, rows, splits):
    _, _, arrays = split_arrays(embedding, rows, splits, "calibration")
    threshold = calibrate(*arrays)["threshold"]
    return {
        "retrieval": retrieval_metrics(*arrays),
        "candidates": candidate_metrics(*arrays, threshold),
    }


def fuse(global_branch, support_branch, alpha):
    if alpha == 1.0:
        return unit(support_branch)
    return unit(np.concatenate([
        global_branch * np.sqrt(1.0 - alpha), support_branch * np.sqrt(alpha)
    ], axis=1))


def prepare_support(aggregate, global_features, projection, orthogonal):
    aggregate = unit(aggregate)
    if orthogonal:
        global_features = unit(global_features)
        aggregate = unit(aggregate - np.sum(aggregate * global_features, axis=1, keepdims=True) * global_features)
    return unit(aggregate @ projection)


def identity_split(vehicle_ids, seed):
    unique = sorted(set(map(str, vehicle_ids)))
    validation_ids = {v for v in unique if int(hashlib.sha256(f"{seed}:{v}".encode()).hexdigest(), 16) % 5 == 0}
    if not validation_ids or len(validation_ids) == len(unique):
        validation_ids = set(unique[::5])
    values = np.asarray(list(map(str, vehicle_ids)))
    return ~np.isin(values, list(validation_ids)), np.isin(values, list(validation_ids))


def per_query_ap(arrays):
    scores, valid, positive, known = arrays
    result = []
    for i in np.flatnonzero(known):
        eligible = np.flatnonzero(valid[i])
        order = eligible[np.argsort(-scores[i, eligible], kind="stable")]
        ranks = np.flatnonzero(positive[i, order]) + 1
        result.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
    return np.asarray(result, dtype=np.float64)


def fit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    train_rows = read_rows(args.training_features / "rows.jsonl")
    train_meta = {r["image_id"]: r for r in read_rows(args.training_rows)}
    if set(r["image_id"] for r in train_rows) != set(train_meta):
        raise ValueError("Training feature rows differ from training metadata")
    train_ids = np.asarray([str(train_meta[r["image_id"]]["vehicle_id"]) for r in train_rows])
    train_cameras = np.asarray([str(train_meta[r["image_id"]]["camera_id"]) for r in train_rows])
    held_rows = read_rows(args.held_features / "rows.jsonl")
    held_ids = {str(r["vehicle_id"]) for split in ("calibration", "validation")
                for kind in ("query", "gallery") for r in read_rows(args.splits / f"{split}_{kind}.jsonl")}
    if held_ids & set(train_ids):
        raise ValueError("Training and held-out identities overlap")

    train_global = unit(np.load(args.training_features / "embeddings.npy"))
    held_global = unit(np.load(args.held_features / "embeddings.npy"))
    train_tokens = unit(np.load(args.training_features / "local_8x8.npy").astype(np.float32))
    held_tokens = unit(np.load(args.held_features / "local_8x8.npy").astype(np.float32))
    if train_tokens.shape[1:] != (64, train_global.shape[1]):
        raise ValueError("Expected 8x8 local token grid with backbone dimension")
    print(json.dumps({"stage": "loaded", "training_rows": len(train_rows),
                      "held_rows": len(held_rows)}), flush=True)

    global_center, global_transform = covariance_transform(covariance_basis(train_global, train_ids), 0.1)
    train_global_metric = apply_metric(train_global, global_center, global_transform)
    held_global_metric = apply_metric(held_global, global_center, global_transform)
    positive_index, negative_index = choose_contrastive_sets(
        train_global_metric, train_ids, train_cameras, args.positive_k, args.negative_k)
    print(json.dumps({"stage": "pairs_selected", "pairs": len(positive_index)}), flush=True)

    utility_projection = orthogonal_projection(train_tokens.shape[2], args.utility_dim, args.seed)
    train_projected = unit(train_tokens @ utility_projection)
    held_projected = unit(held_tokens @ utility_projection)
    targets, positive_match, negative_match = counterfactual_targets(
        train_projected, positive_index, negative_index, args.pair_batch_size)
    print(json.dumps({"stage": "utility_targets", "tokens": int(targets.size)}), flush=True)
    train_inputs = token_inputs(train_projected)
    held_inputs = token_inputs(held_projected)
    rng = np.random.default_rng(args.seed + 1)
    hidden_weights = (rng.standard_normal((train_inputs.shape[2], args.hidden_dim)) /
                      np.sqrt(train_inputs.shape[2])).astype(np.float32)
    hidden_bias = rng.uniform(-0.5, 0.5, args.hidden_dim).astype(np.float32)
    train_hidden = hidden_features(train_inputs, hidden_weights, hidden_bias)
    held_hidden = hidden_features(held_inputs, hidden_weights, hidden_bias)
    train_mask, validation_mask = identity_split(train_ids, args.seed)
    cv_predictor = fit_ridge_predictor(train_hidden, targets, train_mask, validation_mask, args.predictor_ridges)
    cv_utility = predict_utility(train_hidden, cv_predictor, len(train_tokens), 64)
    validation_target = targets[validation_mask].reshape(-1)
    validation_prediction = cv_utility[validation_mask].reshape(-1)
    correlation = float(np.corrcoef(validation_target, validation_prediction)[0, 1])
    predictor = fit_predictor_at_ridge(train_hidden, targets, np.ones(len(train_tokens), bool), cv_predictor["ridge"])
    predictor["validation_mse"] = cv_predictor["validation_mse"]
    train_utility = predict_utility(train_hidden, predictor, len(train_tokens), 64)
    held_utility = predict_utility(held_hidden, predictor, len(held_tokens), 64)
    print(json.dumps({"stage": "utility_predictor", "ridge": predictor["ridge"],
                      "heldout_pearson": correlation}), flush=True)
    support_projection = (np.eye(train_tokens.shape[2], dtype=np.float32) if args.support_dim == train_tokens.shape[2]
                          else orthogonal_projection(train_tokens.shape[2], args.support_dim, args.seed + 2))
    candidate_results = {}
    models = {}

    def consider(name, embedding, kind, parameters):
        result = evaluate_calibration(embedding, held_rows, args.splits)
        candidate_results[name] = {"kind": kind, "parameters": parameters, **result}
        return result["retrieval"]["mAP"]

    baseline_map = consider("global_r0.1", held_global_metric, "global_control", {"ridge": 0.1})

    uniform_train, _, _ = aggregate_tokens(train_tokens)
    uniform_held, _, uniform_entropy = aggregate_tokens(held_tokens)
    uniform_train = prepare_support(uniform_train, train_global, support_projection, args.orthogonal_support)
    uniform_held = prepare_support(uniform_held, held_global, support_projection, args.orthogonal_support)
    uniform_basis = covariance_basis(uniform_train, train_ids)
    best_uniform = None
    for ridge in args.support_ridges:
        center, transform = covariance_transform(uniform_basis, ridge)
        branch = apply_metric(uniform_held, center, transform)
        for alpha in args.fusion_alphas:
            embedding = fuse(held_global_metric, branch, alpha)
            name = f"uniform_r{ridge:g}_a{alpha:g}"
            score = consider(name, embedding, "uniform_spatial_control", {"ridge": ridge, "alpha": alpha})
            item = (score, name, embedding, center, transform, ridge, alpha)
            if best_uniform is None or item[0] > best_uniform[0]:
                best_uniform = item

    best_evidence = None
    evidence_cache = {}
    for temperature in args.temperatures:
        train_support, _, train_entropy = aggregate_tokens(train_tokens, train_utility, temperature)
        held_support, held_weights, held_entropy = aggregate_tokens(held_tokens, held_utility, temperature)
        train_support = prepare_support(train_support, train_global, support_projection, args.orthogonal_support)
        held_support = prepare_support(held_support, held_global, support_projection, args.orthogonal_support)
        basis = covariance_basis(train_support, train_ids)
        evidence_cache[temperature] = (held_weights, held_entropy, train_entropy)
        for ridge in args.support_ridges:
            center, transform = covariance_transform(basis, ridge)
            branch = apply_metric(held_support, center, transform)
            for alpha in args.fusion_alphas:
                embedding = fuse(held_global_metric, branch, alpha)
                name = f"evidence_t{temperature:g}_r{ridge:g}_a{alpha:g}"
                score = consider(name, embedding, "counterfactual_evidence", {
                    "temperature": temperature, "ridge": ridge, "alpha": alpha})
                item = (score, name, embedding, center, transform, temperature, ridge, alpha, branch)
                if best_evidence is None or item[0] > best_evidence[0]:
                    best_evidence = item
        print(json.dumps({"stage": "temperature_complete", "temperature": temperature,
                          "best_calibration_mAP": best_evidence[0]}), flush=True)

    _, evidence_name, evidence_embedding, support_center, support_transform, temperature, support_ridge, alpha, _ = best_evidence
    _, uniform_name, uniform_embedding, uniform_center, uniform_transform, uniform_ridge, uniform_alpha = best_uniform
    evidence_calibration = candidate_results[evidence_name]
    uniform_calibration = candidate_results[uniform_name]

    # Feature-space intervention: remove high-utility versus low-utility tokens.
    ablations = {}
    for removal in ("top", "bottom"):
        aggregate, _, _ = aggregate_tokens(held_tokens, held_utility, temperature, remove=removal)
        prepared = prepare_support(aggregate, held_global, support_projection, args.orthogonal_support)
        branch = apply_metric(prepared, support_center, support_transform)
        embedding = fuse(held_global_metric, branch, alpha)
        ablations[removal] = evaluate_calibration(embedding, held_rows, args.splits)["retrieval"]

    selected_weights, selected_entropy, train_entropy = evidence_cache[temperature]
    calibration_ids = {r["image_id"] for r in read_rows(args.splits / "calibration_query.jsonl")}
    examples = []
    for i in sorted((i for i, r in enumerate(held_rows) if r["image_id"] in calibration_ids),
                    key=lambda i: hashlib.sha256(held_rows[i]["image_id"].encode()).hexdigest())[:20]:
        order = np.argsort(-selected_weights[i])[:8]
        examples.append({"image_id": held_rows[i]["image_id"], "top_tokens": order.tolist(),
                         "top_weights": selected_weights[i, order].astype(float).tolist(),
                         "normalized_entropy": float(selected_entropy[i])})

    np.save(args.output / "held_embeddings_global.npy", held_global_metric.astype(np.float32), allow_pickle=False)
    np.save(args.output / "held_embeddings_evidence.npy", evidence_embedding.astype(np.float32), allow_pickle=False)
    np.save(args.output / "held_embeddings_uniform.npy", uniform_embedding.astype(np.float32), allow_pickle=False)
    (args.output / "held_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held_rows))
    np.savez(args.output / "model.npz", utility_projection=utility_projection,
             hidden_weights=hidden_weights, hidden_bias=hidden_bias,
             predictor_mean_h=predictor["mean_h"], predictor_mean_y=np.array(predictor["mean_y"], np.float32),
             predictor_beta=predictor["beta"], support_projection=support_projection,
             global_center=global_center, global_transform=global_transform,
             support_center=support_center, support_transform=support_transform,
             uniform_center=uniform_center, uniform_transform=uniform_transform,
             temperature=np.array(temperature, np.float32), alpha=np.array(alpha, np.float32),
             uniform_alpha=np.array(uniform_alpha, np.float32),
             orthogonal_support=np.array(int(args.orthogonal_support), np.int8))

    frozen = {
        "evidence": "VERIFIED-LOCAL calibration-only freeze; V4 unopened",
        "seed": args.seed, "new_method": evidence_name, "uniform_control": uniform_name,
        "global_control": "global_r0.1", "temperature": temperature, "support_ridge": support_ridge,
        "alpha": alpha, "uniform_ridge": uniform_ridge, "uniform_alpha": uniform_alpha,
        "thresholds": {
            "global": candidate_results["global_r0.1"]["candidates"]["threshold"],
            "evidence": evidence_calibration["candidates"]["threshold"],
            "uniform": uniform_calibration["candidates"]["threshold"],
        },
        "selection_rule": "best V2 calibration mAP within each family; V4 decides keep/reject",
        "v4_inspected": False,
    }
    (args.output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    report = {
        "evidence": "VERIFIED-LOCAL CPU experiment; not organizer score",
        "method": "contrastive positive-persistence minus nearest-impostor token utility predictor",
        "training_rows": len(train_rows), "training_ids": len(set(train_ids)), "held_rows": len(held_rows),
        "hard_pairs": {"cross_camera_positive": True, "different_id_negative": True,
                       "positive_k": args.positive_k, "negative_k": args.negative_k,
                       "positive_mean": float(positive_match.mean()), "negative_mean": float(negative_match.mean()),
                       "target_mean": float(targets.mean()), "target_positive_fraction": float((targets > 0).mean())},
        "predictor": {"hidden_dim": args.hidden_dim, "utility_dim": args.utility_dim,
                      "identity_heldout_ids": int(len(set(train_ids[validation_mask]))),
                      "ridge": predictor["ridge"], "MSE": predictor["validation_mse"],
                      "pearson": correlation},
        "calibration": {"global": candidate_results["global_r0.1"],
                        "uniform": uniform_calibration, "evidence": evidence_calibration,
                        "evidence_minus_global_mAP": evidence_calibration["retrieval"]["mAP"] - baseline_map,
                        "evidence_minus_uniform_mAP": evidence_calibration["retrieval"]["mAP"] - uniform_calibration["retrieval"]["mAP"]},
        "feature_ablation": {"selected_mAP": evidence_calibration["retrieval"]["mAP"],
                             "remove_top8": ablations["top"], "remove_bottom8": ablations["bottom"],
                             "interpretation": "Feature-token intervention only; not pixel-level causal proof"},
        "entropy": {"train_mean": float(train_entropy.mean()), "held_mean": float(selected_entropy.mean()),
                    "uniform_held_mean": float(uniform_entropy.mean())},
        "examples": examples,
        "limitations": ["automatic plate masks remain provisional", "8x8 tokens are contextual regions, not named semantic parts",
                        "utility predictor can learn dataset-specific texture or background", "V4 is required for promotion"],
    }
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frozen": frozen, "calibration": report["calibration"],
                      "predictor": report["predictor"]}, indent=2), flush=True)


def apply_frozen(features, model_path):
    model = np.load(model_path, allow_pickle=False)
    global_features = unit(np.load(features / "embeddings.npy"))
    tokens = unit(np.load(features / "local_8x8.npy").astype(np.float32))
    projected = unit(tokens @ model["utility_projection"])
    hidden = hidden_features(token_inputs(projected), model["hidden_weights"], model["hidden_bias"])
    predictor = {"mean_h": model["predictor_mean_h"], "mean_y": float(model["predictor_mean_y"]),
                 "beta": model["predictor_beta"]}
    utility = predict_utility(hidden, predictor, len(tokens), tokens.shape[1])
    global_branch = apply_metric(global_features, model["global_center"], model["global_transform"])
    support, weights, entropy = aggregate_tokens(tokens, utility, float(model["temperature"]))
    orthogonal = bool(int(model["orthogonal_support"])) if "orthogonal_support" in model.files else False
    support = prepare_support(support, global_features, model["support_projection"], orthogonal)
    support_branch = apply_metric(support, model["support_center"], model["support_transform"])
    evidence = fuse(global_branch, support_branch, float(model["alpha"]))
    uniform, _, _ = aggregate_tokens(tokens)
    uniform = prepare_support(uniform, global_features, model["support_projection"], orthogonal)
    uniform_branch = apply_metric(uniform, model["uniform_center"], model["uniform_transform"])
    uniform_embedding = fuse(global_branch, uniform_branch, float(model["uniform_alpha"]))
    return global_branch, evidence, uniform_embedding, weights, entropy


def gate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = read_rows(args.features / "rows.jsonl")
    global_branch, evidence, uniform, weights, entropy = apply_frozen(args.features, args.model)
    frozen = json.loads(args.selection.read_text())
    results = {}
    arrays_by_name = {}
    for name, embedding in (("global", global_branch), ("uniform", uniform), ("evidence", evidence)):
        query, gallery, arrays = split_arrays(embedding, rows, args.splits, "validation")
        arrays_by_name[name] = arrays
        results[name] = {"retrieval": retrieval_metrics(*arrays),
                         "candidates": candidate_metrics(*arrays, frozen["thresholds"][name])}
        np.save(args.output / f"embeddings_{name}.npy", embedding.astype(np.float32), allow_pickle=False)
    baseline_ap = per_query_ap(arrays_by_name["global"])
    evidence_ap = per_query_ap(arrays_by_name["evidence"])
    if len(baseline_ap) != len(evidence_ap):
        raise ValueError("Gate known-query counts differ")
    rng = np.random.default_rng(args.bootstrap_seed)
    sample = rng.integers(0, len(baseline_ap), size=(args.bootstrap_samples, len(baseline_ap)))
    deltas = (evidence_ap[sample] - baseline_ap[sample]).mean(axis=1)
    interval = np.quantile(deltas, [0.025, 0.975])
    delta = float(evidence_ap.mean() - baseline_ap.mean())
    promote = bool(interval[0] > 0 and
                   results["evidence"]["candidates"]["pair_F1"] >= results["global"]["candidates"]["pair_F1"] - 0.01 and
                   results["evidence"]["candidates"]["unknown_query_TNR"] >= results["global"]["candidates"]["unknown_query_TNR"] - 0.05)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(args.output / "evidence_weights.npy", weights.astype(np.float16), allow_pickle=False)
    gate_name = frozen.get("gate_name", "independent identity gate")
    report = {
        "evidence": f"VERIFIED-LOCAL untouched {gate_name} CPU gate; not organizer score",
        "frozen_selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "results": results,
        "paired_bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                             "delta_mAP": delta, "CI95": interval.astype(float).tolist()},
        "entropy": {"mean": float(entropy.mean()), "p10": float(np.quantile(entropy, 0.1)),
                    "p90": float(np.quantile(entropy, 0.9))},
        "promotion_gate": "CI lower bound > 0; pair F1 loss <=0.01; TNR loss <=0.05",
        "decision": "promote" if promote else "reject",
        "limitations": ["one untouched identity reserve", "provisional automatic plate masks",
                        "local evaluator may differ from official evaluator"],
    }
    (args.output / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--training-features", type=Path, required=True)
    f.add_argument("--training-rows", type=Path, required=True)
    f.add_argument("--held-features", type=Path, required=True)
    f.add_argument("--splits", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--seed", type=int, default=20260924)
    f.add_argument("--utility-dim", type=int, default=64)
    f.add_argument("--support-dim", type=int, default=192)
    f.add_argument("--hidden-dim", type=int, default=128)
    f.add_argument("--pair-batch-size", type=int, default=64)
    f.add_argument("--positive-k", type=int, default=1)
    f.add_argument("--negative-k", type=int, default=1)
    f.add_argument("--orthogonal-support", action="store_true")
    f.add_argument("--predictor-ridges", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    f.add_argument("--support-ridges", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    f.add_argument("--temperatures", type=float, nargs="+", default=[0.03, 0.07, 0.15, 0.3])
    f.add_argument("--fusion-alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    g = sub.add_parser("gate")
    g.add_argument("--features", type=Path, required=True)
    g.add_argument("--splits", type=Path, required=True)
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--selection", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--bootstrap-samples", type=int, default=5000)
    g.add_argument("--bootstrap-seed", type=int, default=20260924)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    fit(arguments) if arguments.command == "fit" else gate(arguments)
