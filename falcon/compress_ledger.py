"""Compress the promoted CHEL fusion while preserving its cosine geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import calibrate, candidate_metrics, load_scores, read_rows, retrieval_metrics
from .evidence_ledger import apply_frozen, per_query_ap, unit
from .ledger_fusion import fused


GLOBAL_WEIGHT = 0.75
V1_SUPPORT_WEIGHT = 0.1875
V2_SUPPORT_WEIGHT = 0.0625


def decompose(global_embedding, v1_embedding, v2_embedding):
    global_embedding = unit(global_embedding)
    dimension = global_embedding.shape[1]
    if v1_embedding.shape[1] <= dimension or v2_embedding.shape[1] <= dimension:
        raise ValueError("CHEL embeddings do not contain support branches")
    if np.max(np.abs(unit(v1_embedding[:, :dimension]) - global_embedding)) > 1e-5:
        raise ValueError("CHEL-v1 global branch differs from control")
    if np.max(np.abs(unit(v2_embedding[:, :dimension]) - global_embedding)) > 1e-5:
        raise ValueError("CHEL-v2 global branch differs from control")
    return global_embedding, unit(v1_embedding[:, dimension:]), unit(v2_embedding[:, dimension:])


def exact_compact(global_embedding, support_v1, support_v2):
    return unit(np.concatenate([
        global_embedding * np.sqrt(GLOBAL_WEIGHT),
        support_v1 * np.sqrt(V1_SUPPORT_WEIGHT),
        support_v2 * np.sqrt(V2_SUPPORT_WEIGHT),
    ], axis=1))


def support_stack(support_v1, support_v2):
    """Unit support block whose final mixture weight is 0.25."""
    return unit(np.concatenate([
        support_v1 * np.sqrt(V1_SUPPORT_WEIGHT / (1 - GLOBAL_WEIGHT)),
        support_v2 * np.sqrt(V2_SUPPORT_WEIGHT / (1 - GLOBAL_WEIGHT)),
    ], axis=1))


def validate_exact_scores(teacher, compact, rows, splits, split="calibration", tolerance=2e-6):
    query = read_rows(splits / f"{split}_query.jsonl")
    gallery = read_rows(splits / f"{split}_gallery.jsonl")
    teacher_scores = load_scores(teacher, rows, query, gallery)[0]
    compact_scores = load_scores(compact, rows, query, gallery)[0]
    delta = float(np.max(np.abs(teacher_scores - compact_scores)))
    if delta > tolerance:
        raise ValueError(f"Exact compact score mismatch: {delta} > {tolerance}")
    return delta


def pca_components(features, centered, maximum_dimension):
    mean = features.mean(axis=0) if centered else np.zeros(features.shape[1], dtype=np.float32)
    _, singular, vectors = np.linalg.svd((features - mean).astype(np.float32), full_matrices=False)
    return mean.astype(np.float32), vectors[:maximum_dimension].T.astype(np.float32), singular.astype(np.float32)


def project_support(features, mean, components):
    projected = (features - mean) @ components
    norms = np.linalg.norm(projected, axis=1)
    if not np.isfinite(projected).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid compressed support")
    return unit(projected)


def compressed_embedding(global_embedding, support_v1, support_v2, model):
    if "joint_components" in model:
        support = project_support(support_stack(support_v1, support_v2),
                                  model["joint_mean"], model["joint_components"])
        return unit(np.concatenate([
            global_embedding * np.sqrt(GLOBAL_WEIGHT),
            support * np.sqrt(1 - GLOBAL_WEIGHT),
        ], axis=1))
    first = project_support(support_v1, model["v1_mean"], model["v1_components"])
    second = project_support(support_v2, model["v2_mean"], model["v2_components"])
    return exact_compact(global_embedding, first, second)


def arrays(embedding, rows, splits, split):
    query = read_rows(splits / f"{split}_query.jsonl")
    gallery = read_rows(splits / f"{split}_gallery.jsonl")
    return load_scores(embedding, rows, query, gallery)


def metrics(embedding, rows, splits, split):
    value = arrays(embedding, rows, splits, split)
    return retrieval_metrics(*value), value


def repeated_score_delta(embedding, rows, splits, split):
    """Measure same-process evaluator determinism instead of assuming it."""
    first = arrays(embedding, rows, splits, split)[0]
    second = arrays(embedding, rows, splits, split)[0]
    return float(np.max(np.abs(first - second)))


def promotion_decision(dimension, teacher_gain, student_gain, retained,
                       teacher_map, student_map, student_ci_lower,
                       global_f1, student_f1, global_tnr, student_tnr):
    checks = {
        "dimension_at_most_1024": dimension <= 1024,
        "teacher_gain_positive": teacher_gain > 0,
        "student_gain_positive": student_gain > 0,
        "retains_at_least_90pct_teacher_gain": retained is not None and retained >= .9,
        "student_within_0.2pp_teacher": student_map >= teacher_map - .002,
        "student_vs_global_ci_lower_positive": student_ci_lower > 0,
        "pair_f1_loss_at_most_0.01": student_f1 >= global_f1 - .01,
        "unknown_tnr_loss_at_most_0.05": student_tnr >= global_tnr - .05,
    }
    return ("promote" if all(checks.values()) else "reject"), checks


def branches(features, v1_model, v2_model):
    global_v1, evidence_v1, _, _, _ = apply_frozen(features, v1_model)
    global_v2, evidence_v2, _, _, _ = apply_frozen(features, v2_model)
    if np.max(np.abs(global_v1 - global_v2)) != 0:
        raise ValueError("Frozen global branches are not bit-identical")
    return decompose(global_v1, evidence_v1, evidence_v2), evidence_v1, evidence_v2


def fit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (train_global, train_v1, train_v2), _, _ = branches(args.training_features, args.v1_model, args.v2_model)
    (held_global, held_v1, held_v2), held_e1, held_e2 = branches(args.held_features, args.v1_model, args.v2_model)
    train_rows = read_rows(args.training_features / "rows.jsonl")
    held_rows = read_rows(args.held_features / "rows.jsonl")
    teacher = fused(held_e1, held_e2, .75)
    compact = exact_compact(held_global, held_v1, held_v2)
    exact_delta = validate_exact_scores(teacher, compact, held_rows, args.splits)
    global_metrics, _ = metrics(held_global, held_rows, args.splits, "calibration")
    teacher_metrics, teacher_arrays = metrics(teacher, held_rows, args.splits, "calibration")
    teacher_repeat_delta = repeated_score_delta(teacher, held_rows, args.splits, "calibration")
    if teacher_repeat_delta > 1e-8:
        raise ValueError(f"Same-process teacher score noise: {teacher_repeat_delta}")
    teacher_confidence = calibrate(*teacher_arrays)
    teacher_gain = teacher_metrics["mAP"] - global_metrics["mAP"]

    bases = {}
    train_joint = support_stack(train_v1, train_v2)
    for centered in (False, True):
        key = "centered" if centered else "uncentered"
        v1_mean, v1_vectors, v1_singular = pca_components(train_v1, centered, 192)
        v2_mean, v2_vectors, v2_singular = pca_components(train_v2, centered, 192)
        bases[key] = (v1_mean, v1_vectors, v1_singular, v2_mean, v2_vectors, v2_singular)
        print(json.dumps({"stage": "pca", "mode": key}), flush=True)

    candidates = []
    dimension_pairs = ((64, 192), (96, 160), (128, 128), (160, 96), (192, 64))
    for mode, basis in bases.items():
        v1_mean, v1_vectors, _, v2_mean, v2_vectors, _ = basis
        for v1_dimension, v2_dimension in dimension_pairs:
            model = {"v1_mean": v1_mean, "v1_components": v1_vectors[:, :v1_dimension],
                     "v2_mean": v2_mean, "v2_components": v2_vectors[:, :v2_dimension]}
            embedding = compressed_embedding(held_global, held_v1, held_v2, model)
            retrieval, score_arrays = metrics(embedding, held_rows, args.splits, "calibration")
            confidence = calibrate(*score_arrays)
            gain = retrieval["mAP"] - global_metrics["mAP"]
            candidates.append({"family": "separate_pca", "mode": mode, "v1_dimension": v1_dimension,
                               "v2_dimension": v2_dimension, "dimension": embedding.shape[1],
                               "mAP": retrieval["mAP"], "gain": gain,
                               "retained_teacher_gain": gain / teacher_gain if teacher_gain > 0 else None,
                               "candidates": confidence})
    joint_bases = {}
    for centered in (False, True):
        mode = "centered" if centered else "uncentered"
        joint_mean, joint_vectors, joint_singular = pca_components(train_joint, centered, 256)
        joint_bases[mode] = (joint_mean, joint_vectors, joint_singular)
        model = {"joint_mean": joint_mean, "joint_components": joint_vectors}
        embedding = compressed_embedding(held_global, held_v1, held_v2, model)
        retrieval, score_arrays = metrics(embedding, held_rows, args.splits, "calibration")
        confidence = calibrate(*score_arrays); gain = retrieval["mAP"] - global_metrics["mAP"]
        candidates.append({"family": "joint_pca", "mode": mode,
                           "v1_dimension": None, "v2_dimension": None,
                           "dimension": embedding.shape[1], "mAP": retrieval["mAP"], "gain": gain,
                           "retained_teacher_gain": gain / teacher_gain if teacher_gain > 0 else None,
                           "candidates": confidence})

    for projection_seed in range(args.seed, args.seed + 5):
        rng = np.random.default_rng(projection_seed)
        values = rng.choice(np.array([-np.sqrt(3), 0, np.sqrt(3)], dtype=np.float32),
                            size=(train_joint.shape[1], 256), p=(1/6, 2/3, 1/6)) / np.sqrt(256)
        model = {"joint_mean": np.zeros(train_joint.shape[1], dtype=np.float32),
                 "joint_components": values.astype(np.float32)}
        embedding = compressed_embedding(held_global, held_v1, held_v2, model)
        retrieval, score_arrays = metrics(embedding, held_rows, args.splits, "calibration")
        confidence = calibrate(*score_arrays); gain = retrieval["mAP"] - global_metrics["mAP"]
        candidates.append({"family": "support_sparse_jl", "mode": "uncentered",
                           "projection_seed": projection_seed,
                           "v1_dimension": None, "v2_dimension": None,
                           "dimension": embedding.shape[1], "mAP": retrieval["mAP"], "gain": gain,
                           "retained_teacher_gain": gain / teacher_gain if teacher_gain > 0 else None,
                           "candidates": confidence})
    eligible = [c for c in candidates if c["dimension"] <= 1024 and
                c["retained_teacher_gain"] is not None and c["retained_teacher_gain"] >= .9]
    if not eligible:
        winner = max(candidates, key=lambda c: c["mAP"])
        decision = "reject"
    else:
        winner = max(eligible, key=lambda c: (c["mAP"], c["candidates"]["pair_F1"],
                                               c["candidates"]["unknown_query_TNR"]))
        decision = "keep"
    if winner["family"] == "separate_pca":
        basis = bases[winner["mode"]]
        model = {"v1_mean": basis[0], "v1_components": basis[1][:, :winner["v1_dimension"]],
                 "v2_mean": basis[3], "v2_components": basis[4][:, :winner["v2_dimension"]]}
    elif winner["family"] == "joint_pca":
        basis = joint_bases[winner["mode"]]
        model = {"joint_mean": basis[0], "joint_components": basis[1]}
    else:
        rng = np.random.default_rng(winner["projection_seed"])
        values = rng.choice(np.array([-np.sqrt(3), 0, np.sqrt(3)], dtype=np.float32),
                            size=(train_joint.shape[1], 256), p=(1/6, 2/3, 1/6)) / np.sqrt(256)
        model = {"joint_mean": np.zeros(train_joint.shape[1], dtype=np.float32),
                 "joint_components": values.astype(np.float32)}
    np.savez(args.output / "model.npz", **model)
    selection = {
        "evidence": "VERIFIED-LOCAL training-only PCA, V2-calibration selection; V7 unopened",
        "seed": args.seed, "winner": winner, "decision": decision,
        "teacher": {"dimension": teacher.shape[1], "mAP": teacher_metrics["mAP"],
                    "gain": teacher_gain, "candidate_threshold": teacher_confidence["threshold"]},
        "global": {"dimension": held_global.shape[1], "mAP": global_metrics["mAP"],
                   "candidate_threshold": args.global_threshold},
        "gate": {"maximum_dimension": 1024, "minimum_retained_teacher_gain": .9,
                 "V7_inspected": False},
        "exact_compact": {"dimension": compact.shape[1], "max_score_delta": exact_delta},
        "cheap_criterion_checks": {
            "teacher_fails_dimension": bool(teacher.shape[1] > 1024),
            "global_fails_retained_gain": bool(teacher_gain > 0),
        },
        "model_sha256": None,
        "candidates": candidates,
    }
    selection["model_sha256"] = hashlib.sha256((args.output / "model.npz").read_bytes()).hexdigest()
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    winner_embedding = compressed_embedding(held_global, held_v1, held_v2, model)
    np.save(args.output / "held_embedding.npy", winner_embedding.astype(np.float32), allow_pickle=False)
    (args.output / "held_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held_rows))
    report = {"evidence": "VERIFIED-LOCAL CPU compression fit", "selection": selection,
              "training_rows": len(train_rows), "held_rows": len(held_rows),
              "teacher_repeated_score_delta": teacher_repeat_delta,
              "limitations": ["PCA selected on the reused V2 calibration split",
                              "V7 required", "plate masks provisional"]}
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": decision, "winner": winner, "teacher": selection["teacher"],
                      "exact_compact": selection["exact_compact"]}, indent=2), flush=True)


def gate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    selection = json.loads(args.selection.read_text())
    if selection["decision"] != "keep":
        raise ValueError("Compression was not admitted by calibration")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != selection["model_sha256"]:
        raise ValueError("Compression model hash differs from freeze")
    (global_embedding, support_v1, support_v2), evidence_v1, evidence_v2 = branches(
        args.features, args.v1_model, args.v2_model)
    rows = read_rows(args.features / "rows.jsonl")
    teacher = fused(evidence_v1, evidence_v2, .75)
    compact = exact_compact(global_embedding, support_v1, support_v2)
    exact_delta = validate_exact_scores(teacher, compact, rows, args.splits, split="validation")
    model_file = np.load(args.model, allow_pickle=False)
    model = {key: model_file[key] for key in model_file.files}
    student = compressed_embedding(global_embedding, support_v1, support_v2, model)
    global_metrics, global_arrays = metrics(global_embedding, rows, args.splits, "validation")
    teacher_metrics, teacher_arrays = metrics(teacher, rows, args.splits, "validation")
    student_metrics, student_arrays = metrics(student, rows, args.splits, "validation")
    global_ap = per_query_ap(global_arrays); teacher_ap = per_query_ap(teacher_arrays); student_ap = per_query_ap(student_arrays)
    rng = np.random.default_rng(args.seed)
    index = rng.integers(0, len(global_ap), (args.samples, len(global_ap)))
    teacher_delta = teacher_ap - global_ap
    student_delta = student_ap - global_ap
    student_teacher = student_ap - teacher_ap
    teacher_gain = float(teacher_delta.mean()); student_gain = float(student_delta.mean())
    retained = student_gain / teacher_gain if teacher_gain > 0 else None
    thresholds = {"global": selection["global"]["candidate_threshold"],
                  "teacher": selection["teacher"]["candidate_threshold"],
                  "student": selection["winner"]["candidates"]["threshold"]}
    candidate_result = {
        "global": candidate_metrics(*global_arrays, thresholds["global"]),
        "teacher": candidate_metrics(*teacher_arrays, thresholds["teacher"]),
        "student": candidate_metrics(*student_arrays, thresholds["student"]),
    }
    dimension = student.shape[1]
    student_ci_lower = float(np.quantile(student_delta[index].mean(1), .025))
    decision, gate_checks = promotion_decision(
        dimension, teacher_gain, student_gain, retained,
        teacher_metrics["mAP"], student_metrics["mAP"], student_ci_lower,
        candidate_result["global"]["pair_F1"], candidate_result["student"]["pair_F1"],
        candidate_result["global"]["unknown_query_TNR"],
        candidate_result["student"]["unknown_query_TNR"])
    teacher_repeat_delta = repeated_score_delta(teacher, rows, args.splits, "validation")
    if teacher_repeat_delta > 1e-8:
        raise ValueError(f"Same-process teacher score noise: {teacher_repeat_delta}")
    report = {
        "evidence": args.label, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "dimension": dimension, "global": global_metrics, "teacher": teacher_metrics, "student": student_metrics,
        "candidates": candidate_result,
        "paired": {
            "teacher_minus_global": {"delta_mAP": teacher_gain,
                "CI95": np.quantile(teacher_delta[index].mean(1), [.025, .975]).tolist()},
            "student_minus_global": {"delta_mAP": student_gain,
                "CI95": np.quantile(student_delta[index].mean(1), [.025, .975]).tolist()},
            "student_minus_teacher": {"delta_mAP": float(student_teacher.mean()),
                "CI95": np.quantile(student_teacher[index].mean(1), [.025, .975]).tolist()},
        },
        "retained_teacher_gain": retained,
        "controls": {"exact_compact_max_score_delta": exact_delta,
                     "teacher_repeated_score_delta": teacher_repeat_delta},
        "promotion_gate": "dim<=1024; teacher gain>0; retain>=90%; student within0.2p.p. teacher; student-vs-global CI lower>0; F1/TNR guards",
        "promotion_checks": gate_checks,
        "decision": decision,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.save(args.output / "embedding.npy", student.astype(np.float32), allow_pickle=False)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    for name in ("training-features", "held-features", "v1-model", "v2-model", "splits", "output"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--seed", type=int, default=20260925)
    f.add_argument("--global-threshold", type=float, default=.20663335919380188)
    g = sub.add_parser("gate")
    for name in ("features", "v1-model", "v2-model", "model", "selection", "splits", "output"):
        g.add_argument("--" + name, type=Path, required=True)
    g.add_argument("--seed", type=int, default=20260927); g.add_argument("--samples", type=int, default=5000)
    g.add_argument("--label", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args(); fit(args) if args.command == "fit" else gate(args)
