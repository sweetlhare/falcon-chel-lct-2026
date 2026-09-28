"""Calibration-only fusion of two frozen CHEL embeddings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import calibrate, candidate_metrics, load_scores, read_rows, retrieval_metrics
from .evidence_ledger import per_query_ap, unit


def fused(first, second, alpha):
    first = unit(first); second = unit(second)
    if alpha == 1:
        return first
    if alpha == 0:
        return second
    return unit(np.concatenate([first * np.sqrt(alpha), second * np.sqrt(1 - alpha)], axis=1))


def retrieval(embedding, rows, splits, split="validation"):
    query = read_rows(splits / f"{split}_query.jsonl")
    gallery = read_rows(splits / f"{split}_gallery.jsonl")
    arrays = load_scores(embedding, rows, query, gallery)
    return retrieval_metrics(*arrays), arrays


def fit(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    first = np.load(args.first, allow_pickle=False); second = np.load(args.second, allow_pickle=False)
    rows = read_rows(args.rows)
    results = []
    for alpha in (0, .25, .5, .75, 1):
        embedding = fused(first, second, alpha)
        metrics, _ = retrieval(embedding, rows, args.splits, "calibration")
        results.append({"alpha_first": alpha, "mAP": metrics["mAP"], "dimension": embedding.shape[1]})
    winner = max(results, key=lambda r: (r["mAP"], -r["dimension"]))
    winner_embedding = fused(first, second, winner["alpha_first"])
    _, winner_arrays = retrieval(winner_embedding, rows, args.splits, "calibration")
    confidence = calibrate(*winner_arrays)
    selection = {"evidence": "VERIFIED-LOCAL V2-calibration-only fusion freeze",
                 "alpha_first": winner["alpha_first"], "results": results,
                 "candidate_threshold": confidence["threshold"],
                 "candidate_calibration": confidence,
                 "first_sha256": hashlib.sha256(args.first.read_bytes()).hexdigest(),
                 "second_sha256": hashlib.sha256(args.second.read_bytes()).hexdigest()}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


def gate(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    selection = json.loads(args.selection.read_text())
    first = np.load(args.first, allow_pickle=False); second = np.load(args.second, allow_pickle=False)
    global_embedding = np.load(args.global_embedding, allow_pickle=False)
    rows = read_rows(args.rows)
    embedding = fused(first, second, selection["alpha_first"])
    metrics, arrays = retrieval(embedding, rows, args.splits)
    control, control_arrays = retrieval(global_embedding, rows, args.splits)
    delta = per_query_ap(arrays) - per_query_ap(control_arrays)
    rng = np.random.default_rng(args.seed); index = rng.integers(0, len(delta), (args.samples, len(delta)))
    report = {"evidence": args.label, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              "global": control, "fusion": metrics,
              "candidates": {"global": candidate_metrics(*control_arrays, args.global_threshold),
                             "fusion": candidate_metrics(*arrays, selection["candidate_threshold"])},
              "paired": {"delta_mAP": float(delta.mean()),
                         "CI95": np.quantile(delta[index].mean(1), [.025, .975]).tolist(),
                         "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                         "losses": int((delta < 0).sum()), "samples": args.samples, "seed": args.seed},
              "dimension": embedding.shape[1]}
    report["promotion_gate"] = "CI lower bound > 0; pair F1 loss <=0.01; TNR loss <=0.05"
    report["decision"] = ("promote" if report["paired"]["CI95"][0] > 0 and
                           report["candidates"]["fusion"]["pair_F1"] >= report["candidates"]["global"]["pair_F1"] - .01 and
                           report["candidates"]["fusion"]["unknown_query_TNR"] >= report["candidates"]["global"]["unknown_query_TNR"] - .05
                           else "reject")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.save(args.output / "embedding.npy", embedding.astype(np.float32), allow_pickle=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    for name in ("first", "second", "rows", "splits", "output"):
        f.add_argument("--" + name, type=Path, required=True)
    g = sub.add_parser("gate")
    for name in ("first", "second", "global-embedding", "rows", "splits", "selection", "output"):
        g.add_argument("--" + name, type=Path, required=True)
    g.add_argument("--label", required=True); g.add_argument("--seed", type=int, default=20260924)
    g.add_argument("--samples", type=int, default=5000)
    g.add_argument("--global-threshold", type=float, default=.20663335919380188)
    a = p.parse_args(); fit(a) if a.command == "fit" else gate(a)
