"""Full-gallery cross-camera retrieval and separate candidate/refusal evaluation.

This is a local protocol, not the unpublished organizer evaluator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_scores(embeddings, embedding_rows, query, gallery):
    if len(embedding_rows) != len(embeddings):
        raise ValueError('Embedding rows and matrix disagree')
    ids = [r['image_id'] for r in embedding_rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate embedding image_id')
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embedding matrix')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('Zero embedding')
    embeddings = embeddings / norms
    index = {iid: i for i, iid in enumerate(ids)}
    qi = [index[r['image_id']] for r in query]
    gi = [index[r['image_id']] for r in gallery]
    scores = embeddings[qi] @ embeddings[gi].T
    qid = np.array([r['vehicle_id'] for r in query])
    gid = np.array([r['vehicle_id'] for r in gallery])
    qcam = np.array([r['camera_id'] for r in query])
    gcam = np.array([r['camera_id'] for r in gallery])
    valid = qcam[:, None] != gcam[None, :]
    valid &= np.array([r['image_id'] for r in query])[:, None] != np.array([r['image_id'] for r in gallery])[None, :]
    positive = (qid[:, None] == gid[None, :]) & valid
    actual_known = (qid[:, None] == gid[None, :]).any(axis=1)
    declared_known = np.array([r['is_known'] for r in query])
    if not np.array_equal(actual_known, declared_known):
        raise ValueError('Query known flags disagree with gallery membership')
    return scores, valid, positive, actual_known


def retrieval_metrics(scores, valid, positive, known):
    aps, rank1, rank5, inp, recall_k = [], [], [], [], []
    for i in np.flatnonzero(known):
        eligible = np.flatnonzero(valid[i])
        order = eligible[np.argsort(-scores[i, eligible], kind='stable')]
        hits = positive[i, order]
        locations = np.flatnonzero(hits) + 1
        if not len(locations):
            raise ValueError('Known query lacks a cross-camera positive')
        aps.append(float(np.mean(np.arange(1, len(locations)+1)/locations)))
        rank1.append(bool(hits[:1].any()))
        rank5.append(bool(hits[:5].any()))
        inp.append(float(len(locations)/locations[-1]))
        recall_k.append([bool(hits[:k].any()) for k in (10, 20, 50, 100)])
    if not aps:
        raise ValueError('No known queries')
    return {'known_queries': len(aps), 'mAP': float(np.mean(aps)),
            'rank1': float(np.mean(rank1)), 'rank5': float(np.mean(rank5)),
            'mINP': float(np.mean(inp)),
            'hit_recall_at_k': {str(k): float(np.mean(recall_k, axis=0)[j])
                                for j, k in enumerate((10, 20, 50, 100))}}


def candidate_metrics(scores, valid, positive, known, threshold, top_n=10):
    eligible = np.zeros_like(valid)
    for i in range(len(scores)):
        available = np.flatnonzero(valid[i])
        order = available[np.argsort(-scores[i, available], kind='stable')][:top_n]
        eligible[i, order] = True
    accepted = eligible & (scores >= threshold)
    tp = int((accepted & positive).sum())
    fp = int((accepted & ~positive).sum())
    fn = int((~accepted & positive).sum())
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    unknown = ~known
    return {'pair_precision': precision, 'pair_recall': recall,
            'pair_F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'unknown_query_TNR': float(np.mean(~accepted[unknown].any(axis=1))) if unknown.any() else None,
            'known_query_acceptance': float(np.mean(accepted[known].any(axis=1))),
            'known_query_correct_acceptance': float(np.mean((accepted & positive)[known].any(axis=1))),
            'tp': tp, 'fp': fp, 'fn': fn, 'threshold': float(threshold), 'top_n': top_n}


def calibrate(scores, valid, positive, known):
    values, hits, unknown_max = [], [], []
    for i in range(len(scores)):
        available=np.flatnonzero(valid[i])
        order=available[np.argsort(-scores[i,available],kind='stable')][:10]
        values.extend(scores[i,order]);hits.extend(positive[i,order])
        if not known[i]:unknown_max.append(float(scores[i,order].max()) if len(order) else -np.inf)
    if not values:raise ValueError('No eligible calibration candidates')
    values=np.asarray(values);hits=np.asarray(hits,dtype=np.int64)
    order=np.argsort(-values,kind='stable');values=values[order];hits=hits[order]
    end=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1]
    tp=np.cumsum(hits)[end];accepted=end+1;fp=accepted-tp
    total_positive=int(positive.sum())
    thresholds=np.r_[np.nextafter(values[0],np.inf),values[end]]
    f1=np.r_[0.,2*tp/np.maximum(tp+fp+total_positive,1)]
    tnr=(np.searchsorted(np.sort(unknown_max),thresholds,side='left')/len(unknown_max)
         if unknown_max else np.zeros(len(thresholds)))
    best=max(range(len(thresholds)),key=lambda i:(f1[i],tnr[i],thresholds[i]))
    report=candidate_metrics(scores,valid,positive,known,thresholds[best])
    report['calibration_method']='exact top10 score breakpoints; ties accepted together'
    return report


def evaluate(embedding_path, rows_path, splits, output):
    emb = np.load(embedding_path, allow_pickle=False)
    rows = read_rows(rows_path)
    scores_by_split = {}
    for split in ('calibration', 'validation'):
        q = read_rows(splits / f'{split}_query.jsonl')
        g = read_rows(splits / f'{split}_gallery.jsonl')
        scores_by_split[split] = load_scores(emb, rows, q, g)
    selected = calibrate(*scores_by_split['calibration'])
    result = {'evidence': 'VERIFIED-LOCAL; not organizer-equivalent',
              'protocol': 'full AP on known queries; same-camera excluded; pair F1 top10; query TNR',
              'split_manifest_sha256': hashlib.sha256((splits/'manifest.json').read_bytes()).hexdigest(),
              'embeddings_sha256': hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
              'threshold_source': 'calibration only', 'calibration': {}, 'validation': {}}
    for split, arrays in scores_by_split.items():
        result[split] = {'retrieval': retrieval_metrics(*arrays),
                         'candidates': candidate_metrics(*arrays, selected['threshold'])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.embeddings, args.rows, args.splits, args.output)
