"""Replay the selected training-only KISSME fit, with no validation/test inputs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
from falcon.deadline_metric import pair_covariances, psd_metric
from falcon.evidence_ledger import unit

RIDGE = 0.1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aligned_training(features, feature_rows, metadata, center, shape=(2500, 768), expected_ids=400):
    if features.shape != shape or not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f'Expected floating training matrix {shape}')
    if not np.isfinite(features).all() or (np.linalg.norm(features, axis=1) <= 1e-12).any():
        raise ValueError('Training features must be finite and nonzero')
    if len(feature_rows) != shape[0] or len(metadata) != shape[0]:
        raise ValueError('Feature and metadata row counts differ')
    for rows in (feature_rows, metadata):
        ids = [row['image_id'] for row in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or duplicate image_id')
    by_image = {row['image_id']: row for row in metadata}
    if set(by_image) != {row['image_id'] for row in feature_rows}:
        raise ValueError('Training feature images differ from training metadata')
    aligned = []
    for row in feature_rows:
        meta = by_image[row['image_id']]
        if ('source_row' not in row or 'source_row' not in meta or
                row['source_row'] != meta['source_row']):
            raise ValueError('Source CSV row mismatch for ' + row['image_id'])
        label = meta.get('vehicle_id')
        if isinstance(label, bool) or not isinstance(label, (str, int)) or not str(label):
            raise ValueError('Invalid training vehicle_id')
        if 'vehicle_id' in row and str(row['vehicle_id']) != str(label):
            raise ValueError('Vehicle ID mismatch for ' + row['image_id'])
        aligned.append(str(label))
    labels = np.asarray(aligned)
    if len(np.unique(labels)) != expected_ids:
        raise ValueError(f'Expected exactly {expected_ids} training identities')
    raw = unit(features).astype(np.float64)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (shape[1],) or not np.isfinite(center).all():
        raise ValueError('Invalid frozen global center')
    center_error = float(np.max(np.abs(center - raw.mean(0))))
    if center_error >= 2e-7:
        raise ValueError(f'Frozen center differs from training features: {center_error}')
    return raw, labels, center, center_error


def fit(training_features, training_rows, center_weights, output,
        expected_rows=2500, expected_ids=400, dimension=768):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    feature_path = training_features / 'embeddings.npy'
    row_path = training_features / 'rows.jsonl'
    features = np.load(feature_path, allow_pickle=False)
    with np.load(center_weights, allow_pickle=False) as weights:
        center = weights['global_center']
    raw, labels, center, error = aligned_training(
        features, read_rows(row_path), read_rows(training_rows), center,
        shape=(expected_rows, dimension), expected_ids=expected_ids)
    same, different = pair_covariances(raw, labels)
    transform, spectrum = psd_metric(same, different, RIDGE)
    transform32 = transform.astype(np.float32)
    if not spectrum['rank'] or not np.isfinite(transform32).all():
        raise ValueError('Degenerate or nonfinite selected metric')
    if (np.linalg.norm((raw - center) @ transform32, axis=1) <= 1e-10).any():
        raise ValueError('Metric produces zero training embeddings')
    inputs = {'training_embeddings': feature_path, 'training_feature_rows': row_path,
              'training_metadata': training_rows, 'frozen_center_weights': center_weights,
              'metric_core_source': Path(inspect.getsourcefile(pair_covariances)),
              'normalization_source': Path(inspect.getsourcefile(unit)), 'replay_source': Path(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'global_metric.npz'
    np.savez(target, center=center.astype(np.float32), transform=transform32)
    counts = np.unique(labels, return_counts=True)[1]
    same_count = int(np.sum(counts * (counts - 1)))
    manifest = dict(model='KISSME global metric',ridge=RIDGE,dimension=dimension,dtype='float32',
                    training_rows=len(raw),training_ids=sorted(np.unique(labels).tolist()),training_id_count=len(counts),
                    pair_counts=dict(same=same_count,different=len(raw)*(len(raw)-1)-same_count),
                    center_max_difference=error,spectrum=spectrum,
                    selection='ridge 0.1 fixed before this replay; no calibration or test read by this script',
                    scope='global head fit only; not a full training/inference replay or new quality evaluation',
                    sources={key:dict(path=str(path),sha256=sha(path)) for key,path in inputs.items()},
                    artifact=dict(path=target.name,sha256=sha(target)),
                    runtime=dict(python=platform.python_version(),numpy=np.__version__),seconds=time.perf_counter()-started)
    (output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def self_test():
    """Small synthetic alignment/core check; never presented as the 400-ID fit."""
    rng = np.random.default_rng(20260928)
    features = rng.normal(size=(12, 5)).astype(np.float32)
    rows = [dict(image_id=f'image-{i}', source_row=i) for i in range(12)]
    metadata = [dict(row, vehicle_id=str(i // 3)) for i,row in enumerate(rows)]
    center = unit(features).astype(np.float64).mean(0).astype(np.float32)
    raw, labels, _, _ = aligned_training(features, rows, metadata[::-1], center, (12, 5), 4)
    assert labels.tolist() == [str(i // 3) for i in range(12)]
    same, different = pair_covariances(raw, labels)
    transform, spectrum = psd_metric(same, different, RIDGE)
    assert spectrum['rank'] > 0 and np.isfinite(transform.astype(np.float32)).all()
    bad_inputs = []
    missing = [dict(r) for r in metadata];missing[0]['image_id']='wrong-image';bad_inputs.append((features,rows,missing,center))
    wrong_source = [dict(r) for r in metadata];wrong_source[0]['source_row']=100;bad_inputs.append((features,rows,wrong_source,center))
    wrong_id = [dict(r) for r in rows];wrong_id[0]['vehicle_id']='wrong-id';bad_inputs.append((features,wrong_id,metadata,center))
    duplicate = [dict(r) for r in rows];duplicate[0]=duplicate[1];bad_inputs.append((features,duplicate,metadata,center))
    nan = features.copy();nan[0,0]=np.nan;bad_inputs.append((nan,rows,metadata,center))
    bad_inputs.append((features,rows,metadata,center+1))
    for values in bad_inputs:
        try:
            aligned_training(*values, shape=(12, 5), expected_ids=4)
        except ValueError:
            continue
        raise AssertionError('Corrupt training fixture passed')
    return dict(synthetic_fixture=True,rows=12,dimension=5,identities=4,negative_controls=len(bad_inputs),passed=True,
                real_training_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-features',type=Path,default=Path('replay/training_features'))
    parser.add_argument('--training-rows',type=Path,default=Path('training/training_cpu_rows.jsonl'))
    parser.add_argument('--center-weights',type=Path,default=Path('weights/chel_v1.npz'))
    parser.add_argument('--output',type=Path,default=Path('replay/global_metric'))
    parser.add_argument('--expected-rows',type=int,default=2500)
    parser.add_argument('--expected-ids',type=int,default=400)
    parser.add_argument('--dimension',type=int,default=768)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    result = self_test() if args.self_test else fit(
        args.training_features, args.training_rows, args.center_weights, args.output,
        args.expected_rows, args.expected_ids, args.dimension)
    print(json.dumps(result,indent=2))
