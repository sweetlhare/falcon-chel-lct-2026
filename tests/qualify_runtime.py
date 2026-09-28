"""Read-only model/input qualification with a disposable local API and SQLite DB.

Requires separately supplied model, organizer archive and matching inference outputs.
Never starts a public listener or connects to an existing gallery. No quality claim.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import Model, sha
from falcon.audit import image_path


def compare_vectors(actual, expected, tolerance=1e-4):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid vector shape or nonfinite values')
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise ValueError(f'Vector drift {error} exceeds {tolerance}')
    return error


def check_binding(expected, actual):
    if hashlib.sha256(expected).digest() != hashlib.sha256(actual).digest():
        raise ValueError('Output model manifest differs from supplied model')


def controls():
    a = np.eye(3, dtype=np.float32)
    assert compare_vectors(a.copy(), a) == 0
    check_binding(b'manifest', b'manifest')
    calls = [lambda: compare_vectors(a + .01, a), lambda: compare_vectors(a[::-1], a),
             lambda: compare_vectors(a * np.nan, a),
             lambda: compare_vectors(a, a * np.nan),
             lambda: check_binding(b'manifest', b'wrong-model')]
    for call in calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError('Known corruption passed')
    return {'good_controls': 2, 'corrupt_controls_rejected': len(calls)}


def request(base, path, expected=200):
    try:
        with urllib.request.urlopen(base + path, timeout=120) as response:
            code, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        code, raw = error.code, error.read()
    if code != expected:
        raise AssertionError(f'Expected HTTP {expected}, received {code}')
    return json.loads(raw)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_ready(base, process):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Qualification subprocess exited; inspect temporary logs')
        try:
            return request(base, '/health')
        except (OSError, AssertionError):
            time.sleep(.1)
    raise TimeoutError('Qualification service did not become ready')


def api_check(weights, archive, outputs, rows, embeddings, threshold):
    nq = len(rows['query'])
    scores = embeddings[:nq] @ embeddings[nq:].T
    selections = []
    for kind, condition in [('accepted', scores.max(1) >= threshold),
                            ('refused', scores.max(1) < threshold)]:
        indices = np.flatnonzero(condition)
        if len(indices):
            selections.append((kind, int(indices[0])))
    with tempfile.TemporaryDirectory(prefix='falcon-qualification-') as directory:
        temp = Path(directory)
        database_port, api_port = free_port(), free_port()
        while api_port == database_port:
            api_port = free_port()
        env = dict(os.environ, FALCON_WEIGHTS=str(weights), FALCON_OUTPUT=str(outputs),
                   FALCON_ARCHIVE=str(archive), FALCON_THREADS='2',
                   FALCON_DB=str(temp / 'gallery.sqlite'), FALCON_DB_PORT=str(database_port),
                   FALCON_DB_URL=f'http://127.0.0.1:{database_port}')
        database = api = None
        with (temp / 'database.log').open('w') as db_log, (temp / 'api.log').open('w') as api_log:
            try:
                database = subprocess.Popen([sys.executable, '-c',
                    "import os; from db_service import GalleryServer; "
                    "GalleryServer(('127.0.0.1', int(os.environ['FALCON_DB_PORT'])), "
                    "os.environ['FALCON_DB']).serve_forever()"],
                                            cwd=ROOT, env=env, stdout=db_log, stderr=db_log)
                wait_ready(f'http://127.0.0.1:{database_port}', database)
                api = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api:app',
                                        '--host', '127.0.0.1', '--port', str(api_port)],
                                       cwd=ROOT, env=env, stdout=api_log, stderr=api_log)
                base = f'http://127.0.0.1:{api_port}'
                health = wait_ready(base, api)
                assert health['gallery_size'] == len(rows['gallery'])
                gallery_ids = [r['image_id'] for r in rows['gallery']]
                cases = []
                for kind, index in selections:
                    identity = rows['query'][index]['image_id']
                    answer = request(base, '/search-example/' + urllib.parse.quote(identity, safe=''))
                    expected = np.argsort(-scores[index], kind='stable')[:10]
                    assert [x['image_id'] for x in answer['ranking']] == [gallery_ids[j] for j in expected]
                    assert answer['refused'] == (kind == 'refused')
                    assert answer['source'] == 'fresh encoding of the bound example image'
                    target = answer['ranking'][0]['image_id']
                    path = '/explain-example/' + urllib.parse.quote(identity, safe='') + '?'
                    request(base, path + urllib.parse.urlencode({'target_id': target, 'search_token': 'unknown'}), 409)
                    cases.append({'kind': kind, 'exact_ranking': True, 'fresh_encoding': True})
                return {'passed': True, 'gallery_size': health['gallery_size'], 'cases': cases,
                        'both_acceptance_cases_available': len(cases) == 2,
                        'unknown_tokens_rejected': len(cases), 'disposable_database': True,
                        'api_bind': '127.0.0.1', 'database_bind': '127.0.0.1', 'existing_services_contacted': False}
            except Exception:
                # Logs remain in the report stream; no image bytes or embeddings are printed.
                for path in [temp / 'api.log', temp / 'database.log']:
                    if path.exists():
                        print(path.read_text()[-5000:], file=sys.stderr)
                raise
            finally:
                for process in [api, database]:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)


def main(args):
    result = {'controls': controls(), 'scope': 'Eight-image CPU parity and fresh disposable API; not full training, hidden-test quality or plate coverage.'}
    if args.self_test:
        print(json.dumps(result, indent=2))
        return
    if any(value is None for value in [args.weights, args.archive, args.outputs, args.report]):
        raise ValueError('Provide --weights, --archive, --outputs and --report')
    weights, archive, outputs, report = [p.resolve() for p in [args.weights, args.archive, args.outputs, args.report]]
    if report.exists():
        raise FileExistsError(report)
    if any(report.is_relative_to(p) for p in [weights, outputs]):
        raise ValueError('Report must be outside immutable artifact directories')
    start = time.monotonic()
    check_binding((weights / 'manifest.json').read_bytes(), (outputs / 'model_manifest.json').read_bytes())
    recorded = json.loads((outputs / 'report.json').read_text())
    for name in ['rows.json', 'embeddings.npy', 'redactions.json', 'pooling_weights.npy']:
        if recorded['hashes'].get(name) != sha(outputs / name):
            raise ValueError('Recorded output hash mismatch: ' + name)
    rows = json.loads((outputs / 'rows.json').read_text())
    if min(len(rows['query']), len(rows['gallery'])) < 4:
        raise ValueError('Need at least four query and four gallery rows')
    all_rows = rows['query'] + rows['gallery']
    indices = list(range(4)) + list(range(len(rows['query']), len(rows['query']) + 4))
    embeddings = np.load(outputs / 'embeddings.npy', allow_pickle=False)
    masks = {r['image_id']: r['redactions'] for r in json.loads((outputs / 'redactions.json').read_text())['records']}
    model = Model(weights, 2)
    threshold = model.threshold
    tensors = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for index in indices:
            row = all_rows[index]
            with Image.open(io.BytesIO(z.read(image_path(row['image_id'], names)))) as im:
                tensor, _, _ = model.prepare(im, [float(row[k]) for k in ['x', 'y', 'w', 'h']], masks[row['image_id']])
            tensors.append(tensor)
    actual, attention = model.encode_tensors(tensors)
    error = compare_vectors(actual, embeddings[indices])
    cosine_error = float(np.max(np.abs(actual @ actual.T - embeddings[indices] @ embeddings[indices].T)))
    assert cosine_error <= 1e-4 and attention.shape == (8, 64)
    assert np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5)
    result['parity'] = {'images': 8, 'selection': 'First four query and first four gallery rows',
                        'embedding_max_abs_error': error, 'cosine_max_abs_error': cosine_error,
                        'tolerance': 1e-4, 'uses_frozen_redactions': True}
    del model, tensors
    gc.collect()
    result['api'] = api_check(weights, archive, outputs, rows, embeddings, threshold)
    result.update(passed=True, seconds=time.monotonic()-start, weights_manifest_sha256=sha(weights/'manifest.json'),
                  output_report_sha256=sha(outputs/'report.json'),
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py'))},
                  source_revision=args.source_revision)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['weights', 'archive', 'outputs', 'report']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--source-revision', default='unrecorded')
    parser.add_argument('--self-test', action='store_true')
    main(parser.parse_args())
