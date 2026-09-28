"""Qualify public API with supplied weights and synthetic images, without a dataset.

Uses a temporary localhost API and SQLite gallery. Never contacts existing services.
This checks functionality, not vehicle retrieval quality or plate-mask coverage.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qualify_runtime import free_port, wait_ready


def request(base, path, payload=None, expected=200):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    if status != expected:
        raise AssertionError(f'{path}: expected HTTP {expected}, received {status}: {raw[:300]!r}')
    return json.loads(raw)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def synthetic_requests(seed):
    rng = np.random.default_rng(seed)
    requests = []
    for _ in range(2):
        # Synthetic textures have no ground-truth vehicle identity or semantics.
        pixels = rng.integers(0, 256, (128, 160, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='PNG')
        requests.append(dict(image_base64=base64.b64encode(buffer.getvalue()).decode(),
                             bbox=[0, 0, 160, 128], top_n=10))
    return requests


def qualify(weights):
    start = time.monotonic()
    seed = 20260928
    inputs = synthetic_requests(seed)
    result = {'seed': seed, 'synthetic_only': True, 'uses_organizer_data': False,
              'uses_saved_outputs': False, 'quality_claim': False}
    with tempfile.TemporaryDirectory(prefix='falcon-empty-') as directory:
        temporary = Path(directory)
        database_port, api_port = free_port(), free_port()
        while api_port == database_port:
            api_port = free_port()
        env = dict(os.environ, FALCON_WEIGHTS=str(weights),
                   FALCON_OUTPUT=str(temporary / 'absent_outputs'),
                   FALCON_ARCHIVE=str(temporary / 'absent_dataset.zip'),
                   FALCON_THREADS='2', FALCON_DB=str(temporary / 'gallery.sqlite'),
                   FALCON_DB_PORT=str(database_port),
                   FALCON_DB_URL=f'http://127.0.0.1:{database_port}')
        assert not Path(env['FALCON_OUTPUT']).exists()
        assert not Path(env['FALCON_ARCHIVE']).exists()
        database = api = None
        base = f'http://127.0.0.1:{api_port}'
        with (temporary / 'database.log').open('w') as db_log, (temporary / 'api.log').open('w') as api_log:
            def start_api():
                return subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api:app',
                                         '--host', '127.0.0.1', '--port', str(api_port)],
                                        cwd=ROOT, env=env, stdout=api_log, stderr=api_log)

            try:
                database = subprocess.Popen([sys.executable, '-c',
                    "import os; from db_service import GalleryServer; "
                    "GalleryServer(('127.0.0.1', int(os.environ['FALCON_DB_PORT'])), "
                    "os.environ['FALCON_DB']).serve_forever()"],
                    cwd=ROOT, env=env, stdout=db_log, stderr=db_log)
                wait_ready(f'http://127.0.0.1:{database_port}', database)
                api = start_api()
                health = wait_ready(base, api)
                assert health['gallery_size'] == 0
                assert request(base, '/examples') == []
                request(base, '/search-example/nonexistent', expected=404)
                encoded = request(base, '/encode', inputs[0])
                vector = np.asarray(encoded['embedding'])
                assert vector.shape == (1024,) and np.isfinite(vector).all()
                assert abs(np.linalg.norm(vector) - 1) < 1e-5
                assert np.asarray(encoded['pooling_weights']).shape == (8, 8)
                empty = request(base, '/search', inputs[0])
                assert empty['refused'] and empty['ranking'] == [] and empty['gallery_size'] == 0
                result.update(empty_start=True, examples_empty=True, missing_example_404=True,
                              encode_dimension=1024, encode_norm=float(np.linalg.norm(vector)),
                              empty_search_refused=True)
                for index, payload in enumerate(inputs):
                    assert request(base, '/gallery', dict(payload, image_id=f'synthetic-{index}'))['stored']
                assert request(base, '/health')['gallery_size'] == 2
                found = request(base, '/search', inputs[0])
                assert found['ranking'][0]['image_id'] == 'synthetic-0'
                assert found['ranking'][0]['accepted'] and not found['refused']
                token = found['search_token']
                payload = dict(inputs[0], target_id='synthetic-0', search_token=token)
                request(base, '/explain', dict(payload, search_token='unknown'), 409)
                request(base, '/explain', dict(inputs[1], target_id='synthetic-0', search_token=token), 409)
                explanation = request(base, '/explain', payload)
                assert explanation['provenance']['verified_same_search']
                assert explanation['provenance']['search_token'] == token
                for name, value in found['provenance'].items():
                    assert explanation['provenance'][name] == value
                assert len(explanation['regions_xyxy']) == 16
                assert explanation['controls']['intervention_variants'] == 32
                assert explanation['controls']['full_gallery_rescored_for_every_variant']
                request(base, '/gallery', dict(inputs[1], image_id='synthetic-0'))
                request(base, '/explain', payload, 409)
                request(base, '/gallery', dict(inputs[0], image_id='synthetic-0'))
                stop(api)
                api = start_api()
                assert wait_ready(base, api)['gallery_size'] == 2
                again = request(base, '/search', inputs[0])
                assert [r['image_id'] for r in again['ranking']] == [r['image_id'] for r in found['ranking']]
                result.update(gallery_insert_count=2, selfsearch_top1=True,
                              selfsearch_score=found['ranking'][0]['score'], explanation_regions=16,
                              explanation_variants=32, provenance_matches_search=True,
                              search_reference=explanation['controls']['search_reference'],
                              negative_controls={'unknown_token_409': True, 'changed_input_409': True,
                                                 'changed_gallery_409': True},
                              api_restart_retains_gallery=True, api_restart_retains_ranking=True)
            except Exception:
                # Only these synthetic-test subprocess logs are printed on failure.
                for path in [temporary / 'database.log', temporary / 'api.log']:
                    if path.exists():
                        print(path.read_text()[-4000:], file=sys.stderr)
                raise
            finally:
                try:
                    stop(api)
                finally:
                    stop(database)
    result.update(passed=True, seconds=time.monotonic() - start,
                  api_bind='127.0.0.1', database_bind='127.0.0.1',
                  existing_services_contacted=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  weights_manifest_sha256=hashlib.sha256((weights / 'manifest.json').read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ['api.py', 'inference.py', 'explanation.py',
                                              'db_service.py', 'gallery_client.py',
                                              'tests/qualify_runtime.py']})
    return result


def main(args):
    if not __debug__:
        raise ValueError('Run without -O: assertions implement the acceptance checks')
    weights, report = args.weights.resolve(), args.report.resolve()
    if not (weights / 'manifest.json').is_file():
        raise ValueError('--weights must contain a real manifest.json and model files')
    if report.exists():
        raise FileExistsError(report)
    if report.is_relative_to(weights):
        raise ValueError('--report must be outside the weights directory')
    result = qualify(weights)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    main(parser.parse_args())
