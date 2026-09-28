"""Synthetic HTTP storage contract, persistence and frozen-search parity checks."""
import ast
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
import numpy as np
import gallery_client

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SEARCH_SOURCE = "def search_vector(embedding,top_n):\n    with sqlite3.connect(DB) as db:rows=db.execute('SELECT id,embedding FROM gallery ORDER BY rowid').fetchall()\n    if not rows:return dict(candidates=[],ranking=[],refused=True,threshold=STATE['model'].threshold,gallery_size=0)\n    bank=np.stack([np.frombuffer(row[1],dtype='<f4') for row in rows]);scores=bank@embedding\n    order=np.argsort(-scores,kind='stable')[:top_n];threshold=STATE['model'].threshold\n    ranking=[dict(image_id=rows[i][0],score=float(scores[i]),accepted=bool(scores[i]>=threshold),contributions=decompose(embedding,bank[i])) for i in order]\n    return dict(candidates=[r for r in ranking if r['accepted']],ranking=ranking,refused=not any(r['accepted'] for r in ranking),\n                threshold=threshold,gallery_size=len(rows),score_kind='cosine similarity, not probability')"


def start_database(path, port):
    env = dict(os.environ, FALCON_DB=str(path), FALCON_DB_PORT=str(port))
    process = subprocess.Popen([sys.executable, str(ROOT / 'db_service.py')], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError('Database process exited')
        try:
            gallery_client.count()
            return process
        except gallery_client.GalleryUnavailable:
            time.sleep(.05)
    process.terminate()
    process.wait(timeout=5)
    raise TimeoutError('Database did not become ready')


def stop_database(process):
    process.terminate()
    process.wait(timeout=5)


def function_from(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.Module(body=[node], type_ignores=[])


class DatabaseContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / 'gallery.sqlite'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        gallery_client.ORIGIN = f'http://127.0.0.1:{self.port}'
        self.process = start_database(self.database, self.port)

    def tearDown(self):
        if self.process.poll() is None:
            stop_database(self.process)
        self.temp.cleanup()

    def raw_request(self, path, value):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            conn.request('POST', path, json.dumps(value).encode(), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    @staticmethod
    def record(identity='test', value=1.0):
        raw = struct.pack('<1024f', value, *([0.0] * 1023))
        return dict(id=identity, embedding=base64.b64encode(raw).decode(), metadata={'bbox': [1, 2, 3, 4]})

    def test_bytes_order_replacement_and_process_restart(self):
        records = [self.record('a'), self.record('b', .5)]
        self.assertEqual(self.raw_request('/upsert', {'records': records})[0], 200)
        self.assertEqual(gallery_client.count(), 2)
        gallery_client.upsert([('a', base64.b64decode(records[0]['embedding']), {'replacement': True})])
        before = gallery_client.rows()
        self.assertEqual([r[0] for r in before], ['b', 'a'])
        stop_database(self.process)
        self.process = start_database(self.database, self.port)
        self.assertEqual(gallery_client.rows(), before)
        with sqlite3.connect(self.database) as db:
            metadata = json.loads(db.execute('SELECT metadata FROM gallery WHERE id=?', ('a',)).fetchone()[0])
        self.assertEqual(metadata, {'replacement': True})

    def test_invalid_records_rejected_atomically(self):
        good = self.record()
        variants = [dict(good, id='../bad'), dict(good, id='x' * 129),
                    dict(good, embedding=good['embedding'][:-4]),
                    dict(good, embedding='!' * 5464), self.record(value=float('nan')),
                    self.record(value=float('inf')), dict(good, metadata=[]),
                    dict(good, metadata={'bad': float('nan')}),
                    dict(good, metadata={'big': 'x' * 65536})]
        for bad in variants:
            with self.subTest(record=str(bad)[:80]):
                status, _ = self.raw_request('/upsert', {'records': [dict(good, id='valid'), bad]})
                self.assertEqual(status, 422)
                self.assertEqual(gallery_client.count(), 0)
        self.assertEqual(self.raw_request('/upsert', {'records': [good, good]})[0], 422)
        self.assertEqual(gallery_client.count(), 0)

    def test_concurrent_writes_and_invalid_size(self):
        from concurrent.futures import ThreadPoolExecutor
        def insert(index):
            return self.raw_request('/upsert', {'records': [self.record(f'item-{index}')]})[0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(insert, range(12))), [200] * 12)
        self.assertEqual(gallery_client.count(), 12)
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            conn.request('POST', '/upsert', headers={'Content-Length': '20000001'})
            response = conn.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            conn.close()
        self.assertEqual(gallery_client.count(), 12)

    def test_search_math_matches_original_sqlite(self):
        rng = np.random.default_rng(72928)
        bank = rng.standard_normal((9, 1024)).astype(np.float32)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        bank[8] = bank[0]  # Stable tied ordering must be preserved too.
        records = [(f'id-{i}', row.astype('<f4').tobytes(), {'index': i}) for i, row in enumerate(bank)]
        gallery_client.upsert(records)
        old_db = Path(self.temp.name) / 'old.sqlite'
        with sqlite3.connect(old_db) as db:
            db.execute('CREATE TABLE gallery(id TEXT PRIMARY KEY, embedding BLOB, metadata TEXT)')
            db.executemany('INSERT OR REPLACE INTO gallery VALUES (?,?,?)', [(i, v, json.dumps(m)) for i, v, m in records])
        shared = {'np': np, 'STATE': {'model': SimpleNamespace(threshold=.23726077377796173)}}
        exec(compile(function_from(ROOT / 'inference.py', 'decompose'), '<decompose>', 'exec'), shared)
        original = dict(shared, sqlite3=sqlite3, DB=old_db)
        exec(compile(ast.parse(FROZEN_SEARCH_SOURCE), '<original>', 'exec'), original)
        candidate = dict(shared, gallery_client=gallery_client)
        exec(compile(function_from(ROOT / 'api.py', 'search_vector'), '<candidate>', 'exec'), candidate)
        for query in [bank[0], bank[3], np.zeros(1024, np.float32), -bank[0]]:
            self.assertEqual(original['search_vector'](query, 9), candidate['search_vector'](query, 9))
        answer = candidate['search_vector'](bank[0], 9)
        self.assertEqual(answer['ranking'][0]['image_id'], 'id-0')
        self.assertGreater(answer['ranking'][0]['score'], .999)


if __name__ == '__main__':
    unittest.main(verbosity=2)
