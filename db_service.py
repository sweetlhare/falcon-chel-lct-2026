"""Internal relational gallery service. No inference or third-party imports."""
import base64
import json
import math
import os
import re
import sqlite3
import struct
import threading
from contextlib import contextmanager, closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 20_000_000


def validate_record(record):
    if not isinstance(record, dict) or set(record) != {'id', 'embedding', 'metadata'}:
        raise ValueError('Record must contain id, embedding and metadata')
    identity = record['id']
    if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', identity):
        raise ValueError('Invalid gallery id')
    vector = record['embedding']
    if not isinstance(vector, str) or len(vector) != 5464:
        raise ValueError('Expected base64 of 1024 float32 values')
    raw = base64.b64decode(vector, validate=True)
    if len(raw) != 4096 or not all(math.isfinite(x) for x in struct.unpack('<1024f', raw)):
        raise ValueError('Embedding must contain 1024 finite float32 values')
    if not isinstance(record['metadata'], dict):
        raise ValueError('Metadata must be a JSON object')
    metadata = json.dumps(record['metadata'], ensure_ascii=False, allow_nan=False)
    if len(metadata.encode('utf-8')) > 65536:
        raise ValueError('Metadata exceeds 64 KiB')
    return identity, raw, metadata


class GalleryServer(ThreadingHTTPServer):
    def __init__(self, address, database):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS gallery (id TEXT PRIMARY KEY, embedding BLOB NOT NULL, metadata TEXT NOT NULL)')
        super().__init__(address, Handler)

    @contextmanager
    def transaction(self):
        if not self.lock.acquire(timeout=5):
            raise TimeoutError('Gallery busy')
        try:
            with closing(sqlite3.connect(self.database, timeout=5)) as db, db:
                yield db
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def reply(self, status, value):
        raw = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path not in ('/health', '/count', '/records'):
            return self.reply(404, {'detail': 'Unknown database endpoint'})
        try:
            with self.server.transaction() as db:
                if self.path == '/records':
                    rows = db.execute('SELECT id,embedding,metadata FROM gallery ORDER BY rowid').fetchall()
                    answer = {'records': [dict(id=r[0], embedding=base64.b64encode(r[1]).decode(), metadata=json.loads(r[2])) for r in rows]}
                else:
                    answer = {'status': 'ok', 'count': db.execute('SELECT COUNT(*) FROM gallery').fetchone()[0]}
            self.reply(200, answer)
        except (sqlite3.Error, TimeoutError):
            self.reply(503, {'detail': 'Gallery database unavailable'})

    def do_POST(self):
        if self.path != '/upsert':
            return self.reply(404, {'detail': 'Unknown database endpoint'})
        if self.headers.get('Transfer-Encoding'):
            return self.reply(400, {'detail': 'Content-Length required'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, {'detail': 'Invalid Content-Length'})
        if not 0 < size <= MAX_BODY:
            return self.reply(413, {'detail': 'Request must contain 1..20 MB'})
        try:
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or set(payload) != {'records'}:
                raise ValueError('Expected records object')
            records = payload['records']
            if not isinstance(records, list) or not 1 <= len(records) <= 1000:
                raise ValueError('Expected 1..1000 records')
            rows = [validate_record(record) for record in records]
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError('Duplicate id in transaction')
        except (ValueError, TypeError, RecursionError):
            return self.reply(422, {'detail': 'Invalid gallery records'})
        try:
            with self.server.transaction() as db:
                db.executemany('INSERT OR REPLACE INTO gallery VALUES (?,?,?)', rows)
            self.reply(200, {'stored': len(rows)})
        except (sqlite3.Error, TimeoutError):
            self.reply(503, {'detail': 'Gallery database unavailable'})


if __name__ == '__main__':
    GalleryServer(('0.0.0.0', int(os.environ.get('FALCON_DB_PORT', '8002'))),
                  os.environ.get('FALCON_DB', '/storage/gallery.sqlite')).serve_forever()
